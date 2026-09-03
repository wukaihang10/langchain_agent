import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import AgentMiddleware, HumanInTheLoopMiddleware
from langchain_core.language_models.fake_chat_models import FakeMessagesListChatModel
from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langgraph.graph import END, START, MessagesState, StateGraph
from langgraph.types import interrupt

from langchain_agent.app.session_continuation import (
    ContinuationAction,
    ContinuationRequest,
    ContinuationStatus,
    SessionContinuation,
)
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.middleware.turn_recovery import TurnRecoveryMiddleware
from langchain_agent.harness.permissions.models import (
    ToolCategory,
    ToolPolicy,
    ToolRisk,
)
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry
from langchain_agent.persistence.checkpoints import open_checkpointer
from langchain_agent.persistence.sessions import SessionStore


UNSAFE_POLICY = ToolPolicy(
    category=ToolCategory.WRITE,
    idempotent=False,
    side_effect=True,
    risk=ToolRisk.HIGH,
)
SAFE_POLICY = ToolPolicy(
    category=ToolCategory.READ,
    idempotent=True,
    side_effect=False,
    risk=ToolRisk.LOW,
)

TOOL_BEHAVIOR = {
    "write_calls": 0,
    "completed_calls": 0,
    "safe_calls": 0,
    "unsafe_calls": 0,
    "fail_safe": False,
    "fail_unsafe": False,
}


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


class CountingToolCallingFakeModel(ToolCallingFakeModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        return super()._generate(messages, stop, run_manager, **kwargs)


class FailingToolCallingFakeModel(ToolCallingFakeModel):
    calls: int = 0

    def _generate(self, messages, stop=None, run_manager=None, **kwargs):
        self.calls += 1
        raise ConnectionError("model disconnected")


class LifecycleProbeMiddleware(AgentMiddleware):
    def __init__(self):
        self.after_model_calls = 0
        self.after_agent_calls = 0

    def after_model(self, state, runtime):
        self.after_model_calls += 1

    def after_agent(self, state, runtime):
        self.after_agent_calls += 1


@tool
def write_note(path: str) -> str:
    """Write a test note."""

    TOOL_BEHAVIOR["write_calls"] += 1
    return f"wrote {path}"


@tool
def completed_lookup(query: str) -> str:
    """Return a completed sibling result."""

    TOOL_BEHAVIOR["completed_calls"] += 1
    return f"completed {query}"


@tool
def recoverable_lookup(query: str) -> str:
    """Return a replay-safe result unless failure is enabled."""

    TOOL_BEHAVIOR["safe_calls"] += 1
    if TOOL_BEHAVIOR["fail_safe"]:
        raise ConnectionError("safe lookup disconnected")
    return f"safe {query}"


@tool
def uncertain_create(name: str) -> str:
    """Create an external item unless failure is enabled."""

    TOOL_BEHAVIOR["unsafe_calls"] += 1
    if TOOL_BEHAVIOR["fail_unsafe"]:
        raise ConnectionError("create disconnected")
    return f"created {name}"


def approval_graph(checkpointer):
    def request_tool(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "write_file",
                            "args": {"path": "notes.txt"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def review_tool(_state):
        result = interrupt(
            {
                "action_requests": [
                    {
                        "name": "write_file",
                        "args": {"path": "notes.txt"},
                    }
                ],
                "review_configs": [
                    {"allowed_decisions": ["approve", "reject"]}
                ],
            }
        )
        return {
            "messages": [
                ToolMessage(
                    content=str(result),
                    tool_call_id="call-1",
                )
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("request_tool", request_tool)
    builder.add_node("review_tool", review_tool)
    builder.add_edge(START, "request_tool")
    builder.add_edge("request_tool", "review_tool")
    builder.add_edge("review_tool", END)
    return builder.compile(checkpointer=checkpointer)


def tool_graph(checkpointer, *, fail: bool):
    def request_tool(_state):
        return {
            "messages": [
                AIMessage(
                    content="",
                    tool_calls=[
                        {
                            "name": "lookup",
                            "args": {"query": "checkpoint"},
                            "id": "call-1",
                            "type": "tool_call",
                        }
                    ],
                )
            ]
        }

    def run_tool(_state):
        if fail:
            raise ConnectionError("tool process stopped")
        return {
            "messages": [
                ToolMessage(content="found", tool_call_id="call-1")
            ]
        }

    builder = StateGraph(MessagesState)
    builder.add_node("request_tool", request_tool)
    builder.add_node("tools", run_tool)
    builder.add_edge(START, "request_tool")
    builder.add_edge("request_tool", "tools")
    builder.add_edge("tools", END)
    return builder.compile(checkpointer=checkpointer)


def hitl_agent(checkpointer, responses):
    return create_agent(
        model=ToolCallingFakeModel(responses=responses),
        tools=[write_note],
        middleware=[HumanInTheLoopMiddleware(interrupt_on={write_note.name: True})],
        checkpointer=checkpointer,
    )


def recovery_agent(checkpointer, responses, tools):
    return create_agent(
        model=ToolCallingFakeModel(responses=responses),
        tools=tools,
        middleware=[TurnRecoveryMiddleware()],
        context_schema=AgentContext,
        checkpointer=checkpointer,
    )


def agent_runtime(session, root):
    return SimpleNamespace(
        session=session,
        invoke_config={"configurable": {"thread_id": session.thread_id}},
        context=AgentContext(
            repository_path=str(root),
            repository_knowledge=object(),
        ),
    )


def runtime(session):
    return SimpleNamespace(
        session=session,
        invoke_config={"configurable": {"thread_id": session.thread_id}},
        context=None,
    )


class DurableSessionContinuationTests(unittest.IsolatedAsyncioTestCase):
    def setUp(self):
        TOOL_BEHAVIOR.update(
            write_calls=0,
            completed_calls=0,
            safe_calls=0,
            unsafe_calls=0,
            fail_safe=False,
            fail_unsafe=False,
        )

    async def test_create_agent_hitl_payload_is_recognized_after_sqlite_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="durable", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}
            tool_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": write_note.name,
                        "args": {"path": "notes.txt"},
                        "id": "call-1",
                        "type": "tool_call",
                    }
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = hitl_agent(checkpointer, [tool_call])
                await agent.ainvoke(
                    {"messages": [HumanMessage(content="write notes")]},
                    config=config,
                )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = hitl_agent(
                    checkpointer,
                    [AIMessage(content="The write was rejected.")],
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {write_note.name: UNSAFE_POLICY}
                    ),
                    session_store=store,
                )
                inspection = await continuation.inspect(runtime(session))

                self.assertEqual(
                    inspection.status,
                    ContinuationStatus.WAITING_HUMAN,
                )
                self.assertEqual(
                    inspection.interrupts[0].value["action_requests"][0]["name"],
                    write_note.name,
                )

                result = await continuation.execute(
                    runtime(session),
                    ContinuationRequest(
                        action=ContinuationAction.ANSWER_INTERRUPT,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        decisions=({"type": "reject", "message": "not now"},),
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)

    async def test_hitl_turn_can_be_terminated_after_sqlite_reopens(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="terminate-hitl", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}
            tool_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": write_note.name,
                        "args": {"path": "notes.txt"},
                        "id": "call-1",
                        "type": "tool_call",
                    },
                    {
                        "name": write_note.name,
                        "args": {"path": "other.txt"},
                        "id": "call-2",
                        "type": "tool_call",
                    },
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = create_agent(
                    model=ToolCallingFakeModel(responses=[tool_call]),
                    tools=[write_note],
                    middleware=[
                        TurnRecoveryMiddleware(),
                        HumanInTheLoopMiddleware(
                            interrupt_on={write_note.name: True}
                        ),
                    ],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                await agent.ainvoke(
                    {"messages": [HumanMessage(content="write notes")]},
                    config=config,
                    context=agent_runtime(session, root).context,
                )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                model = CountingToolCallingFakeModel(
                    responses=[AIMessage(content="fresh turn response")]
                )
                agent = create_agent(
                    model=model,
                    tools=[write_note],
                    middleware=[
                        TurnRecoveryMiddleware(),
                        HumanInTheLoopMiddleware(
                            interrupt_on={write_note.name: True}
                        ),
                    ],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {write_note.name: UNSAFE_POLICY}
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)

                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.TERMINATE_TURN,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["write_calls"], 0)
                self.assertEqual(model.calls, 0)
                tool_results = {
                    message.tool_call_id: message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id in {"call-1", "call-2"}
                }
                self.assertEqual(set(tool_results), {"call-1", "call-2"})
                self.assertTrue(
                    all(message.status == "error" for message in tool_results.values())
                )
                terminal = result.value["messages"][-1]
                self.assertIsInstance(terminal, AIMessage)
                self.assertFalse(terminal.tool_calls)
                self.assertEqual(
                    terminal.additional_kwargs["recovery"]["action"],
                    "TERMINATE_TURN",
                )

                next_result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.START_TURN,
                        observed_checkpoint_id=result.inspection.checkpoint_id,
                        message="start something new",
                    ),
                )
                self.assertEqual(
                    next_result.inspection.status,
                    ContinuationStatus.READY,
                )
                self.assertEqual(
                    next_result.value["messages"][-1].content,
                    "fresh turn response",
                )
                self.assertEqual(model.calls, 1)

    async def test_pending_model_termination_uses_normal_completion_hooks(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="terminate-model", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = create_agent(
                    model=FailingToolCallingFakeModel(
                        responses=[AIMessage(content="unused")]
                    ),
                    tools=[],
                    middleware=[TurnRecoveryMiddleware()],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                with self.assertRaisesRegex(ConnectionError, "model disconnected"):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="do work")]},
                        config=config,
                        context=agent_runtime(session, root).context,
                    )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                model = CountingToolCallingFakeModel(
                    responses=[AIMessage(content="must not run")]
                )
                probe = LifecycleProbeMiddleware()
                agent = create_agent(
                    model=model,
                    tools=[],
                    middleware=[TurnRecoveryMiddleware(), probe],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)
                self.assertEqual(inspection.status, ContinuationStatus.RESUMABLE)

                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.TERMINATE_TURN,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(model.calls, 0)
                self.assertEqual(probe.after_model_calls, 1)
                self.assertEqual(probe.after_agent_calls, 1)
                self.assertEqual(
                    result.value["messages"][-1].additional_kwargs["recovery"][
                        "action"
                    ],
                    "TERMINATE_TURN",
                )

    async def test_hitl_interrupt_survives_reopening_sqlite_and_can_be_answered(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="durable", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = approval_graph(checkpointer)
                await graph.ainvoke(
                    {"messages": [HumanMessage(content="write notes")]},
                    config=config,
                )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = approval_graph(checkpointer)
                continuation = SessionContinuation(
                    agent=graph,
                    policy_registry=ToolPolicyRegistry(
                        {"write_file": UNSAFE_POLICY}
                    ),
                    session_store=store,
                )
                inspection = await continuation.inspect(runtime(session))

                self.assertEqual(
                    inspection.status,
                    ContinuationStatus.WAITING_HUMAN,
                )
                self.assertEqual(
                    inspection.allowed_actions,
                    {
                        ContinuationAction.ANSWER_INTERRUPT,
                        ContinuationAction.TERMINATE_TURN,
                    },
                )

                result = await continuation.execute(
                    runtime(session),
                    ContinuationRequest(
                        action=ContinuationAction.ANSWER_INTERRUPT,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        decisions=({"type": "reject", "message": "not now"},),
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(
                    result.value["messages"][-1].tool_call_id,
                    "call-1",
                )

    async def test_safe_tool_failure_survives_reopening_sqlite_and_can_continue(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="durable", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = tool_graph(checkpointer, fail=True)
                with self.assertRaisesRegex(ConnectionError, "tool process stopped"):
                    await graph.ainvoke(
                        {"messages": [HumanMessage(content="look it up")]},
                        config=config,
                    )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = tool_graph(checkpointer, fail=False)
                continuation = SessionContinuation(
                    agent=graph,
                    policy_registry=ToolPolicyRegistry({"lookup": SAFE_POLICY}),
                    session_store=store,
                )
                inspection = await continuation.inspect(runtime(session))

                self.assertEqual(inspection.status, ContinuationStatus.RESUMABLE)
                self.assertEqual(
                    inspection.allowed_actions,
                    {
                        ContinuationAction.CONTINUE,
                        ContinuationAction.TERMINATE_TURN,
                    },
                )

                result = await continuation.execute(
                    runtime(session),
                    ContinuationRequest(
                        action=ContinuationAction.CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(result.value["messages"][-1].content, "found")

    async def test_unsafe_failure_reopens_as_outcome_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="durable", repository_path=str(root))
            config = {"configurable": {"thread_id": session.thread_id}}

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = tool_graph(checkpointer, fail=True)
                with self.assertRaisesRegex(ConnectionError, "tool process stopped"):
                    await graph.ainvoke(
                        {"messages": [HumanMessage(content="look it up")]},
                        config=config,
                    )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                graph = tool_graph(checkpointer, fail=False)
                continuation = SessionContinuation(
                    agent=graph,
                    policy_registry=ToolPolicyRegistry({"lookup": UNSAFE_POLICY}),
                    session_store=store,
                )

                inspection = await continuation.inspect(runtime(session))

                self.assertEqual(
                    inspection.status,
                    ContinuationStatus.OUTCOME_UNKNOWN,
                )
                self.assertEqual(
                    inspection.allowed_actions,
                    {
                        ContinuationAction.CONTINUE,
                        ContinuationAction.TERMINATE_TURN,
                    },
                )

    async def test_automatic_unknown_result_preserves_completed_sibling(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="durable", repository_path=str(root))
            run_config = {"configurable": {"thread_id": session.thread_id}}
            TOOL_BEHAVIOR["fail_unsafe"] = True
            tool_calls = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": completed_lookup.name,
                        "args": {"query": "one"},
                        "id": "completed-id-1",
                    },
                    {
                        "name": completed_lookup.name,
                        "args": {"query": "two"},
                        "id": "completed-id-2",
                    },
                    {
                        "name": uncertain_create.name,
                        "args": {"name": "item"},
                        "id": "unsafe-id",
                    },
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [tool_calls],
                    [completed_lookup, uncertain_create],
                )
                with self.assertRaisesRegex(ConnectionError, "create disconnected"):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="run both")]},
                        config=run_config,
                        context=agent_runtime(session, root).context,
                    )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [AIMessage(content="Recovery complete.")],
                    [completed_lookup, uncertain_create],
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {
                            completed_lookup.name: SAFE_POLICY,
                            uncertain_create.name: UNSAFE_POLICY,
                        }
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)

                self.assertEqual(inspection.status, ContinuationStatus.OUTCOME_UNKNOWN)
                self.assertEqual(
                    [call.id for call in inspection.unresolved_tool_calls],
                    ["unsafe-id"],
                )
                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["completed_calls"], 2)
                self.assertEqual(TOOL_BEHAVIOR["unsafe_calls"], 1)
                recovered = next(
                    message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "unsafe-id"
                )
                self.assertEqual(recovered.status, "error")
                self.assertEqual(
                    recovered.additional_kwargs["recovery"],
                    {
                        "generated": True,
                        "kind": "OUTCOME_UNKNOWN",
                        "outcome": "outcome_unknown",
                    },
                )
                self.assertIn("Do not infer success or failure", recovered.content)

            async with open_checkpointer(checkpoint_path) as checkpointer:
                final_agent = recovery_agent(checkpointer, [], [])
                final_continuation = SessionContinuation(
                    agent=final_agent,
                    policy_registry=ToolPolicyRegistry(),
                    session_store=store,
                )
                final_inspection = await final_continuation.inspect(
                    agent_runtime(session, root)
                )
                self.assertEqual(final_inspection.status, ContinuationStatus.READY)
                persisted = await final_agent.aget_state(
                    {"configurable": {"thread_id": session.thread_id}}
                )
                persisted_recovery = next(
                    message
                    for message in persisted.values["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "unsafe-id"
                )
                self.assertEqual(
                    persisted_recovery.additional_kwargs["recovery"],
                    {
                        "generated": True,
                        "kind": "OUTCOME_UNKNOWN",
                        "outcome": "outcome_unknown",
                    },
                )

    async def test_safe_failure_retries_while_unsafe_failure_is_recorded_unknown(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="mixed", repository_path=str(root))
            run_config = {"configurable": {"thread_id": session.thread_id}}
            TOOL_BEHAVIOR["fail_safe"] = True
            TOOL_BEHAVIOR["fail_unsafe"] = True
            tool_calls = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": recoverable_lookup.name,
                        "args": {"query": "one"},
                        "id": "safe-id",
                    },
                    {
                        "name": uncertain_create.name,
                        "args": {"name": "item"},
                        "id": "unsafe-id",
                    },
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [tool_calls],
                    [recoverable_lookup, uncertain_create],
                )
                with self.assertRaises(Exception):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="run both")]},
                        config=run_config,
                        context=agent_runtime(session, root).context,
                    )

            TOOL_BEHAVIOR["fail_safe"] = False
            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [AIMessage(content="Mixed recovery complete.")],
                    [recoverable_lookup, uncertain_create],
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {
                            recoverable_lookup.name: SAFE_POLICY,
                            uncertain_create.name: UNSAFE_POLICY,
                        }
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)
                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["safe_calls"], 2)
                self.assertEqual(TOOL_BEHAVIOR["unsafe_calls"], 1)
                unknown = next(
                    message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "unsafe-id"
                )
                self.assertEqual(unknown.status, "error")
                self.assertIn("No retry was performed", unknown.content)

    async def test_mixed_failed_tools_are_terminated_without_reexecution(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="terminate-mixed", repository_path=str(root))
            run_config = {"configurable": {"thread_id": session.thread_id}}
            TOOL_BEHAVIOR["fail_safe"] = True
            TOOL_BEHAVIOR["fail_unsafe"] = True
            tool_calls = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": completed_lookup.name,
                        "args": {"query": "zero"},
                        "id": "completed-id",
                    },
                    {
                        "name": recoverable_lookup.name,
                        "args": {"query": "one"},
                        "id": "safe-id",
                    },
                    {
                        "name": uncertain_create.name,
                        "args": {"name": "item"},
                        "id": "unsafe-id",
                    },
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [tool_calls],
                    [completed_lookup, recoverable_lookup, uncertain_create],
                )
                with self.assertRaises(Exception):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="run both")]},
                        config=run_config,
                        context=agent_runtime(session, root).context,
                    )

            safe_calls = TOOL_BEHAVIOR["safe_calls"]
            unsafe_calls = TOOL_BEHAVIOR["unsafe_calls"]
            completed_calls = TOOL_BEHAVIOR["completed_calls"]
            async with open_checkpointer(checkpoint_path) as checkpointer:
                model = CountingToolCallingFakeModel(
                    responses=[AIMessage(content="must not run")]
                )
                agent = create_agent(
                    model=model,
                    tools=[completed_lookup, recoverable_lookup, uncertain_create],
                    middleware=[TurnRecoveryMiddleware()],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {
                            completed_lookup.name: SAFE_POLICY,
                            recoverable_lookup.name: SAFE_POLICY,
                            uncertain_create.name: UNSAFE_POLICY,
                        }
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)
                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.TERMINATE_TURN,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["safe_calls"], safe_calls)
                self.assertEqual(TOOL_BEHAVIOR["unsafe_calls"], unsafe_calls)
                self.assertEqual(
                    TOOL_BEHAVIOR["completed_calls"],
                    completed_calls,
                )
                self.assertEqual(model.calls, 0)
                recovered = {
                    message.tool_call_id: message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id in {"safe-id", "unsafe-id"}
                }
                self.assertEqual(set(recovered), {"safe-id", "unsafe-id"})
                completed = next(
                    message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "completed-id"
                )
                self.assertEqual(completed.content, "completed zero")
                self.assertNotIn("recovery", completed.additional_kwargs)
                self.assertEqual(
                    recovered["safe-id"].additional_kwargs["recovery"]["outcome"],
                    "cancelled",
                )
                self.assertEqual(
                    recovered["unsafe-id"].additional_kwargs["recovery"]["outcome"],
                    "outcome_unknown",
                )
                self.assertEqual(
                    result.value["messages"][-1].additional_kwargs["recovery"][
                        "action"
                    ],
                    "TERMINATE_TURN",
                )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                final_agent = recovery_agent(checkpointer, [], [])
                final = await final_agent.aget_state(run_config)
                self.assertFalse(final.next)
                self.assertEqual(
                    final.values["messages"][-1].additional_kwargs["recovery"][
                        "action"
                    ],
                    "TERMINATE_TURN",
                )

    async def test_unsafe_call_is_not_retried_when_tool_becomes_available(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="retry", repository_path=str(root))
            run_config = {"configurable": {"thread_id": session.thread_id}}
            TOOL_BEHAVIOR["fail_unsafe"] = True
            tool_calls = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": completed_lookup.name,
                        "args": {"query": "one"},
                        "id": "completed-id",
                    },
                    {
                        "name": uncertain_create.name,
                        "args": {"name": "item"},
                        "id": "unsafe-id",
                    },
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [tool_calls],
                    [completed_lookup, uncertain_create],
                )
                with self.assertRaisesRegex(ConnectionError, "create disconnected"):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="run both")]},
                        config=run_config,
                        context=agent_runtime(session, root).context,
                    )

            TOOL_BEHAVIOR["fail_unsafe"] = False
            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [AIMessage(content="Retry complete.")],
                    [completed_lookup, uncertain_create],
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {
                            completed_lookup.name: SAFE_POLICY,
                            uncertain_create.name: UNSAFE_POLICY,
                        }
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)
                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["completed_calls"], 1)
                self.assertEqual(TOOL_BEHAVIOR["unsafe_calls"], 1)
                recovered = next(
                    message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "unsafe-id"
                )
                self.assertEqual(recovered.status, "error")
                self.assertIn("No retry was performed", recovered.content)
                self.assertEqual(
                    recovered.additional_kwargs["recovery"]["kind"],
                    "OUTCOME_UNKNOWN",
                )

    async def test_recovered_turn_can_enter_hitl_again_for_a_new_risky_call(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            checkpoint_path = root / "checkpoints.sqlite"
            store = SessionStore(root / "sessions.json")
            session = store.create(name="hitl-again", repository_path=str(root))
            run_config = {"configurable": {"thread_id": session.thread_id}}
            TOOL_BEHAVIOR["fail_unsafe"] = True
            failed_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": uncertain_create.name,
                        "args": {"name": "item"},
                        "id": "unsafe-id",
                    }
                ],
            )

            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = recovery_agent(
                    checkpointer,
                    [failed_call],
                    [uncertain_create],
                )
                with self.assertRaisesRegex(ConnectionError, "create disconnected"):
                    await agent.ainvoke(
                        {"messages": [HumanMessage(content="create it")]},
                        config=run_config,
                        context=agent_runtime(session, root).context,
                    )

            next_risky_call = AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": write_note.name,
                        "args": {"path": "notes.txt"},
                        "id": "next-risky-id",
                    }
                ],
            )
            async with open_checkpointer(checkpoint_path) as checkpointer:
                agent = create_agent(
                    model=ToolCallingFakeModel(responses=[next_risky_call]),
                    tools=[uncertain_create, write_note],
                    middleware=[
                        TurnRecoveryMiddleware(),
                        HumanInTheLoopMiddleware(
                            interrupt_on={write_note.name: True}
                        ),
                    ],
                    context_schema=AgentContext,
                    checkpointer=checkpointer,
                )
                continuation = SessionContinuation(
                    agent=agent,
                    policy_registry=ToolPolicyRegistry(
                        {
                            uncertain_create.name: UNSAFE_POLICY,
                            write_note.name: UNSAFE_POLICY,
                        }
                    ),
                    session_store=store,
                )
                active_runtime = agent_runtime(session, root)
                inspection = await continuation.inspect(active_runtime)
                result = await continuation.execute(
                    active_runtime,
                    ContinuationRequest(
                        action=ContinuationAction.CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                    ),
                )

                self.assertEqual(
                    result.inspection.status,
                    ContinuationStatus.WAITING_HUMAN,
                )
                self.assertEqual(
                    result.inspection.allowed_actions,
                    {
                        ContinuationAction.ANSWER_INTERRUPT,
                        ContinuationAction.TERMINATE_TURN,
                    },
                )
                self.assertEqual(
                    result.inspection.interrupts[0].value["action_requests"][0][
                        "name"
                    ],
                    write_note.name,
                )


if __name__ == "__main__":
    unittest.main()
