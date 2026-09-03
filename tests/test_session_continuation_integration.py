import tempfile
import unittest
from pathlib import Path
from types import SimpleNamespace

from langchain.agents import create_agent
from langchain.agents.middleware import HumanInTheLoopMiddleware
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
    ToolCallResolution,
    ToolCallResolutionKind,
)
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.middleware.tool_recovery import ToolRecoveryMiddleware
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
    "completed_calls": 0,
    "safe_calls": 0,
    "unsafe_calls": 0,
    "fail_safe": False,
    "fail_unsafe": False,
}


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@tool
def write_note(path: str) -> str:
    """Write a test note."""

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
        middleware=[ToolRecoveryMiddleware()],
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
                    {ContinuationAction.ANSWER_INTERRUPT},
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
                    {ContinuationAction.CONTINUE},
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
                    {ContinuationAction.RESOLVE_AND_CONTINUE},
                )

    async def test_confirmed_success_preserves_completed_sibling(self):
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
                        action=ContinuationAction.RESOLVE_AND_CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        resolutions=(
                            ToolCallResolution(
                                tool_call_id="unsafe-id",
                                kind=ToolCallResolutionKind.CONFIRM_SUCCEEDED,
                                result_summary="item id 42",
                            ),
                        ),
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
                self.assertEqual(recovered.status, "success")
                self.assertEqual(
                    recovered.additional_kwargs["recovery"]["resolution_kind"],
                    "CONFIRM_SUCCEEDED",
                )
                self.assertIn("item id 42", recovered.content)

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
                        "resolution_kind": "CONFIRM_SUCCEEDED",
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
                        action=ContinuationAction.RESOLVE_AND_CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        resolutions=(
                            ToolCallResolution(
                                tool_call_id="unsafe-id",
                                kind=(
                                    ToolCallResolutionKind.RECORD_OUTCOME_UNKNOWN
                                ),
                                note="external state unavailable",
                            ),
                        ),
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

    async def test_risky_retry_executes_only_failed_tool_once(self):
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
                        action=ContinuationAction.RESOLVE_AND_CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        resolutions=(
                            ToolCallResolution(
                                tool_call_id="unsafe-id",
                                kind=ToolCallResolutionKind.RETRY_DESPITE_RISK,
                            ),
                        ),
                    ),
                )

                self.assertEqual(result.inspection.status, ContinuationStatus.READY)
                self.assertEqual(TOOL_BEHAVIOR["completed_calls"], 1)
                self.assertEqual(TOOL_BEHAVIOR["unsafe_calls"], 2)
                retried = next(
                    message
                    for message in result.value["messages"]
                    if isinstance(message, ToolMessage)
                    and message.tool_call_id == "unsafe-id"
                )
                self.assertEqual(retried.content, "created item")
                self.assertNotIn("recovery", retried.additional_kwargs)

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
                        ToolRecoveryMiddleware(),
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
                        action=ContinuationAction.RESOLVE_AND_CONTINUE,
                        observed_checkpoint_id=inspection.checkpoint_id,
                        resolutions=(
                            ToolCallResolution(
                                tool_call_id="unsafe-id",
                                kind=ToolCallResolutionKind.CONFIRM_SUCCEEDED,
                            ),
                        ),
                    ),
                )

                self.assertEqual(
                    result.inspection.status,
                    ContinuationStatus.WAITING_HUMAN,
                )
                self.assertEqual(
                    result.inspection.allowed_actions,
                    {ContinuationAction.ANSWER_INTERRUPT},
                )
                self.assertEqual(
                    result.inspection.interrupts[0].value["action_requests"][0][
                        "name"
                    ],
                    write_note.name,
                )


if __name__ == "__main__":
    unittest.main()
