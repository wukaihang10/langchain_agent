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
)
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


class ToolCallingFakeModel(FakeMessagesListChatModel):
    def bind_tools(self, tools, *, tool_choice=None, **kwargs):
        return self


@tool
def write_note(path: str) -> str:
    """Write a test note."""

    return f"wrote {path}"


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


def runtime(session):
    return SimpleNamespace(
        session=session,
        invoke_config={"configurable": {"thread_id": session.thread_id}},
        context=None,
    )


class DurableSessionContinuationTests(unittest.IsolatedAsyncioTestCase):
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

    async def test_unsafe_tool_failure_survives_reopening_sqlite_and_stays_blocked(self):
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
                self.assertFalse(inspection.allowed_actions)


if __name__ == "__main__":
    unittest.main()
