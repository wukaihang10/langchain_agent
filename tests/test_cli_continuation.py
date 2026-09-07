import asyncio
import io
import unittest
from types import SimpleNamespace
from unittest.mock import AsyncMock, patch

from langchain_agent.app.session_continuation import (
    ContinuationAction,
    ContinuationInspection,
    ContinuationResult,
    ContinuationStatus,
    PendingInterrupt,
    StopOutcome,
    StopResult,
    UnresolvedToolCall,
)
from langchain_agent.cli.app import _execute_with_live_hitl, run_cli
from langchain_agent.cli.rendering import render_continuation


def inspection(status, *, checkpoint_id, interrupts=(), unresolved=()):
    allowed = {
        ContinuationStatus.EMPTY: {ContinuationAction.START_TURN},
        ContinuationStatus.READY: {ContinuationAction.START_TURN},
        ContinuationStatus.WAITING_HUMAN: {
            ContinuationAction.ANSWER_INTERRUPT,
            ContinuationAction.TERMINATE_TURN,
        },
        ContinuationStatus.RESUMABLE: {
            ContinuationAction.CONTINUE,
            ContinuationAction.TERMINATE_TURN,
        },
        ContinuationStatus.OUTCOME_UNKNOWN: {
            ContinuationAction.CONTINUE,
            ContinuationAction.TERMINATE_TURN,
        },
        ContinuationStatus.NEEDS_REPAIR: set(),
    }
    return ContinuationInspection(
        status=status,
        checkpoint_id=checkpoint_id,
        pending_nodes=(),
        interrupts=tuple(interrupts),
        unresolved_tool_calls=tuple(unresolved),
        allowed_actions=frozenset(allowed[status]),
        reason=f"reason for {status.value}",
    )


class FakeContinuation:
    def __init__(self, results):
        self.results = list(results)
        self.requests = []

    async def execute(self, runtime, request):
        self.requests.append((runtime, request))
        return self.results.pop(0)


class ContinuationRenderingTests(unittest.TestCase):
    def test_rendering_emits_one_semantic_status(self):
        output = io.StringIO()
        with patch("sys.stdout", output):
            render_continuation(
                inspection(
                    ContinuationStatus.OUTCOME_UNKNOWN,
                    checkpoint_id="checkpoint-1",
                    unresolved=(
                        UnresolvedToolCall(
                            "safe", "search", {}, True, True, False
                        ),
                        UnresolvedToolCall(
                            "unsafe", "create", {}, False, True, True
                        ),
                    ),
                )
            )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("Status:"), 1)
        self.assertIn("Status: OUTCOME_UNKNOWN", rendered)
        self.assertIn("Automatic retry Tools: search (safe)", rendered)
        self.assertIn("Outcome-unknown Tools: create (unsafe)", rendered)
        self.assertIn("/continue", rendered)
        self.assertIn("/terminate", rendered)

    def test_waiting_human_renders_interrupt_description_and_cli_command(self):
        output = io.StringIO()
        interrupt = PendingInterrupt(
            id="interrupt-1",
            value={
                "action_requests": [
                    {
                        "name": "write_file",
                        "args": {"path": "notes.txt", "content": "hello"},
                        "description": "Tool execution requires permission",
                    }
                ],
                "review_configs": [
                    {"allowed_decisions": ["approve", "reject"]}
                ],
            },
        )

        with patch("sys.stdout", output):
            render_continuation(
                inspection(
                    ContinuationStatus.WAITING_HUMAN,
                    checkpoint_id="checkpoint-1",
                    interrupts=[interrupt],
                )
            )

        rendered = output.getvalue()
        self.assertIn(
            "Tool Interrupt: Tool execution requires permission",
            rendered,
        )
        self.assertIn("Allowed decisions: approve, reject", rendered)
        self.assertIn("/continue", rendered)
        self.assertNotIn("ANSWER_INTERRUPT", rendered)


class LiveHitlRoutingTests(unittest.IsolatedAsyncioTestCase):
    @patch(
        "langchain_agent.cli.app.collect_hitl_decisions",
        return_value=[{"type": "approve"}],
    )
    async def test_cli_submits_decision_through_continuation_interface(
        self,
        collect_hitl_decisions,
    ):
        interrupt = PendingInterrupt(
            id="interrupt-1",
            value={"action_requests": [], "review_configs": []},
        )
        waiting = inspection(
            ContinuationStatus.WAITING_HUMAN,
            checkpoint_id="checkpoint-2",
            interrupts=[interrupt],
        )
        ready = inspection(
            ContinuationStatus.READY,
            checkpoint_id="checkpoint-3",
        )
        continuation = FakeContinuation(
            [
                ContinuationResult(value={"messages": []}, inspection=waiting),
                ContinuationResult(
                    value={"messages": [SimpleNamespace(content="done")]},
                    inspection=ready,
                ),
            ]
        )
        application = SimpleNamespace(continuation=continuation)
        runtime = object()
        start_request = SimpleNamespace(action=ContinuationAction.START_TURN)

        with patch("langchain_agent.cli.app.render_continuation") as render:
            value, final_inspection = await _execute_with_live_hitl(
                application=application,
                runtime=runtime,
                request=start_request,
            )

        self.assertEqual(value["messages"][-1].content, "done")
        self.assertEqual(final_inspection.status, ContinuationStatus.READY)
        self.assertIs(continuation.requests[0][1], start_request)
        answer = continuation.requests[1][1]
        self.assertEqual(answer.action, ContinuationAction.ANSWER_INTERRUPT)
        self.assertEqual(answer.observed_checkpoint_id, "checkpoint-2")
        self.assertEqual(answer.decisions, ({"type": "approve"},))
        render.assert_not_called()
        collect_hitl_decisions.assert_called_once_with((interrupt,))

    async def test_outcome_unknown_continue_needs_no_user_resolution(self):
        uncertain_call = UnresolvedToolCall(
            "unsafe-id", "create", {"name": "x"}, False, True, True
        )
        uncertain = inspection(
            ContinuationStatus.OUTCOME_UNKNOWN,
            checkpoint_id="checkpoint-1",
            unresolved=(uncertain_call,),
        )
        ready = inspection(ContinuationStatus.READY, checkpoint_id="checkpoint-2")
        continuation = SimpleNamespace(
            inspect=AsyncMock(return_value=uncertain),
            execute=AsyncMock(
                return_value=ContinuationResult(
                    value={"messages": [SimpleNamespace(content="done")]},
                    inspection=ready,
                )
            ),
        )
        application = SimpleNamespace(
            continuation=continuation,
            mcp=SimpleNamespace(tools=[]),
            session_store=object(),
        )
        session = SimpleNamespace(thread_id="thread-1", name="test")
        runtime = SimpleNamespace(session=session, context=object())
        prompt_session = SimpleNamespace(
            prompt_async=AsyncMock(
                side_effect=["/new", "/continue", "stale prompt", "/exit"]
            )
        )

        with (
            patch("langchain_agent.cli.app.PromptSession", return_value=prompt_session),
            patch(
                "langchain_agent.cli.app.create_session_interactively",
                return_value=session,
            ),
            patch(
                "langchain_agent.cli.app._build_session_runtime",
                return_value=runtime,
            ),
            patch("langchain_agent.cli.app.render_active_session"),
            patch("langchain_agent.cli.app.render_continuation"),
            patch("langchain_agent.cli.app.render_result"),
            patch("langchain_agent.cli.app.render_mcp_tools"),
        ):
            await run_cli(application)

        request = continuation.execute.await_args.args[1]
        self.assertEqual(request.action, ContinuationAction.CONTINUE)
        self.assertEqual(request.observed_checkpoint_id, "checkpoint-1")

    async def test_terminate_command_submits_only_the_checkpoint_guard(self):
        uncertain_call = UnresolvedToolCall(
            "unsafe-id", "create", {"name": "x"}, False, True, True
        )
        uncertain = inspection(
            ContinuationStatus.OUTCOME_UNKNOWN,
            checkpoint_id="checkpoint-1",
            unresolved=(uncertain_call,),
        )
        ready = inspection(ContinuationStatus.READY, checkpoint_id="checkpoint-2")
        continuation = SimpleNamespace(
            inspect=AsyncMock(return_value=uncertain),
            execute=AsyncMock(
                return_value=ContinuationResult(
                    value={"messages": [SimpleNamespace(content="terminated")]},
                    inspection=ready,
                )
            ),
        )
        application = SimpleNamespace(
            continuation=continuation,
            mcp=SimpleNamespace(tools=[]),
            session_store=object(),
        )
        session = SimpleNamespace(thread_id="thread-1", name="test")
        runtime = SimpleNamespace(session=session, context=object())
        prompt_session = SimpleNamespace(
            prompt_async=AsyncMock(
                side_effect=["/new", "/terminate", "stale prompt", "/exit"]
            )
        )

        with (
            patch("langchain_agent.cli.app.PromptSession", return_value=prompt_session),
            patch(
                "langchain_agent.cli.app.create_session_interactively",
                return_value=session,
            ),
            patch(
                "langchain_agent.cli.app._build_session_runtime",
                return_value=runtime,
            ),
            patch("langchain_agent.cli.app.collect_hitl_decisions") as collect_hitl,
            patch("langchain_agent.cli.app.render_active_session"),
            patch("langchain_agent.cli.app.render_continuation"),
            patch("langchain_agent.cli.app.render_result"),
            patch("langchain_agent.cli.app.render_mcp_tools"),
        ):
            await run_cli(application)

        request = continuation.execute.await_args.args[1]
        self.assertEqual(request.action, ContinuationAction.TERMINATE_TURN)
        self.assertEqual(request.observed_checkpoint_id, "checkpoint-1")
        self.assertIsNone(request.message)
        self.assertFalse(request.decisions)
        collect_hitl.assert_not_called()


class RunningTurnPromptTests(unittest.IsolatedAsyncioTestCase):
    async def test_agent_completion_cancels_and_awaits_running_prompt(self):
        ready = inspection(ContinuationStatus.READY, checkpoint_id="checkpoint-2")
        execute_started = asyncio.Event()
        finish_execution = asyncio.Event()
        prompt_started = asyncio.Event()
        prompt_cancelled = asyncio.Event()

        async def execute(runtime, request):
            execute_started.set()
            await finish_execution.wait()
            return ContinuationResult(
                value={"messages": [SimpleNamespace(content="done")]},
                inspection=ready,
            )

        async def prompt_async(prompt):
            prompt_started.set()
            try:
                await asyncio.Future()
            except asyncio.CancelledError:
                prompt_cancelled.set()
                raise

        application = SimpleNamespace(
            continuation=SimpleNamespace(execute=execute),
        )
        runtime = SimpleNamespace(session=SimpleNamespace(thread_id="thread-1"))
        prompt_session = SimpleNamespace(prompt_async=prompt_async)

        running = asyncio.create_task(
            _execute_with_live_hitl(
                application=application,
                runtime=runtime,
                request=SimpleNamespace(action=ContinuationAction.START_TURN),
                prompt_session=prompt_session,
            )
        )
        await asyncio.gather(execute_started.wait(), prompt_started.wait())
        finish_execution.set()
        value, final_inspection = await running

        self.assertEqual(value["messages"][-1].content, "done")
        self.assertEqual(final_inspection.status, ContinuationStatus.READY)
        self.assertTrue(prompt_cancelled.is_set())

    async def test_running_prompt_routes_only_stop_and_never_queues_text(self):
        ready = inspection(ContinuationStatus.READY, checkpoint_id="checkpoint-2")
        stop_called = asyncio.Event()
        stop_calls = []

        async def execute(runtime, request):
            await stop_called.wait()
            return ContinuationResult(
                value={"messages": [SimpleNamespace(content="terminated")]},
                inspection=ready,
            )

        async def stop(thread_id):
            stop_calls.append(thread_id)
            stop_called.set()
            return StopResult(
                outcome=StopOutcome.STOPPED,
                inspection=ready,
                value={"messages": [SimpleNamespace(content="terminated")]},
            )

        prompt_session = SimpleNamespace(
            prompt_async=AsyncMock(side_effect=["queue this", "/stop"])
        )
        application = SimpleNamespace(
            continuation=SimpleNamespace(execute=execute, stop=stop),
        )
        runtime = SimpleNamespace(session=SimpleNamespace(thread_id="thread-1"))
        output = io.StringIO()

        with patch("sys.stdout", output):
            value, final_inspection = await _execute_with_live_hitl(
                application=application,
                runtime=runtime,
                request=SimpleNamespace(action=ContinuationAction.START_TURN),
                prompt_session=prompt_session,
            )

        self.assertEqual(stop_calls, ["thread-1"])
        self.assertEqual(value["messages"][-1].content, "terminated")
        self.assertEqual(final_inspection.status, ContinuationStatus.READY)
        self.assertIn("not queued", output.getvalue())
        self.assertIn("waiting for it to settle", output.getvalue())
        self.assertIn("stopped safely", output.getvalue())


if __name__ == "__main__":
    unittest.main()
