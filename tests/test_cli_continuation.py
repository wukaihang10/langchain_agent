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
    ToolCallResolution,
    ToolCallResolutionKind,
    UnresolvedToolCall,
)
from langchain_agent.cli.app import _execute_with_live_hitl, run_cli
from langchain_agent.cli.recovery import collect_tool_call_resolutions
from langchain_agent.cli.rendering import render_continuation


def inspection(status, *, checkpoint_id, interrupts=(), unresolved=()):
    allowed = {
        ContinuationStatus.EMPTY: {ContinuationAction.START_TURN},
        ContinuationStatus.READY: {ContinuationAction.START_TURN},
        ContinuationStatus.WAITING_HUMAN: {ContinuationAction.ANSWER_INTERRUPT},
        ContinuationStatus.RESUMABLE: {ContinuationAction.CONTINUE},
        ContinuationStatus.OUTCOME_UNKNOWN: {
            ContinuationAction.RESOLVE_AND_CONTINUE
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
        self.assertIn("Requires resolution Tools: create (unsafe)", rendered)
        self.assertIn("/continue", rendered)

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

    async def test_outcome_unknown_continue_submits_collected_resolutions(self):
        uncertain_call = UnresolvedToolCall(
            "unsafe-id", "create", {"name": "x"}, False, True, True
        )
        uncertain = inspection(
            ContinuationStatus.OUTCOME_UNKNOWN,
            checkpoint_id="checkpoint-1",
            unresolved=(uncertain_call,),
        )
        ready = inspection(ContinuationStatus.READY, checkpoint_id="checkpoint-2")
        resolution = ToolCallResolution(
            "unsafe-id",
            ToolCallResolutionKind.RECORD_OUTCOME_UNKNOWN,
        )
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
            prompt_async=AsyncMock(side_effect=["/new", "/continue", "/exit"])
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
            patch(
                "langchain_agent.cli.app.collect_tool_call_resolutions",
                return_value=[resolution],
            ) as collect,
            patch("langchain_agent.cli.app.render_active_session"),
            patch("langchain_agent.cli.app.render_continuation"),
            patch("langchain_agent.cli.app.render_result"),
            patch("langchain_agent.cli.app.render_mcp_tools"),
        ):
            await run_cli(application)

        request = continuation.execute.await_args.args[1]
        self.assertEqual(request.action, ContinuationAction.RESOLVE_AND_CONTINUE)
        self.assertEqual(request.observed_checkpoint_id, "checkpoint-1")
        self.assertEqual(request.resolutions, (resolution,))
        collect.assert_called_once_with((uncertain_call,))


class ToolRecoveryPromptTests(unittest.TestCase):
    @patch(
        "builtins.input",
        side_effect=[
            "1",
            "resource 42",
            "3",
            "n",
            "3",
            "y",
        ],
    )
    def test_collects_only_unsafe_calls_and_confirms_risky_retry(self, _input):
        calls = (
            UnresolvedToolCall("safe", "search", {}, True, True),
            UnresolvedToolCall(
                "success", "create", {"name": "x"}, False, True, True
            ),
            UnresolvedToolCall("retry", "send", {"id": 1}, False, True, True),
        )

        with patch("sys.stdout", io.StringIO()) as output:
            resolutions = collect_tool_call_resolutions(calls)

        self.assertEqual(
            [item.tool_call_id for item in resolutions],
            ["success", "retry"],
        )
        self.assertEqual(
            resolutions[0].kind,
            ToolCallResolutionKind.CONFIRM_SUCCEEDED,
        )
        self.assertEqual(resolutions[0].result_summary, "resource 42")
        self.assertEqual(
            resolutions[1].kind,
            ToolCallResolutionKind.RETRY_DESPITE_RISK,
        )
        self.assertIn("duplicate side effect", output.getvalue().lower())


if __name__ == "__main__":
    unittest.main()
