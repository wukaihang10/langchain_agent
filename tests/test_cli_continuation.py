import io
import unittest
from types import SimpleNamespace
from unittest.mock import patch

from langchain_agent.app.session_continuation import (
    ContinuationAction,
    ContinuationInspection,
    ContinuationResult,
    ContinuationStatus,
    PendingInterrupt,
)
from langchain_agent.cli.app import _execute_with_live_hitl
from langchain_agent.cli.rendering import render_continuation


def inspection(status, *, checkpoint_id, interrupts=()):
    allowed = {
        ContinuationStatus.EMPTY: {ContinuationAction.START_TURN},
        ContinuationStatus.READY: {ContinuationAction.START_TURN},
        ContinuationStatus.WAITING_HUMAN: {ContinuationAction.ANSWER_INTERRUPT},
        ContinuationStatus.RESUMABLE: {ContinuationAction.CONTINUE},
        ContinuationStatus.OUTCOME_UNKNOWN: set(),
        ContinuationStatus.NEEDS_REPAIR: set(),
    }
    return ContinuationInspection(
        status=status,
        checkpoint_id=checkpoint_id,
        pending_nodes=(),
        interrupts=tuple(interrupts),
        unresolved_tool_calls=(),
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
                )
            )

        rendered = output.getvalue()
        self.assertEqual(rendered.count("Status:"), 1)
        self.assertIn("Status: OUTCOME_UNKNOWN", rendered)

    def test_waiting_human_renders_original_request_and_cli_command(self):
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
        self.assertIn("Tool: write_file", rendered)
        self.assertIn("notes.txt", rendered)
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
        render.assert_called_once_with(waiting)
        collect_hitl_decisions.assert_called_once_with((interrupt,))


if __name__ == "__main__":
    unittest.main()
