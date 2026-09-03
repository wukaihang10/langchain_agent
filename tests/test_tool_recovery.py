import unittest
from types import SimpleNamespace

from langchain_core.messages import ToolMessage

from langchain_agent.harness.middleware.tool_recovery import (
    RecoveryDirectiveKind,
    ToolRecoveryDirective,
    ToolRecoveryMiddleware,
    ToolRecoveryPlan,
)


def request(plan):
    return SimpleNamespace(
        tool_call={"id": "call-1", "name": "create_item", "args": {}},
        runtime=SimpleNamespace(
            context=SimpleNamespace(tool_recovery=plan),
        ),
    )


class ToolRecoveryMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_confirmed_success_is_auditable_without_execution(self):
        plan = ToolRecoveryPlan(
            {
                "call-1": ToolRecoveryDirective(
                    kind=RecoveryDirectiveKind.SYNTHETIC_SUCCESS,
                    resolution_kind="CONFIRM_SUCCEEDED",
                    result_summary="created item 42",
                )
            }
        )
        executions = 0

        async def handler(_request):
            nonlocal executions
            executions += 1
            return ToolMessage(content="real", tool_call_id="call-1")

        result = await ToolRecoveryMiddleware().awrap_tool_call(
            request(plan), handler
        )

        self.assertEqual(executions, 0)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.status, "success")
        self.assertIn("verified by the user", result.content)
        self.assertIn("created item 42", result.content)
        self.assertEqual(
            result.additional_kwargs["recovery"],
            {"generated": True, "resolution_kind": "CONFIRM_SUCCEEDED"},
        )

    async def test_error_resolutions_are_truthful_and_do_not_execute(self):
        cases = (
            (
                "CONFIRM_NOT_APPLIED",
                "did not take effect",
                "checked the remote list",
            ),
            (
                "RECORD_OUTCOME_UNKNOWN",
                "may have succeeded or failed",
                "cannot access the service",
            ),
        )
        for resolution_kind, expected, note in cases:
            with self.subTest(resolution_kind=resolution_kind):
                plan = ToolRecoveryPlan(
                    {
                        "call-1": ToolRecoveryDirective(
                            kind=RecoveryDirectiveKind.SYNTHETIC_ERROR,
                            resolution_kind=resolution_kind,
                            note=note,
                        )
                    }
                )

                async def handler(_request):
                    self.fail("synthetic recovery must bypass real execution")

                result = await ToolRecoveryMiddleware().awrap_tool_call(
                    request(plan), handler
                )

                self.assertEqual(result.status, "error")
                self.assertIn(expected, result.content)
                self.assertIn(note, result.content)
                if resolution_kind == "RECORD_OUTCOME_UNKNOWN":
                    self.assertIn("No retry was performed", result.content)

    async def test_explicit_retry_executes_once_and_directive_cannot_be_reused(self):
        plan = ToolRecoveryPlan(
            {
                "call-1": ToolRecoveryDirective(
                    kind=RecoveryDirectiveKind.RETRY,
                    resolution_kind="RETRY_DESPITE_RISK",
                )
            }
        )
        executions = 0

        async def handler(_request):
            nonlocal executions
            executions += 1
            return ToolMessage(content="retried", tool_call_id="call-1")

        middleware = ToolRecoveryMiddleware()
        result = await middleware.awrap_tool_call(request(plan), handler)

        self.assertEqual(result.content, "retried")
        self.assertEqual(executions, 1)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            await middleware.awrap_tool_call(request(plan), handler)
        self.assertEqual(executions, 1)
