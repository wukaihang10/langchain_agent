import unittest
from types import SimpleNamespace

from langchain.agents.middleware.types import ModelResponse
from langchain_core.messages import AIMessage, ToolMessage

from langchain_agent.harness.middleware.turn_recovery import (
    RecoveryDirectiveKind,
    TurnRecoveryMiddleware,
    TurnRecoveryMode,
    TurnRecoveryPlan,
    ToolRecoveryDirective,
)


def request(plan):
    return SimpleNamespace(
        tool_call={"id": "call-1", "name": "create_item", "args": {}},
        runtime=SimpleNamespace(
            context=SimpleNamespace(turn_recovery=plan),
        ),
    )


def model_request(plan):
    return SimpleNamespace(
        runtime=SimpleNamespace(
            context=SimpleNamespace(turn_recovery=plan),
        )
    )


class TurnRecoveryMiddlewareTests(unittest.IsolatedAsyncioTestCase):
    async def test_absent_plan_delegates_tool_and_model_calls(self):
        middleware = TurnRecoveryMiddleware()
        calls = []

        async def tool_handler(_request):
            calls.append("tool")
            return ToolMessage(content="real tool", tool_call_id="call-1")

        async def model_handler(_request):
            calls.append("model")
            return ModelResponse(result=[AIMessage(content="real model")])

        tool_result = await middleware.awrap_tool_call(request(None), tool_handler)
        model_result = await middleware.awrap_model_call(
            model_request(None), model_handler
        )

        self.assertEqual(tool_result.content, "real tool")
        self.assertEqual(model_result.result[0].content, "real model")
        self.assertEqual(calls, ["tool", "model"])

    async def test_terminate_plan_rejects_any_directive_that_could_execute_a_tool(self):
        with self.assertRaisesRegex(ValueError, "TERMINATE"):
            TurnRecoveryPlan(
                TurnRecoveryMode.TERMINATE,
                {
                    "call-1": ToolRecoveryDirective(
                        kind=RecoveryDirectiveKind.RETRY,
                        resolution_kind="RETRY_DESPITE_RISK",
                    )
                },
            )

    async def test_confirmed_success_is_auditable_without_execution(self):
        plan = TurnRecoveryPlan(
            TurnRecoveryMode.CONTINUE,
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

        result = await TurnRecoveryMiddleware().awrap_tool_call(
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
                plan = TurnRecoveryPlan(
                    TurnRecoveryMode.CONTINUE,
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

                result = await TurnRecoveryMiddleware().awrap_tool_call(
                    request(plan), handler
                )

                self.assertEqual(result.status, "error")
                self.assertIn(expected, result.content)
                self.assertIn(note, result.content)
                if resolution_kind == "RECORD_OUTCOME_UNKNOWN":
                    self.assertIn("No retry was performed", result.content)

    async def test_explicit_retry_executes_once_and_directive_cannot_be_reused(self):
        plan = TurnRecoveryPlan(
            TurnRecoveryMode.CONTINUE,
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

        middleware = TurnRecoveryMiddleware()
        result = await middleware.awrap_tool_call(request(plan), handler)

        self.assertEqual(result.content, "retried")
        self.assertEqual(executions, 1)
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            await middleware.awrap_tool_call(request(plan), handler)
        self.assertEqual(executions, 1)

    async def test_terminate_mode_returns_terminal_message_without_calling_model(self):
        plan = TurnRecoveryPlan(TurnRecoveryMode.TERMINATE, {})

        async def handler(_request):
            self.fail("turn termination must bypass the main Agent model")

        result = await TurnRecoveryMiddleware().awrap_model_call(
            model_request(plan), handler
        )

        self.assertIsInstance(result, ModelResponse)
        self.assertEqual(len(result.result), 1)
        message = result.result[0]
        self.assertIsInstance(message, AIMessage)
        self.assertFalse(message.tool_calls)
        self.assertIn("terminated at the user's request", message.content)
        self.assertEqual(
            message.additional_kwargs["recovery"],
            {"generated": True, "action": "TERMINATE_TURN"},
        )

    async def test_terminate_mode_records_cancelled_and_unknown_tool_outcomes(self):
        cases = (
            (
                "CANCELLED_BY_TERMINATION",
                "No new tool execution occurred during termination",
                "cancelled",
            ),
            (
                "OUTCOME_UNKNOWN_AT_TERMINATION",
                "may have succeeded or failed",
                "outcome_unknown",
            ),
        )

        for resolution_kind, expected_content, outcome in cases:
            with self.subTest(resolution_kind=resolution_kind):
                plan = TurnRecoveryPlan(
                    TurnRecoveryMode.TERMINATE,
                    {
                        "call-1": ToolRecoveryDirective(
                            kind=RecoveryDirectiveKind.SYNTHETIC_ERROR,
                            resolution_kind=resolution_kind,
                        )
                    },
                )

                async def handler(_request):
                    self.fail("turn termination must bypass real tool execution")

                result = await TurnRecoveryMiddleware().awrap_tool_call(
                    request(plan), handler
                )

                self.assertEqual(result.status, "error")
                self.assertEqual(result.tool_call_id, "call-1")
                self.assertIn(expected_content, result.content)
                self.assertEqual(
                    result.additional_kwargs["recovery"],
                    {
                        "generated": True,
                        "action": "TERMINATE_TURN",
                        "resolution_kind": resolution_kind,
                        "outcome": outcome,
                    },
                )

    async def test_terminate_mode_fails_closed_when_a_tool_has_no_directive(self):
        plan = TurnRecoveryPlan(TurnRecoveryMode.TERMINATE, {})
        executions = 0

        async def handler(_request):
            nonlocal executions
            executions += 1
            return ToolMessage(content="real", tool_call_id="call-1")

        with self.assertRaisesRegex(RuntimeError, "no recovery directive"):
            await TurnRecoveryMiddleware().awrap_tool_call(request(plan), handler)

        self.assertEqual(executions, 0)
