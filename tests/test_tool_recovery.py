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

    async def test_plans_reject_directives_from_the_other_mode(self):
        with self.assertRaisesRegex(ValueError, "TERMINATE"):
            TurnRecoveryPlan(
                TurnRecoveryMode.TERMINATE,
                {
                    "call-1": ToolRecoveryDirective(
                        RecoveryDirectiveKind.OUTCOME_UNKNOWN,
                    )
                },
            )

        with self.assertRaisesRegex(ValueError, "CONTINUE"):
            TurnRecoveryPlan(
                TurnRecoveryMode.CONTINUE,
                {
                    "call-1": ToolRecoveryDirective(
                        RecoveryDirectiveKind.CANCELLED_BY_TERMINATION,
                    )
                },
            )

    async def test_continue_records_unknown_outcome_without_execution(self):
        plan = TurnRecoveryPlan(
            TurnRecoveryMode.CONTINUE,
            {
                "call-1": ToolRecoveryDirective(
                    RecoveryDirectiveKind.OUTCOME_UNKNOWN,
                )
            },
        )
        executions = 0

        async def handler(_request):
            nonlocal executions
            executions += 1
            return ToolMessage(content="real", tool_call_id="call-1")

        middleware = TurnRecoveryMiddleware()
        result = await middleware.awrap_tool_call(request(plan), handler)

        self.assertEqual(executions, 0)
        self.assertEqual(result.tool_call_id, "call-1")
        self.assertEqual(result.status, "error")
        self.assertIn("may have succeeded or failed", result.content)
        self.assertIn("No retry was performed", result.content)
        self.assertIn("Do not infer success or failure", result.content)
        self.assertIn("Verify external state with a read-only operation", result.content)
        self.assertEqual(
            result.additional_kwargs["recovery"],
            {
                "generated": True,
                "kind": "OUTCOME_UNKNOWN",
                "outcome": "outcome_unknown",
            },
        )
        with self.assertRaisesRegex(RuntimeError, "already consumed"):
            await middleware.awrap_tool_call(request(plan), handler)
        self.assertEqual(executions, 0)

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

        for directive_kind, expected_content, outcome in cases:
            with self.subTest(directive_kind=directive_kind):
                plan = TurnRecoveryPlan(
                    TurnRecoveryMode.TERMINATE,
                    {
                        "call-1": ToolRecoveryDirective(
                            RecoveryDirectiveKind(directive_kind),
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
                        "kind": directive_kind,
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
