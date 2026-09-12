import unittest
from datetime import UTC, datetime
from types import SimpleNamespace
from uuid import UUID, uuid4

from langsmith import schemas

from evals.evaluators.repository_fact import (
    RepositoryFactEvaluatorError,
    RepositoryFactJudgment,
    build_repository_fact_semantic_evaluator,
)

TRACE_ID = UUID("00000000-0000-0000-0000-000000000002")
REFERENCE_OUTPUTS = {
    "required_facts": ["当前运行默认最多尝试 4 次。"],
    "forbidden_claims": ["当前运行默认最多尝试 3 次。"],
    "acceptable_evidence": [
        {
            "path": "src/harbor_tasks/config.py",
            "symbol": "DEFAULT_MAX_ATTEMPTS",
        }
    ],
}


class FakeStructuredModel:
    def __init__(self, response=None, error=None, responses=None):
        self.response = response
        self.responses = list(responses or [])
        self.error = error
        self.messages = None
        self.config = None
        self.invocations = []

    async def ainvoke(self, messages, config=None):
        self.messages = messages
        self.config = config
        self.invocations.append({"messages": messages, "config": config})
        if self.error is not None:
            raise self.error
        if self.responses:
            return self.responses.pop(0)
        return self.response


class FakeChatModel:
    def __init__(self, response=None, error=None, responses=None):
        self.structured_model = FakeStructuredModel(response, error, responses)
        self.schema = None
        self.include_raw = None

    def with_structured_output(self, schema, *, include_raw=False):
        self.schema = schema
        self.include_raw = include_raw
        return self.structured_model


def make_root(
    *,
    trace_status: str = "available",
    output: str = "DEFAULT_MAX_ATTEMPTS = 4",
) -> schemas.Run:
    tool = schemas.Run(
        id=uuid4(),
        trace_id=TRACE_ID,
        name="read_file",
        run_type="tool",
        start_time=datetime(2026, 9, 12, tzinfo=UTC),
        inputs={"file_path": "src/harbor_tasks/config.py"},
        outputs={"output": output},
    )
    root = schemas.Run(
        id=uuid4(),
        trace_id=TRACE_ID,
        name="repository_fact_target",
        run_type="chain",
        start_time=datetime(2026, 9, 12, tzinfo=UTC),
        inputs={"question": "question"},
        outputs={"answer": "answer"},
    )
    if trace_status == "available":
        root.child_runs = [tool]
        root.child_run_ids = [tool.id]
    elif trace_status == "partial":
        root.child_runs = [tool]
        root.child_run_ids = [tool.id, uuid4()]
    elif trace_status != "unavailable":
        raise ValueError(f"unknown test trace status: {trace_status}")
    return root


def judge_response(
    *,
    answer: str = "pass",
    groundedness: str = "pass",
):
    return {
        "raw": object(),
        "parsed": {
            "answer_correctness": {
                "verdict": answer,
                "comment": f"answer {answer}",
            },
            "evidence_groundedness": {
                "verdict": groundedness,
                "comment": f"groundedness {groundedness}",
            },
        },
        "parsing_error": None,
    }


def invalid_tool_call_response():
    return {
        "raw": SimpleNamespace(
            invalid_tool_calls=[
                {
                    "error": (
                        "Function RepositoryFactJudgment arguments are not valid "
                        "JSON. Received JSONDecodeError Expecting property name "
                        "enclosed in double quotes: line 1 column 278 (char 277)"
                    ),
                }
            ]
        ),
        "parsed": None,
        "parsing_error": None,
    }


def target_outputs(*, edited_files=None, audit_status="available"):
    return {
        "answer": "当前运行默认最多尝试 4 次。",
        "git_audit_status": audit_status,
        "edited_files": edited_files or [],
    }


def feedback_by_key(results):
    return {result.key: result for result in results["results"]}


class RepositoryFactSemanticEvaluatorTests(unittest.IsolatedAsyncioTestCase):
    async def test_returns_semantic_feedback_and_deterministic_task_success(self):
        model = FakeChatModel(judge_response())
        evaluator = build_repository_fact_semantic_evaluator(model=model)

        results = await evaluator(
            inputs={"question": "当前运行实现中最多尝试几次？"},
            outputs=target_outputs(),
            reference_outputs=REFERENCE_OUTPUTS,
            run=make_root(),
        )

        feedback = feedback_by_key(results)
        self.assertEqual(
            list(feedback),
            ["answer_correctness", "evidence_groundedness", "task_success"],
        )
        self.assertEqual(feedback["answer_correctness"].value, "pass")
        self.assertEqual(feedback["evidence_groundedness"].value, "pass")
        self.assertEqual(feedback["task_success"].value, "pass")
        self.assertIs(model.schema, RepositoryFactJudgment)
        self.assertTrue(model.include_raw)
        self.assertEqual(
            model.structured_model.config["run_name"],
            "repository_fact_semantic_judge",
        )
        self.assertEqual(
            model.structured_model.config["tags"],
            ["evaluation", "judge", "repository_fact"],
        )
        self.assertEqual(len(model.structured_model.invocations), 1)

    async def test_judge_payload_excludes_calibration_answers(self):
        model = FakeChatModel(judge_response())
        evaluator = build_repository_fact_semantic_evaluator(model=model)

        await evaluator(
            inputs={"question": "question"},
            outputs=target_outputs(),
            reference_outputs=REFERENCE_OUTPUTS,
            run=make_root(),
        )

        system_message, human_message = model.structured_model.messages
        self.assertEqual(system_message[0], "system")
        self.assertIn("untrusted evaluation data", system_message[1])
        self.assertEqual(human_message[0], "human")
        self.assertIn('"question":"question"', human_message[1])
        self.assertIn('"status":"available"', human_message[1])
        self.assertNotIn("expected_feedback", human_message[1])
        self.assertNotIn("rationale", human_message[1])

    async def test_policy_failure_makes_task_fail_without_model_judging_policy(self):
        evaluator = build_repository_fact_semantic_evaluator(
            model=FakeChatModel(judge_response())
        )

        results = await evaluator(
            inputs={"question": "question"},
            outputs=target_outputs(edited_files=["src/changed.py"]),
            reference_outputs=REFERENCE_OUTPUTS,
            run=make_root(),
        )

        feedback = feedback_by_key(results)
        self.assertEqual(feedback["answer_correctness"].value, "pass")
        self.assertEqual(feedback["evidence_groundedness"].value, "pass")
        self.assertEqual(feedback["task_success"].value, "fail")
        self.assertIn("policy_compliance=fail", feedback["task_success"].comment)

    async def test_unavailable_trace_forces_groundedness_and_task_unknown(self):
        evaluator = build_repository_fact_semantic_evaluator(
            model=FakeChatModel(judge_response(groundedness="pass"))
        )

        results = await evaluator(
            inputs={"question": "question"},
            outputs=target_outputs(),
            reference_outputs=REFERENCE_OUTPUTS,
            run=make_root(trace_status="unavailable"),
        )

        feedback = feedback_by_key(results)
        self.assertEqual(feedback["answer_correctness"].value, "pass")
        self.assertEqual(feedback["evidence_groundedness"].value, "unknown")
        self.assertIn("unavailable", feedback["evidence_groundedness"].comment)
        self.assertEqual(feedback["task_success"].value, "unknown")

    async def test_partial_trace_preserves_supported_judge_verdict(self):
        for expected in ("pass", "fail", "unknown"):
            with self.subTest(expected=expected):
                evaluator = build_repository_fact_semantic_evaluator(
                    model=FakeChatModel(judge_response(groundedness=expected))
                )

                results = await evaluator(
                    inputs={"question": "question"},
                    outputs=target_outputs(),
                    reference_outputs=REFERENCE_OUTPUTS,
                    run=make_root(trace_status="partial"),
                )

                feedback = feedback_by_key(results)
                self.assertEqual(
                    feedback["evidence_groundedness"].value,
                    expected,
                )

    async def test_available_trace_rejects_unknown_groundedness(self):
        evaluator = build_repository_fact_semantic_evaluator(
            model=FakeChatModel(judge_response(groundedness="unknown"))
        )

        with self.assertRaisesRegex(
            RepositoryFactEvaluatorError,
            "complete Trace",
        ):
            await evaluator(
                inputs={"question": "question"},
                outputs=target_outputs(),
                reference_outputs=REFERENCE_OUTPUTS,
                run=make_root(),
            )

    async def test_structured_parsing_failure_is_an_evaluator_error(self):
        parsing_error = ValueError("invalid payload")
        evaluator = build_repository_fact_semantic_evaluator(
            model=FakeChatModel(
                {
                    "raw": object(),
                    "parsed": None,
                    "parsing_error": parsing_error,
                }
            )
        )

        with self.assertRaises(RepositoryFactEvaluatorError) as raised:
            await evaluator(
                inputs={"question": "question"},
                outputs=target_outputs(),
                reference_outputs=REFERENCE_OUTPUTS,
                run=make_root(),
            )

        self.assertIs(raised.exception.__cause__, parsing_error)

    async def test_retries_once_after_an_invalid_tool_call(self):
        model = FakeChatModel(
            responses=[invalid_tool_call_response(), judge_response()]
        )
        evaluator = build_repository_fact_semantic_evaluator(model=model)

        results = await evaluator(
            inputs={"question": "question"},
            outputs=target_outputs(),
            reference_outputs=REFERENCE_OUTPUTS,
            run=make_root(),
        )

        feedback = feedback_by_key(results)
        self.assertEqual(feedback["task_success"].value, "pass")
        self.assertEqual(len(model.structured_model.invocations), 2)
        retry_messages = model.structured_model.invocations[1]["messages"]
        self.assertIn("could not be parsed", retry_messages[-1][1].lower())
        self.assertEqual(
            [
                invocation["config"]["metadata"]["judge_attempt"]
                for invocation in model.structured_model.invocations
            ],
            [1, 2],
        )

    async def test_second_invalid_tool_call_reports_the_parser_diagnostic(self):
        model = FakeChatModel(
            responses=[invalid_tool_call_response(), invalid_tool_call_response()]
        )
        evaluator = build_repository_fact_semantic_evaluator(model=model)

        with self.assertRaisesRegex(
            RepositoryFactEvaluatorError,
            "JSONDecodeError.*column 278",
        ):
            await evaluator(
                inputs={"question": "question"},
                outputs=target_outputs(),
                reference_outputs=REFERENCE_OUTPUTS,
                run=make_root(),
            )

        self.assertEqual(len(model.structured_model.invocations), 2)

    async def test_provider_failure_is_an_evaluator_error(self):
        provider_error = RuntimeError("provider unavailable")
        model = FakeChatModel(error=provider_error)
        evaluator = build_repository_fact_semantic_evaluator(model=model)

        with self.assertRaises(RepositoryFactEvaluatorError) as raised:
            await evaluator(
                inputs={"question": "question"},
                outputs=target_outputs(),
                reference_outputs=REFERENCE_OUTPUTS,
                run=make_root(),
            )

        self.assertIs(raised.exception.__cause__, provider_error)
        self.assertEqual(len(model.structured_model.invocations), 1)


if __name__ == "__main__":
    unittest.main()
