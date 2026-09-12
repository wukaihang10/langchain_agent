from __future__ import annotations

import asyncio
import json
from collections.abc import Awaitable, Callable, Mapping, Sequence
from datetime import UTC, datetime
from pathlib import Path
from typing import TYPE_CHECKING, Any
from uuid import NAMESPACE_URL, UUID, uuid5

from dotenv import load_dotenv
from langsmith import Client, schemas, tracing_context
from langsmith.evaluation import EvaluationResults

from evals.evaluators.repository_fact import (
    build_repository_fact_semantic_evaluator,
)
from langchain_agent.app.config import AppPaths
from langchain_agent.integrations.model import create_model

if TYPE_CHECKING:
    from langchain_core.language_models import BaseChatModel


PROJECT_ROOT = Path(__file__).resolve().parents[2]
CALIBRATION_PATH = (
    PROJECT_ROOT
    / "tests"
    / "fixtures"
    / "evaluators"
    / "repository_fact"
    / "semantic_v0.jsonl"
)
BASELINE_EXPERIMENT_ID = "58613711-f185-4232-bfdc-8c5b0818401c"
MANUAL_REVIEW_VERSION = "repository-fact-manual-v0"
_CALIBRATION_TRACE_ID = UUID("00000000-0000-0000-0000-000000000003")


def load_repository_fact_calibration_cases(
    path: Path = CALIBRATION_PATH,
) -> list[dict[str, Any]]:
    """Load the local semantic-evaluator calibration cases."""

    cases = []
    with path.open(encoding="utf-8") as calibration_file:
        for line_number, line in enumerate(calibration_file, start=1):
            if not line.strip():
                continue
            try:
                case = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on calibration line {line_number}: {error.msg}"
                ) from error
            if not isinstance(case, dict):
                raise TypeError(
                    f"calibration line {line_number} must contain an object"
                )
            cases.append(case)
    return cases


async def calibrate_repository_fact_semantic_evaluator(
    *,
    model: BaseChatModel,
    client: Client | None = None,
    experiment_id: str = BASELINE_EXPERIMENT_ID,
) -> dict[str, Any]:
    """Compare the semantic evaluator with synthetic and reviewed gold labels."""

    evaluator = build_repository_fact_semantic_evaluator(model=model)
    langsmith_client = client or Client()
    with tracing_context(enabled=False):
        fixture_rows = await _calibrate_fixture(evaluator)
        baseline_rows = await _calibrate_baseline(
            evaluator,
            client=langsmith_client,
            experiment_id=experiment_id,
        )

    return {
        "fixture": _summary(fixture_rows),
        "baseline": _summary(baseline_rows),
        "rows": {
            "fixture": fixture_rows,
            "baseline": baseline_rows,
        },
    }


async def _calibrate_fixture(
    evaluator: Callable[..., Awaitable[EvaluationResults]],
) -> list[dict[str, Any]]:
    rows = []
    for case in load_repository_fact_calibration_cases():
        outputs = {
            **case["outputs"],
            "git_audit_status": "available",
            "edited_files": [],
        }
        results = await evaluator(
            inputs=case["inputs"],
            outputs=outputs,
            reference_outputs=case["reference_outputs"],
            run=_calibration_run(case),
        )
        expected = {
            **case["expected_feedback"],
            "task_success": _expected_task_success(case["expected_feedback"].values()),
        }
        actual = _feedback_values(results)
        rows.append(
            {
                "case_id": case["metadata"]["case_id"],
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }
        )
    return rows


async def _calibrate_baseline(
    evaluator: Callable[..., Awaitable[EvaluationResults]],
    *,
    client: Client,
    experiment_id: str,
) -> list[dict[str, Any]]:
    roots = list(
        client.list_runs(
            project_id=experiment_id,
            is_root=True,
            limit=100,
        )
    )
    rows = []
    for shallow_root in roots:
        run = client.read_run(shallow_root.id, load_child_runs=True)
        if run.reference_example_id is None:
            raise ValueError(f"Run {run.id} has no reference Example")
        example = client.read_example(run.reference_example_id)
        expected = _manual_feedback_values(client, run.id)
        results = await evaluator(
            inputs=run.inputs,
            outputs=run.outputs or {},
            reference_outputs=example.outputs or {},
            run=run,
        )
        actual = _feedback_values(results)
        metadata = (run.extra or {}).get("metadata", {})
        rows.append(
            {
                "case_id": metadata.get("ls_example_case_id", str(run.id)),
                "expected": expected,
                "actual": actual,
                "matches": actual == expected,
            }
        )
    return sorted(rows, key=lambda row: row["case_id"])


def _calibration_run(case: Mapping[str, Any]) -> schemas.Run:
    trace = case["trace"]
    tool_runs = [
        _calibration_tool_run(case, position, evidence)
        for position, evidence in enumerate(trace["evidence"], start=1)
    ]
    child_run_ids = [run.id for run in tool_runs]
    if trace["status"] == "partial":
        child_run_ids.append(
            uuid5(
                NAMESPACE_URL,
                f"{case['metadata']['case_id']}/missing-child",
            )
        )
    elif trace["status"] == "unavailable":
        tool_runs = None
        child_run_ids = None
    elif trace["status"] != "available":
        raise ValueError(f"unsupported trace status: {trace['status']}")

    return schemas.Run(
        id=uuid5(
            NAMESPACE_URL,
            f"{case['metadata']['case_id']}/root",
        ),
        trace_id=_CALIBRATION_TRACE_ID,
        name="repository_fact_calibration",
        run_type="chain",
        start_time=datetime(2026, 9, 12, tzinfo=UTC),
        inputs=dict(case["inputs"]),
        outputs=dict(case["outputs"]),
        child_runs=tool_runs,
        child_run_ids=child_run_ids,
    )


def _calibration_tool_run(
    case: Mapping[str, Any],
    position: int,
    evidence: Mapping[str, Any],
) -> schemas.Run:
    case_id = case["metadata"]["case_id"]
    return schemas.Run(
        id=uuid5(NAMESPACE_URL, f"{case_id}/tool/{position}"),
        trace_id=_CALIBRATION_TRACE_ID,
        name=evidence["tool_name"],
        run_type="tool",
        start_time=datetime(2026, 9, 12, tzinfo=UTC),
        dotted_order=f"{position:04d}",
        inputs=dict(evidence["inputs"]),
        outputs={"output": evidence["output"]},
        error=evidence["error"],
    )


def _manual_feedback_values(client: Client, run_id: UUID) -> dict[str, str]:
    required_keys = {
        "answer_correctness",
        "evidence_groundedness",
        "task_success",
    }
    values = {
        feedback.key: feedback.value
        for feedback in client.list_feedback(run_ids=[run_id], limit=100)
        if feedback.key in required_keys
        and feedback.feedback_source is not None
        and (feedback.feedback_source.metadata or {}).get("review_version")
        == MANUAL_REVIEW_VERSION
    }
    if set(values) != required_keys or not all(
        isinstance(value, str) for value in values.values()
    ):
        raise ValueError(f"Run {run_id} has incomplete manual semantic feedback")
    return values


def _feedback_values(results: EvaluationResults) -> dict[str, str]:
    values = {
        result.key: result.value
        for result in results["results"]
        if result.key in {"answer_correctness", "evidence_groundedness", "task_success"}
    }
    if len(values) != 3 or not all(isinstance(value, str) for value in values.values()):
        raise ValueError("semantic evaluator returned incomplete feedback")
    return values


def _expected_task_success(verdicts: Sequence[str]) -> str:
    verdict_set = set(verdicts)
    if "fail" in verdict_set:
        return "fail"
    if verdict_set == {"pass"}:
        return "pass"
    return "unknown"


def _summary(rows: Sequence[Mapping[str, Any]]) -> dict[str, int]:
    matched = sum(bool(row["matches"]) for row in rows)
    return {
        "total": len(rows),
        "matched": matched,
        "mismatched": len(rows) - matched,
    }


def main() -> None:
    load_dotenv(AppPaths.user_default().environment_path, override=False)
    result = asyncio.run(
        calibrate_repository_fact_semantic_evaluator(
            model=create_model(thinking=False),
        )
    )
    print(json.dumps(result, ensure_ascii=False, indent=2))
    if result["fixture"]["mismatched"] or result["baseline"]["mismatched"]:
        raise SystemExit(1)


if __name__ == "__main__":
    main()
