from __future__ import annotations

import json
from dataclasses import dataclass, field
from typing import Any, Literal, TypedDict
from uuid import UUID

from langsmith import schemas

TraceEvidenceStatus = Literal["available", "partial", "unavailable"]

_MAX_TEXT_CHARS = 12_000
_MAX_PROJECTION_CHARS = 60_000
_TRUNCATION_MARKER = "\n[trace evidence truncated]"


class ToolEvidence(TypedDict):
    """Stable, JSON-compatible evidence projected from one tool Run."""

    tool_name: str
    inputs: dict[str, Any]
    output: str | None
    error: str | None


class TraceEvidence(TypedDict):
    """Tool evidence and the completeness of its source Trace."""

    status: TraceEvidenceStatus
    evidence: list[ToolEvidence]


@dataclass
class _ProjectionState:
    evidence: list[ToolEvidence] = field(default_factory=list)
    visited_run_ids: set[UUID] = field(default_factory=set)
    declared_child_ids: set[UUID] = field(default_factory=set)
    projected_chars: int = 0
    partial: bool = False


def project_tool_evidence(run: schemas.Run | None) -> TraceEvidence:
    """Project a loaded LangSmith Run tree into stable tool evidence.

    The root Run must have its nested runs loaded. A missing root or a root with no loaded ``child_runs`` is reported as unavailable rather than as an empty, fully observed Trace.
    """

    if run is None or run.child_runs is None:
        return {"status": "unavailable", "evidence": []}

    state = _ProjectionState()
    _visit_run(run, state)
    if state.declared_child_ids - state.visited_run_ids:
        state.partial = True

    status: TraceEvidenceStatus = "partial" if state.partial else "available"
    return {"status": status, "evidence": state.evidence}


def _visit_run(run: schemas.Run, state: _ProjectionState) -> None:
    if run.id in state.visited_run_ids:
        state.partial = True
        return

    state.visited_run_ids.add(run.id)
    state.declared_child_ids.update(run.child_run_ids or [])

    if run.run_type == "tool":
        _append_tool_evidence(run, state)

    for child in sorted(run.child_runs or [], key=_run_order):
        _visit_run(child, state)


def _append_tool_evidence(run: schemas.Run, state: _ProjectionState) -> None:
    output, output_truncated = _bounded_text(_tool_output(run))
    error, error_truncated = _bounded_text(run.error)
    item: ToolEvidence = {
        "tool_name": run.name,
        "inputs": _json_mapping(run.inputs),
        "output": output,
        "error": error,
    }
    item_chars = len(
        json.dumps(
            item,
            ensure_ascii=False,
            separators=(",", ":"),
            sort_keys=True,
        )
    )

    if state.projected_chars + item_chars > _MAX_PROJECTION_CHARS:
        state.partial = True
        return

    state.evidence.append(item)
    state.projected_chars += item_chars
    if output_truncated or error_truncated:
        state.partial = True


def _tool_output(run: schemas.Run) -> Any:
    outputs = run.outputs
    if isinstance(outputs, dict) and "output" in outputs:
        return outputs["output"]
    return outputs


def _bounded_text(value: Any) -> tuple[str | None, bool]:
    if value is None:
        return None, False

    text = value if isinstance(value, str) else _json_text(value)
    if len(text) <= _MAX_TEXT_CHARS:
        return text, False

    content_chars = _MAX_TEXT_CHARS - len(_TRUNCATION_MARKER)
    return text[:content_chars] + _TRUNCATION_MARKER, True


def _json_mapping(value: Any) -> dict[str, Any]:
    normalized = _json_value(value)
    if not isinstance(normalized, dict):
        raise TypeError("tool Run inputs must be an object")
    return normalized


def _json_text(value: Any) -> str:
    return json.dumps(
        _json_value(value),
        ensure_ascii=False,
        separators=(",", ":"),
        sort_keys=True,
    )


def _json_value(value: Any) -> Any:
    return json.loads(
        json.dumps(
            value,
            default=str,
            ensure_ascii=False,
            sort_keys=True,
        )
    )


def _run_order(run: schemas.Run) -> tuple[int, str, str]:
    if run.dotted_order:
        return 0, run.dotted_order, str(run.id)
    return 1, run.start_time.isoformat(), str(run.id)
