from __future__ import annotations

import json
import re
from collections.abc import Iterable, Mapping
from pathlib import Path
from typing import Any

REPOSITORY_FACT_DATASET_PATH = (
    Path(__file__).resolve().parent / "repository_fact_v1.jsonl"
)
CASE_ID_PATTERN = re.compile(r"repository_fact_[0-9]{3}\Z")


def load_repository_fact_dataset(
    path: Path = REPOSITORY_FACT_DATASET_PATH,
) -> list[dict[str, Any]]:
    """Load and validate the local source used for Dataset synchronization."""

    examples: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as dataset_file:
        for line_number, line in enumerate(dataset_file, start=1):
            if not line.strip():
                continue

            try:
                example = json.loads(line)
            except json.JSONDecodeError as error:
                raise ValueError(
                    f"invalid JSON on dataset line {line_number}: {error.msg}"
                ) from error

            if not isinstance(example, dict):
                raise TypeError(
                    f"dataset line {line_number} must contain a JSON object"
                )
            examples.append(example)

    validate_repository_fact_examples(examples)
    return examples


def validate_repository_fact_examples(
    examples: Iterable[Mapping[str, Any]],
) -> None:
    """Reject examples that violate the repository-fact source contract."""

    seen_case_ids: set[str] = set()
    for position, example in enumerate(examples, start=1):
        label = f"example {position}"
        _require_exact_keys(
            example,
            {"inputs", "reference_outputs", "metadata"},
            label,
        )

        inputs = _require_mapping(example, "inputs", label)
        _require_exact_keys(inputs, {"question"}, f"{label}.inputs")
        _require_text(inputs, "question", f"{label}.inputs")

        reference = _require_mapping(example, "reference_outputs", label)
        _require_exact_keys(
            reference,
            {"required_facts", "forbidden_claims", "acceptable_evidence"},
            f"{label}.reference_outputs",
        )
        _require_text_list(
            reference,
            "required_facts",
            f"{label}.reference_outputs",
        )
        _require_text_list(
            reference,
            "forbidden_claims",
            f"{label}.reference_outputs",
        )
        _validate_evidence(reference, label)

        metadata = _require_mapping(example, "metadata", label)
        _require_exact_keys(
            metadata,
            {"slice", "case_type", "case_id"},
            f"{label}.metadata",
        )
        if _require_text(metadata, "slice", f"{label}.metadata") != "repository_fact":
            raise ValueError(f"{label}.metadata.slice must be 'repository_fact'")
        _require_text(metadata, "case_type", f"{label}.metadata")
        case_id = _require_text(metadata, "case_id", f"{label}.metadata")
        if CASE_ID_PATTERN.fullmatch(case_id) is None:
            raise ValueError(
                f"{label}.metadata.case_id must match repository_fact_[0-9]{{3}}"
            )
        if case_id in seen_case_ids:
            raise ValueError(f"duplicate repository-fact case_id: {case_id}")
        seen_case_ids.add(case_id)


def _require_mapping(
    value: Mapping[str, Any],
    key: str,
    label: str,
) -> Mapping[str, Any]:
    nested = value.get(key)
    if not isinstance(nested, Mapping):
        raise TypeError(f"{label}.{key} must be an object")
    return nested


def _require_exact_keys(
    value: Mapping[str, Any],
    expected_keys: set[str],
    label: str,
) -> None:
    actual_keys = set(value)
    if actual_keys == expected_keys:
        return

    missing = sorted(expected_keys - actual_keys)
    unexpected = sorted(str(key) for key in actual_keys - expected_keys)
    details = []
    if missing:
        details.append(f"missing: {', '.join(missing)}")
    if unexpected:
        details.append(f"unexpected: {', '.join(unexpected)}")
    raise ValueError(f"{label} fields are invalid ({'; '.join(details)})")


def _require_text(value: Mapping[str, Any], key: str, label: str) -> str:
    text = value.get(key)
    if not isinstance(text, str) or not text.strip():
        raise ValueError(f"{label}.{key} must be a non-empty string")
    return text


def _require_text_list(value: Mapping[str, Any], key: str, label: str) -> None:
    items = value.get(key)
    if not isinstance(items, list) or not items:
        raise ValueError(f"{label}.{key} must be a non-empty list")
    if any(not isinstance(item, str) or not item.strip() for item in items):
        raise ValueError(f"{label}.{key} must contain non-empty strings")


def _validate_evidence(reference: Mapping[str, Any], label: str) -> None:
    evidence = reference.get("acceptable_evidence")
    evidence_label = f"{label}.reference_outputs.acceptable_evidence"
    if not isinstance(evidence, list) or not evidence:
        raise ValueError(f"{evidence_label} must be a non-empty list")

    for position, location in enumerate(evidence, start=1):
        location_label = f"{evidence_label}[{position}]"
        if not isinstance(location, Mapping):
            raise TypeError(f"{location_label} must be an object")
        _require_exact_keys(location, {"path", "symbol"}, location_label)
        _require_text(location, "path", location_label)
        _require_text(location, "symbol", location_label)
