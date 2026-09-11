from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass
from typing import Any, Protocol, TypedDict
from uuid import uuid4

from langchain_agent.app.config import AppConfig
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.permissions.models import PermissionMode


class AgentInvoker(Protocol):
    async def ainvoke(
        self,
        input_value: dict[str, Any],
        *,
        config: dict[str, Any],
        context: AgentContext,
    ) -> Mapping[str, Any]: ...


class RepositoryFactTargetOutput(TypedDict):
    answer: str
    git_audit_status: str
    edited_files: list[str]


@dataclass(frozen=True)
class RepositoryFactTarget:
    """Adapt one repository-fact example to an isolated Agent invocation."""

    agent: AgentInvoker
    context: AgentContext
    config: AppConfig
    fixture_version: str

    def __post_init__(self) -> None:
        if self.context.permission_mode is not PermissionMode.READ_ONLY:
            raise ValueError("repository-fact evaluation requires read-only context")

        if not self.fixture_version.strip():
            raise ValueError("fixture_version must not be empty")

    async def __call__(
        self,
        inputs: Mapping[str, Any],
    ) -> RepositoryFactTargetOutput:
        question = _validate_inputs(inputs)
        thread_id = f"eval-{uuid4().hex}"
        invoke_config = {
            "configurable": {
                "thread_id": thread_id,
            },
            "run_name": "repository_fact_target",
            "tags": ["langchain-agent", "evaluation", "repository_fact"],
            "metadata": {
                "agent_version": self.config.agent_version,
                "fixture_version": self.fixture_version,
                "permission_mode": self.context.permission_mode.value,
            },
        }
        result = await self.agent.ainvoke(
            {
                "messages": [
                    {
                        "role": "user",
                        "content": question,
                    }
                ]
            },
            config=invoke_config,
            context=self.context,
        )

        return {
            "answer": _final_answer(result),
            "git_audit_status": _git_audit_status(result),
            "edited_files": _edited_files(result),
        }


def _validate_inputs(inputs: Mapping[str, Any]) -> str:
    expected_keys = {"question"}
    unexpected_keys = set(inputs) - expected_keys
    if unexpected_keys:
        unexpected = ", ".join(sorted(str(key) for key in unexpected_keys))
        raise ValueError(f"unexpected repository-fact input fields: {unexpected}")

    return _required_text(inputs, "question")


def _required_text(inputs: Mapping[str, Any], key: str) -> str:
    value = inputs.get(key)
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{key} must be a non-empty string")

    return value.strip()


def _final_answer(result: Mapping[str, Any]) -> str:
    messages = result.get("messages")
    if (
        not isinstance(messages, Sequence)
        or isinstance(messages, (str, bytes))
        or not messages
    ):
        raise ValueError("Agent result must contain at least one message")

    content = getattr(messages[-1], "content", None)
    if not isinstance(content, str):
        raise TypeError("final Agent message content must be a string")

    return content


def _git_audit_status(result: Mapping[str, Any]) -> str:
    status = result.get("git_audit_status")
    if status not in {"available", "unavailable"}:
        raise ValueError("Agent result must contain a valid git_audit_status")

    return status


def _edited_files(result: Mapping[str, Any]) -> list[str]:
    edited_files = result.get("edited_file_list")
    if not isinstance(edited_files, list) or not all(
        isinstance(path, str) for path in edited_files
    ):
        raise ValueError("Agent result must contain a string edited_file_list")

    return list(edited_files)
