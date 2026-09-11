"""Compose authorization and retry policy before invoking a tool."""

from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

from harbor_tasks.authorization import authorize
from harbor_tasks.policies import get_tool_policy
from harbor_tasks.retry import maximum_attempts


@dataclass(frozen=True)
class ExecutionResult:
    status: str
    attempts: int
    value: Any = None


def execute_tool(
    tool_name: str,
    invoke: Callable[[], Any],
) -> ExecutionResult:
    """Authorize a tool, then invoke it within the allowed attempt budget."""

    decision = authorize(tool_name)
    if not decision.allowed:
        return ExecutionResult(
            status="denied",
            attempts=0,
        )

    policy = get_tool_policy(tool_name)
    if policy is None:
        raise RuntimeError("authorized tool must have a registered policy")

    attempt_limit = maximum_attempts(policy)

    for attempt in range(1, attempt_limit + 1):
        try:
            return ExecutionResult(
                status="completed",
                attempts=attempt,
                value=invoke(),
            )
        except TimeoutError:
            if attempt == attempt_limit:
                return ExecutionResult(
                    status="exhausted",
                    attempts=attempt,
                )

    raise AssertionError("attempt loop must return a result")
