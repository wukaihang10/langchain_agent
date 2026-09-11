"""Authorization decisions for Harbor Tasks tool execution."""

from dataclasses import dataclass

from harbor_tasks.policies import get_tool_policy


@dataclass(frozen=True)
class AuthorizationDecision:
    allowed: bool
    reason: str


def authorize(tool_name: str) -> AuthorizationDecision:
    """Fail closed when a tool has no registered policy."""

    if get_tool_policy(tool_name) is None:
        return AuthorizationDecision(
            allowed=False,
            reason="unregistered_tool",
        )

    return AuthorizationDecision(
        allowed=True,
        reason="registered_tool",
    )
