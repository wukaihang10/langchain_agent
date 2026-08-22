from langgraph.prebuilt.tool_node import ToolCallRequest

from langchain_agent.permissions.types import (
    PermissionAction,
    PermissionDecision,
    PermissionMode,
    ToolCategory,
    ToolRisk,
    ToolPolicy,
)
from langchain_agent.context import AgentContext

TRUSTED_METADATA_KEYS = {
    "category",
    "idempotent",
    "side_effect",
    "risk",
}


def check_permission(
    *,
    tool_call: dict,
    policy: ToolPolicy | None,
    context: AgentContext,
) -> PermissionDecision:
    if context.permission_mode == PermissionMode.FULL_ACCESS:
        return PermissionDecision(action=PermissionAction.ALLOW)

    if not policy:
        if context.permission_mode == PermissionMode.READ_ONLY:
            return PermissionDecision(
                action=PermissionAction.DENY,
                reason=(
                    "This tool has no trusted local "
                    "permission metadata, so it "
                    "cannot be verified as read-only."
                ),
            )

        return PermissionDecision(
            action=PermissionAction.ASK,
            reason=("This tool has no trusted local " "permission metadata."),
        )

    if context.permission_mode == PermissionMode.READ_ONLY and policy.side_effect:
        return PermissionDecision(
            action=PermissionAction.DENY,
            reason=("Write operations are disabled " "in read-only mode."),
        )

    if context.permission_mode == PermissionMode.DEFAULT and policy.side_effect:
        return PermissionDecision(
            action=PermissionAction.ASK,
            reason=("This tool has side effects."),
        )

    return PermissionDecision(action=PermissionAction.ALLOW)
