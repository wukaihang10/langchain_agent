"""Registered tool policy data."""

from dataclasses import dataclass


@dataclass(frozen=True)
class ToolPolicy:
    idempotent: bool
    side_effect: bool


POLICIES = {
    "search_catalog": ToolPolicy(idempotent=True, side_effect=False),
    "inspect_once": ToolPolicy(idempotent=False, side_effect=False),
    "refresh_cache": ToolPolicy(idempotent=True, side_effect=True),
    "publish_manifest": ToolPolicy(idempotent=False, side_effect=True),
}


def get_tool_policy(tool_name: str) -> ToolPolicy | None:
    """Return the registered policy, or ``None`` for an unknown tool."""

    return POLICIES.get(tool_name)
