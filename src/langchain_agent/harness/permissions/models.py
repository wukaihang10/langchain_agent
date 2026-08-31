from dataclasses import dataclass
from enum import StrEnum


class ToolCategory(StrEnum):
    READ = "read"
    WRITE = "write"
    EXECUTE = "execute"


class ToolRisk(StrEnum):
    LOW = "low"
    MEDIUM = "medium"
    HIGH = "high"


class PermissionMode(StrEnum):
    DEFAULT = "default"
    READ_ONLY = "read_only"
    FULL_ACCESS = "full_access"


class PermissionAction(StrEnum):
    ALLOW = "allow"
    ASK = "ask"
    DENY = "deny"


@dataclass(frozen=True)
class ToolPolicy:
    category: ToolCategory
    idempotent: bool
    side_effect: bool
    risk: ToolRisk


@dataclass(frozen=True)
class PermissionDecision:
    action: PermissionAction
    reason: str | None = None
