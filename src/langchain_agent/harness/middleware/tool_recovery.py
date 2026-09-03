from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


class RecoveryDirectiveKind(StrEnum):
    SYNTHETIC_SUCCESS = "SYNTHETIC_SUCCESS"
    SYNTHETIC_ERROR = "SYNTHETIC_ERROR"
    RETRY = "RETRY"


@dataclass(frozen=True)
class ToolRecoveryDirective:
    kind: RecoveryDirectiveKind
    resolution_kind: str
    result_summary: str | None = None
    note: str | None = None


@dataclass
class ToolRecoveryPlan:
    """Invocation-scoped directives consumed once at the tool-call seam."""

    directives: Mapping[str, ToolRecoveryDirective]
    _consumed: set[str] = field(default_factory=set, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        self.directives = MappingProxyType(dict(self.directives))

    async def claim(self, tool_call_id: str) -> ToolRecoveryDirective | None:
        async with self._lock:
            directive = self.directives.get(tool_call_id)
            if directive is None:
                return None
            if tool_call_id in self._consumed:
                raise RuntimeError(
                    "Recovery directive for tool call "
                    f"{tool_call_id} was already consumed."
                )
            self._consumed.add(tool_call_id)
            return directive


class ToolRecoveryMiddleware(AgentMiddleware):
    """Apply one-invocation recovery directives before ordinary tool policy."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command],
        ],
    ) -> ToolMessage | Command:
        context = request.runtime.context
        plan = getattr(context, "tool_recovery", None)
        if plan is None:
            return await handler(request)

        tool_call_id = request.tool_call["id"]
        directive = await plan.claim(tool_call_id)
        if directive is None:
            return await handler(request)
        if directive.kind == RecoveryDirectiveKind.RETRY:
            return await handler(request)

        return _synthetic_tool_message(request, directive)


def _synthetic_tool_message(
    request: ToolCallRequest,
    directive: ToolRecoveryDirective,
) -> ToolMessage:
    tool_name = request.tool_call["name"]
    if directive.resolution_kind == "CONFIRM_SUCCEEDED":
        content = (
            f"Recovery-generated result for `{tool_name}`: the operation was "
            "verified by the user as successful; no new tool execution occurred."
        )
        if directive.result_summary:
            content += f" Verified result: {directive.result_summary}"
        status = "success"
    elif directive.resolution_kind == "CONFIRM_NOT_APPLIED":
        content = (
            f"Recovery-generated result for `{tool_name}`: the user verified "
            "that the operation did not take effect; no retry was performed."
        )
        if directive.note:
            content += f" Verification note: {directive.note}"
        status = "error"
    elif directive.resolution_kind == "RECORD_OUTCOME_UNKNOWN":
        content = (
            f"Recovery-generated result for `{tool_name}`: the external operation "
            "may have succeeded or failed. No retry was performed."
        )
        if directive.note:
            content += f" Note: {directive.note}"
        status = "error"
    else:
        raise ValueError(
            "Synthetic recovery directive has unsupported resolution kind: "
            f"{directive.resolution_kind}"
        )

    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call["id"],
        status=status,
        additional_kwargs={
            "recovery": {
                "generated": True,
                "resolution_kind": directive.resolution_kind,
            }
        },
    )
