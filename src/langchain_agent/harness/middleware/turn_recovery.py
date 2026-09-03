from __future__ import annotations

import asyncio
from collections.abc import Awaitable, Callable, Mapping
from dataclasses import dataclass, field
from enum import StrEnum
from types import MappingProxyType

from langchain.agents.middleware import AgentMiddleware
from langchain.agents.middleware.types import ModelRequest, ModelResponse
from langchain_core.messages import AIMessage, ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command


class TurnRecoveryMode(StrEnum):
    CONTINUE = "CONTINUE"
    TERMINATE = "TERMINATE"


class RecoveryDirectiveKind(StrEnum):
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    CANCELLED_BY_TERMINATION = "CANCELLED_BY_TERMINATION"
    OUTCOME_UNKNOWN_AT_TERMINATION = "OUTCOME_UNKNOWN_AT_TERMINATION"


@dataclass(frozen=True)
class ToolRecoveryDirective:
    kind: RecoveryDirectiveKind


@dataclass
class TurnRecoveryPlan:
    """Invocation-scoped recovery behavior for tool and model calls."""

    mode: TurnRecoveryMode
    directives: Mapping[str, ToolRecoveryDirective]
    _consumed: set[str] = field(default_factory=set, init=False, repr=False)
    _lock: asyncio.Lock = field(default_factory=asyncio.Lock, init=False, repr=False)

    def __post_init__(self) -> None:
        directives = dict(self.directives)
        termination_kinds = {
            RecoveryDirectiveKind.CANCELLED_BY_TERMINATION,
            RecoveryDirectiveKind.OUTCOME_UNKNOWN_AT_TERMINATION,
        }
        if self.mode == TurnRecoveryMode.TERMINATE and any(
            directive.kind not in termination_kinds
            for directive in directives.values()
        ):
            raise ValueError(
                "TERMINATE plans may contain only synthetic termination errors."
            )
        if self.mode == TurnRecoveryMode.CONTINUE and any(
            directive.kind != RecoveryDirectiveKind.OUTCOME_UNKNOWN
            for directive in directives.values()
        ):
            raise ValueError(
                "CONTINUE plans may contain only outcome-unknown directives."
            )
        self.directives = MappingProxyType(directives)

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


class TurnRecoveryMiddleware(AgentMiddleware):
    """Enforce one-invocation continuation or termination at Agent seams."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: Callable[
            [ToolCallRequest],
            Awaitable[ToolMessage | Command],
        ],
    ) -> ToolMessage | Command:
        plan = _plan_from(request)
        if plan is None:
            return await handler(request)

        tool_call_id = request.tool_call["id"]
        directive = await plan.claim(tool_call_id)
        if directive is None:
            if plan.mode == TurnRecoveryMode.TERMINATE:
                raise RuntimeError(
                    "Turn termination has no recovery directive for tool call "
                    f"{tool_call_id}; refusing real tool execution."
                )
            return await handler(request)
        return _synthetic_tool_message(request, directive)

    async def awrap_model_call(
        self,
        request: ModelRequest,
        handler: Callable[[ModelRequest], Awaitable[ModelResponse]],
    ) -> ModelResponse:
        plan = _plan_from(request)
        if plan is None or plan.mode == TurnRecoveryMode.CONTINUE:
            return await handler(request)

        return ModelResponse(
            result=[
                AIMessage(
                    content=(
                        "The previous turn was terminated at the user's request. "
                        "No further task work was performed."
                    ),
                    additional_kwargs={
                        "recovery": {
                            "generated": True,
                            "action": "TERMINATE_TURN",
                        }
                    },
                )
            ]
        )


def _plan_from(request: ToolCallRequest | ModelRequest) -> TurnRecoveryPlan | None:
    runtime = request.runtime
    context = runtime.context if runtime is not None else None
    return getattr(context, "turn_recovery", None)


def _synthetic_tool_message(
    request: ToolCallRequest,
    directive: ToolRecoveryDirective,
) -> ToolMessage:
    tool_name = request.tool_call["name"]
    termination_outcome: str | None = None
    if directive.kind == RecoveryDirectiveKind.OUTCOME_UNKNOWN:
        content = (
            f"Recovery-generated result for `{tool_name}`: the external operation "
            "may have succeeded or failed before execution was interrupted. No "
            "retry was performed during recovery. Do not infer success or failure "
            "from this message. Verify external state with a read-only operation "
            "when possible before proposing any new execution; if verification is "
            "unavailable, report the uncertainty to the user."
        )
    elif directive.kind == RecoveryDirectiveKind.CANCELLED_BY_TERMINATION:
        content = (
            f"Recovery-generated result for `{tool_name}`: the user terminated "
            "the turn, so this pending call was cancelled. No new tool execution "
            "occurred during termination."
        )
        termination_outcome = "cancelled"
    elif directive.kind == RecoveryDirectiveKind.OUTCOME_UNKNOWN_AT_TERMINATION:
        content = (
            f"Recovery-generated result for `{tool_name}`: the user terminated "
            "the turn, but the earlier external operation may have succeeded or "
            "failed. It was not retried."
        )
        termination_outcome = "outcome_unknown"
    else:
        raise ValueError(
            "Synthetic recovery directive has unsupported kind: "
            f"{directive.kind}"
        )

    recovery_metadata = {
        "generated": True,
        "kind": directive.kind.value,
    }
    outcome = termination_outcome
    if directive.kind == RecoveryDirectiveKind.OUTCOME_UNKNOWN:
        outcome = "outcome_unknown"
    if outcome is not None:
        recovery_metadata["outcome"] = outcome
    if termination_outcome is not None:
        recovery_metadata.update(
            {
                "action": "TERMINATE_TURN",
            }
        )

    return ToolMessage(
        content=content,
        tool_call_id=request.tool_call["id"],
        status="error",
        additional_kwargs={"recovery": recovery_metadata},
    )
