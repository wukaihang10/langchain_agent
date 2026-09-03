from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, replace
from enum import StrEnum
from typing import Any

from langchain_core.messages import AIMessage, ToolMessage
from langgraph.types import Command

from langchain_agent.app.session_runtime import SessionRuntime
from langchain_agent.harness.middleware.turn_recovery import (
    RecoveryDirectiveKind,
    TurnRecoveryMode,
    TurnRecoveryPlan,
    ToolRecoveryDirective,
)
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry
from langchain_agent.persistence.sessions import SessionStore


class ContinuationStatus(StrEnum):
    EMPTY = "EMPTY"
    READY = "READY"
    WAITING_HUMAN = "WAITING_HUMAN"
    RESUMABLE = "RESUMABLE"
    OUTCOME_UNKNOWN = "OUTCOME_UNKNOWN"
    NEEDS_REPAIR = "NEEDS_REPAIR"


class ContinuationAction(StrEnum):
    START_TURN = "START_TURN"
    ANSWER_INTERRUPT = "ANSWER_INTERRUPT"
    CONTINUE = "CONTINUE"
    TERMINATE_TURN = "TERMINATE_TURN"


@dataclass(frozen=True)
class PendingInterrupt:
    id: str
    value: Any


@dataclass(frozen=True)
class UnresolvedToolCall:
    id: str
    name: str
    args: Any
    replay_safe: bool
    policy_known: bool
    outcome_unknown: bool = False


@dataclass(frozen=True)
class ContinuationInspection:
    status: ContinuationStatus
    checkpoint_id: str | None
    pending_nodes: tuple[str, ...]
    interrupts: tuple[PendingInterrupt, ...]
    unresolved_tool_calls: tuple[UnresolvedToolCall, ...]
    allowed_actions: frozenset[ContinuationAction]
    reason: str


@dataclass(frozen=True)
class ContinuationRequest:
    action: ContinuationAction
    observed_checkpoint_id: str | None
    message: str | None = None
    decisions: tuple[dict[str, Any], ...] = ()


@dataclass(frozen=True)
class ContinuationResult:
    value: Any
    inspection: ContinuationInspection


class ContinuationError(RuntimeError):
    """Base error for a rejected continuation request."""


class StaleContinuationError(ContinuationError):
    """The inspected checkpoint is no longer the latest checkpoint."""


class InvalidContinuationAction(ContinuationError):
    """The requested action is not valid for the current session status."""


_ALLOWED_ACTIONS = {
    ContinuationStatus.EMPTY: frozenset({ContinuationAction.START_TURN}),
    ContinuationStatus.READY: frozenset({ContinuationAction.START_TURN}),
    ContinuationStatus.WAITING_HUMAN: frozenset(
        {
            ContinuationAction.ANSWER_INTERRUPT,
            ContinuationAction.TERMINATE_TURN,
        }
    ),
    ContinuationStatus.RESUMABLE: frozenset(
        {ContinuationAction.CONTINUE, ContinuationAction.TERMINATE_TURN}
    ),
    ContinuationStatus.OUTCOME_UNKNOWN: frozenset(
        {
            ContinuationAction.CONTINUE,
            ContinuationAction.TERMINATE_TURN,
        }
    ),
    ContinuationStatus.NEEDS_REPAIR: frozenset(),
}


class SessionContinuation:
    """Inspect and execute the only valid operation for a persisted session."""

    def __init__(
        self,
        *,
        agent: Any,
        policy_registry: ToolPolicyRegistry,
        session_store: SessionStore,
    ) -> None:
        self._agent = agent
        self._policy_registry = policy_registry
        self._session_store = session_store

    async def inspect(
        self,
        runtime: SessionRuntime,
    ) -> ContinuationInspection:
        snapshot = await self._agent.aget_state(runtime.invoke_config)
        return self._classify(snapshot)

    async def execute(
        self,
        runtime: SessionRuntime,
        request: ContinuationRequest,
    ) -> ContinuationResult:
        latest = await self.inspect(runtime)

        if latest.checkpoint_id != request.observed_checkpoint_id:
            raise StaleContinuationError(
                "The session checkpoint changed after inspection; inspect the "
                "session again before continuing."
            )

        if request.action not in latest.allowed_actions:
            raise InvalidContinuationAction(
                f"{request.action.value} is not allowed while the session is "
                f"{latest.status.value}: {latest.reason}"
            )

        input_value = self._build_input(latest, request)
        invocation_context = runtime.context
        if (
            request.action == ContinuationAction.CONTINUE
            and latest.status == ContinuationStatus.OUTCOME_UNKNOWN
        ):
            recovery_plan = self._build_outcome_unknown_plan(latest)
            invocation_context = replace(
                runtime.context,
                turn_recovery=recovery_plan,
            )
        elif request.action == ContinuationAction.TERMINATE_TURN:
            recovery_plan = self._build_termination_plan(latest)
            invocation_context = replace(
                runtime.context,
                turn_recovery=recovery_plan,
            )

        # Touch only after both stale-state and action validation have succeeded,
        # immediately before real graph execution begins.
        self._session_store.touch(runtime.session.thread_id)
        value = await self._agent.ainvoke(
            input_value,
            config=runtime.invoke_config,
            context=invocation_context,
        )
        return ContinuationResult(
            value=value,
            inspection=await self.inspect(runtime),
        )

    def _build_input(
        self,
        inspection: ContinuationInspection,
        request: ContinuationRequest,
    ) -> Any:
        if request.action == ContinuationAction.START_TURN:
            if request.message is None or not request.message.strip():
                raise InvalidContinuationAction(
                    "START_TURN requires a non-empty human message."
                )

            return {
                "messages": [
                    {
                        "role": "user",
                        "content": request.message,
                    }
                ]
            }

        if request.action == ContinuationAction.ANSWER_INTERRUPT:
            if not request.decisions:
                raise InvalidContinuationAction(
                    "ANSWER_INTERRUPT requires at least one human decision."
                )

            return Command(resume={"decisions": list(request.decisions)})

        if request.action == ContinuationAction.TERMINATE_TURN:
            if request.message is not None or request.decisions:
                raise InvalidContinuationAction(
                    "TERMINATE_TURN does not accept a human message or HITL "
                    "decisions."
                )
            if inspection.interrupts:
                decisions = _termination_hitl_decisions(inspection.interrupts)
                if not decisions:
                    raise InvalidContinuationAction(
                        "TERMINATE_TURN cannot resume a human interrupt that has "
                        "no HITL action requests."
                    )
                return Command(resume={"decisions": decisions})

        return None

    def _build_outcome_unknown_plan(
        self,
        inspection: ContinuationInspection,
    ) -> TurnRecoveryPlan:
        directives = {
            call.id: ToolRecoveryDirective(RecoveryDirectiveKind.OUTCOME_UNKNOWN)
            for call in inspection.unresolved_tool_calls
            if call.outcome_unknown
        }
        if not directives:
            raise InvalidContinuationAction(
                "OUTCOME_UNKNOWN continuation has no uncertain tool calls to "
                "record. Inspect the latest checkpoint before continuing."
            )
        return TurnRecoveryPlan(TurnRecoveryMode.CONTINUE, directives)

    def _build_termination_plan(
        self,
        inspection: ContinuationInspection,
    ) -> TurnRecoveryPlan:
        protected_call_ids, errors = _match_hitl_calls(
            inspection.interrupts,
            inspection.unresolved_tool_calls,
        )
        if errors:
            raise InvalidContinuationAction(
                "TERMINATE_TURN cannot safely match the pending HITL requests: "
                f"{errors[0]}"
            )

        directives = {
            call.id: ToolRecoveryDirective(
                (
                    RecoveryDirectiveKind.CANCELLED_BY_TERMINATION
                    if call.replay_safe
                    else RecoveryDirectiveKind.OUTCOME_UNKNOWN_AT_TERMINATION
                )
            )
            for call in inspection.unresolved_tool_calls
            if call.id not in protected_call_ids
        }
        return TurnRecoveryPlan(TurnRecoveryMode.TERMINATE, directives)

    def _classify(self, snapshot: Any) -> ContinuationInspection:
        if snapshot is None:
            return _inspection(
                status=ContinuationStatus.EMPTY,
                checkpoint_id=None,
                reason="The session has no checkpoint and can start its first turn.",
            )

        checkpoint_id = _checkpoint_id(snapshot)
        messages = _messages(snapshot)
        tasks = tuple(getattr(snapshot, "tasks", ()) or ())
        pending_nodes = _pending_nodes(snapshot, tasks)
        pending_task_ids = _pending_task_ids(tasks)
        task_errors = _task_errors(tasks)
        interrupts = _interrupts(snapshot, tasks)
        unresolved, protocol_errors = self._inspect_messages(messages)

        has_material_state = bool(
            messages or pending_nodes or interrupts or getattr(snapshot, "values", {})
        )
        if checkpoint_id is None and not has_material_state:
            return _inspection(
                status=ContinuationStatus.EMPTY,
                checkpoint_id=None,
                reason="The session has no checkpoint and can start its first turn.",
            )

        if checkpoint_id is None:
            return _inspection(
                status=ContinuationStatus.NEEDS_REPAIR,
                checkpoint_id=None,
                pending_nodes=pending_nodes,
                interrupts=interrupts,
                unresolved=unresolved,
                reason=(
                    "Checkpoint state exists without a checkpoint identity; graph "
                    "execution is blocked until the persisted state is repaired."
                ),
            )

        interrupt_call_ids, interrupt_errors = _match_hitl_calls(
            interrupts,
            unresolved,
        )
        structural_errors = [*protocol_errors, *interrupt_errors]
        unprotected = [call for call in unresolved if call.id not in interrupt_call_ids]

        if structural_errors:
            unknown_outcome_calls = [
                call for call in unprotected if not call.replay_safe
            ]
            uncertainty_warning = ""
            if unknown_outcome_calls:
                names = ", ".join(
                    f"{call.name} ({call.id})" for call in unknown_outcome_calls
                )
                uncertainty_warning = (
                    " External outcome may be unknown for unresolved tool call(s): "
                    f"{names}. Verify external state before any future repair."
                )

            return _inspection(
                status=ContinuationStatus.NEEDS_REPAIR,
                checkpoint_id=checkpoint_id,
                pending_nodes=pending_nodes,
                interrupts=interrupts,
                unresolved=unresolved,
                reason=(
                    "Persisted state is structurally inconsistent: "
                    f"{structural_errors[0]} No checkpoint history was modified."
                    f"{uncertainty_warning}"
                ),
            )

        has_continuation_path = bool(
            (getattr(snapshot, "next", ()) or ()) or pending_task_ids
        )
        # LangChain create_agent dispatches tool calls through the "tools" node.
        # A pending model node cannot consume tool-call recovery directives.
        has_tool_continuation_path = "tools" in pending_nodes
        uncertain = [
            call
            for call in unprotected
            if has_tool_continuation_path and not call.replay_safe
        ]
        uncertain_ids = {call.id for call in uncertain}
        unresolved = tuple(
            replace(
                call,
                outcome_unknown=call.id in uncertain_ids,
            )
            for call in unresolved
        )

        if uncertain:
            names = ", ".join(f"{call.name} ({call.id})" for call in uncertain)
            return _inspection(
                status=ContinuationStatus.OUTCOME_UNKNOWN,
                checkpoint_id=checkpoint_id,
                pending_nodes=pending_nodes,
                interrupts=interrupts,
                unresolved=unresolved,
                reason=(
                    "External outcome is unknown for pending tool call(s): "
                    f"{names}. Continuing records an uncertain tool result without "
                    "replaying these calls, then returns control to the Agent."
                ),
            )

        if interrupts:
            return _inspection(
                status=ContinuationStatus.WAITING_HUMAN,
                checkpoint_id=checkpoint_id,
                pending_nodes=pending_nodes,
                interrupts=interrupts,
                unresolved=unresolved,
                reason=(
                    "The graph is waiting for a persisted human decision. Review "
                    "the original request before answering the interrupt."
                ),
            )

        if unresolved and not has_tool_continuation_path:
            ids = ", ".join(call.id for call in unresolved)
            uncertainty_warning = (
                " The external outcome may be unknown for one or more calls; "
                "ordinary continuation is still blocked because no pending tool "
                "execution path exists."
                if any(not call.replay_safe for call in unresolved)
                else ""
            )
            return _inspection(
                status=ContinuationStatus.NEEDS_REPAIR,
                checkpoint_id=checkpoint_id,
                pending_nodes=pending_nodes,
                unresolved=unresolved,
                reason=(
                    "Tool call(s) have no matching result and no pending tool "
                    f"execution path: {ids}. The history requires explicit repair."
                    f"{uncertainty_warning}"
                ),
            )

        if has_continuation_path:
            error_detail = (
                " Recorded task errors will be replayed only under the same "
                f"safety policy: {'; '.join(task_errors)}."
                if task_errors
                else ""
            )
            return _inspection(
                status=ContinuationStatus.RESUMABLE,
                checkpoint_id=checkpoint_id,
                pending_nodes=pending_nodes,
                unresolved=unresolved,
                reason=(
                    "The graph has pending work and every unresolved operation is "
                    f"classified as safe to replay.{error_detail}"
                ),
            )

        return _inspection(
            status=ContinuationStatus.READY,
            checkpoint_id=checkpoint_id,
            reason="The previous graph execution completed and a new turn may start.",
        )

    def _inspect_messages(
        self,
        messages: Sequence[Any],
    ) -> tuple[tuple[UnresolvedToolCall, ...], list[str]]:
        known_ids: set[str] = set()
        unresolved: dict[str, UnresolvedToolCall] = {}
        completed_ids: set[str] = set()
        errors: list[str] = []

        for message in messages:
            tool_calls = _tool_calls(message)
            if tool_calls is not None:
                if unresolved:
                    errors.append(
                        "A new AI message appears before earlier tool calls were "
                        "resolved."
                    )

                for call in tool_calls:
                    call_id = call.get("id")
                    name = call.get("name")
                    if not isinstance(call_id, str) or not call_id:
                        errors.append("An AI tool call is missing a stable ID.")
                        continue
                    if not isinstance(name, str) or not name:
                        errors.append(f"Tool call {call_id} is missing its tool name.")
                        continue
                    if call_id in known_ids:
                        errors.append(f"Tool call ID {call_id} is duplicated.")
                        continue

                    known_ids.add(call_id)
                    policy = self._policy_registry.get(name)
                    unresolved[call_id] = UnresolvedToolCall(
                        id=call_id,
                        name=name,
                        args=call.get("args", {}),
                        replay_safe=(
                            policy is not None
                            and policy.idempotent
                            and not policy.side_effect
                        ),
                        policy_known=policy is not None,
                    )
                continue

            tool_call_id = _tool_result_id(message)
            if tool_call_id is not None:
                if tool_call_id in unresolved:
                    del unresolved[tool_call_id]
                    completed_ids.add(tool_call_id)
                elif tool_call_id in completed_ids:
                    errors.append(
                        f"Tool call {tool_call_id} has more than one result message."
                    )
                else:
                    errors.append(
                        f"Tool result {tool_call_id} has no matching AI tool call."
                    )
                continue

            if unresolved:
                errors.append(
                    "A non-tool message appears before pending tool calls were "
                    "resolved."
                )

        return tuple(unresolved.values()), errors


def _inspection(
    *,
    status: ContinuationStatus,
    checkpoint_id: str | None,
    pending_nodes: tuple[str, ...] = (),
    interrupts: tuple[PendingInterrupt, ...] = (),
    unresolved: tuple[UnresolvedToolCall, ...] = (),
    reason: str,
) -> ContinuationInspection:
    return ContinuationInspection(
        status=status,
        checkpoint_id=checkpoint_id,
        pending_nodes=pending_nodes,
        interrupts=interrupts,
        unresolved_tool_calls=unresolved,
        allowed_actions=_ALLOWED_ACTIONS[status],
        reason=reason,
    )


def _checkpoint_id(snapshot: Any) -> str | None:
    config = getattr(snapshot, "config", None) or {}
    configurable = config.get("configurable", {})
    checkpoint_id = configurable.get("checkpoint_id")
    return checkpoint_id if isinstance(checkpoint_id, str) else None


def _messages(snapshot: Any) -> tuple[Any, ...]:
    values = getattr(snapshot, "values", None)
    if not isinstance(values, Mapping):
        return ()
    messages = values.get("messages", ())
    return tuple(messages) if isinstance(messages, Sequence) else ()


def _pending_nodes(snapshot: Any, tasks: tuple[Any, ...]) -> tuple[str, ...]:
    names = [
        name for name in (getattr(snapshot, "next", ()) or ()) if isinstance(name, str)
    ]
    names.extend(
        task.name for task in tasks if isinstance(getattr(task, "name", None), str)
    )
    return tuple(dict.fromkeys(names))


def _pending_task_ids(tasks: tuple[Any, ...]) -> tuple[str, ...]:
    return tuple(
        task_id
        for task in tasks
        if isinstance((task_id := getattr(task, "id", None)), str)
    )


def _task_errors(tasks: tuple[Any, ...]) -> tuple[str, ...]:
    details = []
    for task in tasks:
        error = getattr(task, "error", None)
        if error is None:
            continue
        task_id = getattr(task, "id", "unknown-task")
        task_name = getattr(task, "name", "unknown-node")
        details.append(f"{task_name} [{task_id}] {type(error).__name__}: {error}")
    return tuple(details)


def _interrupts(
    snapshot: Any,
    tasks: tuple[Any, ...],
) -> tuple[PendingInterrupt, ...]:
    raw = list(getattr(snapshot, "interrupts", ()) or ())
    for task in tasks:
        raw.extend(getattr(task, "interrupts", ()) or ())

    result: list[PendingInterrupt] = []
    seen: set[str] = set()
    for item in raw:
        interrupt_id = getattr(item, "id", None)
        if not isinstance(interrupt_id, str) or interrupt_id in seen:
            continue
        seen.add(interrupt_id)
        result.append(
            PendingInterrupt(
                id=interrupt_id,
                value=getattr(item, "value", None),
            )
        )
    return tuple(result)


def _tool_calls(message: Any) -> Sequence[Mapping[str, Any]] | None:
    if isinstance(message, AIMessage):
        return message.tool_calls
    if isinstance(message, Mapping) and message.get("type") in {"ai", "assistant"}:
        calls = message.get("tool_calls", ())
        return calls if isinstance(calls, Sequence) else ()
    return None


def _tool_result_id(message: Any) -> str | None:
    if isinstance(message, ToolMessage):
        return message.tool_call_id
    if isinstance(message, Mapping) and message.get("type") in {"tool"}:
        value = message.get("tool_call_id")
        return value if isinstance(value, str) else None
    return None


def _match_hitl_calls(
    interrupts: tuple[PendingInterrupt, ...],
    unresolved: tuple[UnresolvedToolCall, ...],
) -> tuple[set[str], list[str]]:
    matched: set[str] = set()
    errors: list[str] = []

    for interrupt_item in interrupts:
        value = interrupt_item.value
        if not isinstance(value, Mapping):
            continue
        requests = value.get("action_requests")
        if requests is None:
            continue
        if not isinstance(requests, Sequence):
            errors.append(f"Interrupt {interrupt_item.id} has invalid action requests.")
            continue

        for action in requests:
            if not isinstance(action, Mapping):
                errors.append(f"Interrupt {interrupt_item.id} has a malformed action.")
                continue
            name = action.get("name")
            args = action.get("args", {})
            match = next(
                (
                    call
                    for call in unresolved
                    if call.id not in matched
                    and call.name == name
                    and call.args == args
                ),
                None,
            )
            if match is None:
                errors.append(
                    f"Interrupt {interrupt_item.id} does not match an unresolved "
                    "tool call."
                )
            else:
                matched.add(match.id)

    return matched, errors


def _termination_hitl_decisions(
    interrupts: tuple[PendingInterrupt, ...],
) -> list[dict[str, str]]:
    decisions: list[dict[str, str]] = []
    for interrupt_item in interrupts:
        value = interrupt_item.value
        if not isinstance(value, Mapping):
            continue
        action_requests = value.get("action_requests", ())
        if not isinstance(action_requests, Sequence):
            continue
        decisions.extend(
            {
                "type": "reject",
                "message": "User terminated the turn.",
            }
            for action in action_requests
            if isinstance(action, Mapping)
        )
    return decisions
