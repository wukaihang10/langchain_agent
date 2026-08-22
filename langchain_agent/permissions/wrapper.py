from collections.abc import Awaitable, Callable

from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain_agent.permissions.types import (
    ToolExecutionAction,
)

ToolExecute = Callable[
    [ToolCallRequest],
    ToolMessage | Command,
]

AsyncToolExecute = Callable[
    [ToolCallRequest],
    Awaitable[ToolMessage | Command],
]


def _before_tool_execution(
    request: ToolCallRequest,
) -> ToolMessage | None:
    tool_call = request.tool_call
    tool_call_id = tool_call["id"]

    if tool_call["name"] == "write_todos":
        last_message = request.state["messages"][-1]

        write_todo_calls = [
            call for call in last_message.tool_calls if call["name"] == "write_todos"
        ]

        if len(write_todo_calls) > 1:
            return ToolMessage(
                content=(
                    "Only one write_todos call is "
                    "allowed per model turn. "
                    "Consolidate the task list into "
                    "a single write_todos call."
                ),
                tool_call_id=tool_call_id,
            )

    decisions = request.state.get(
        "tool_decisions",
        {},
    )

    decision = decisions.get(tool_call_id)

    if decision is None:
        raise RuntimeError(
            "No permission decision found for " f"tool call {tool_call_id}."
        )

    if decision.type == ToolExecutionAction.APPROVE:
        return None

    if decision.type == ToolExecutionAction.REJECT:
        return ToolMessage(
            content=(decision.message or "Tool execution was rejected."),
            tool_call_id=tool_call_id,
            status="error",
        )

    raise ValueError(f"Unsupported tool decision: {decision}")


def tool_wrapper(
    request: ToolCallRequest,
    execute: ToolExecute,
) -> ToolMessage | Command:
    short_circuit = _before_tool_execution(request)

    if short_circuit is not None:
        return short_circuit

    return execute(request)


async def async_tool_wrapper(
    request: ToolCallRequest,
    execute: AsyncToolExecute,
) -> ToolMessage | Command:
    short_circuit = _before_tool_execution(request)

    if short_circuit is not None:
        return short_circuit

    return await execute(request)
