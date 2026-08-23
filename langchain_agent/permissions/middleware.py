from collections.abc import Awaitable, Callable

from langchain.agents.middleware import (
    AgentMiddleware,
    HumanInTheLoopMiddleware,
)
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain_agent.permissions.policy import check_permission
from langchain_agent.permissions.registry import ToolPolicyRegistry
from langchain_agent.permissions.types import PermissionAction

AsyncToolHandler = Callable[
    [ToolCallRequest],
    Awaitable[ToolMessage | Command],
]


class PermissionEnforcementMiddleware(AgentMiddleware):
    """
    Final enforcement boundary for application tools.

    ALLOW / ASK:
        execution may proceed. ASK is expected to have already
        passed through HumanInTheLoopMiddleware.

    DENY:
        short-circuit without executing the tool.
    """

    def __init__(
        self,
        *,
        registry: ToolPolicyRegistry,
        protected_tool_names: set[str],
    ):
        self.registry = registry
        self.protected_tool_names = protected_tool_names

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolMessage | Command:
        tool_call = request.tool_call
        tool_name = tool_call["name"]

        # Framework/internal tools such as write_todos are outside
        # the application permission policy.
        if tool_name not in self.protected_tool_names:
            return await handler(request)

        decision = check_permission(
            tool_call=tool_call,
            policy=self.registry.get(tool_name),
            context=request.runtime.context,
        )

        if decision.action == PermissionAction.DENY:
            return ToolMessage(
                content=(
                    decision.reason or "Tool execution is denied by permission policy."
                ),
                tool_call_id=tool_call["id"],
                status="error",
            )

        return await handler(request)


def build_hitl_middleware(
    *,
    tools,
    registry: ToolPolicyRegistry,
) -> HumanInTheLoopMiddleware:
    protected_tool_names = {tool.name for tool in tools}

    def should_interrupt(
        request: ToolCallRequest,
    ) -> bool:
        tool_name = request.tool_call["name"]

        if tool_name not in protected_tool_names:
            return False

        decision = check_permission(
            tool_call=request.tool_call,
            policy=registry.get(tool_name),
            context=request.runtime.context,
        )

        return decision.action == PermissionAction.ASK

    interrupt_on = {
        tool_name: {
            "allowed_decisions": [
                "approve",
                "reject",
            ],
            "when": should_interrupt,
        }
        for tool_name in protected_tool_names
    }

    return HumanInTheLoopMiddleware(
        interrupt_on=interrupt_on,
        description_prefix=("Tool execution requires permission"),
    )
