from collections.abc import Awaitable, Callable

from langchain.agents.middleware import AgentMiddleware
from langchain_core.messages import ToolMessage
from langgraph.prebuilt.tool_node import ToolCallRequest
from langgraph.types import Command

from langchain_agent.repository_knowledge import RepositoryKnowledgeError
from langchain_agent.tools.errors import RepositoryToolError

AsyncToolHandler = Callable[
    [ToolCallRequest],
    Awaitable[ToolMessage | Command],
]


class ToolErrorMiddleware(AgentMiddleware):
    """Translate expected domain failures into observations the model can act on."""

    async def awrap_tool_call(
        self,
        request: ToolCallRequest,
        handler: AsyncToolHandler,
    ) -> ToolMessage | Command:
        try:
            return await handler(request)
        except (RepositoryKnowledgeError, RepositoryToolError) as error:
            return ToolMessage(
                content=f"Repository tool failed: {error}",
                tool_call_id=request.tool_call["id"],
                status="error",
            )
