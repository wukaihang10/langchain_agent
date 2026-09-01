from langchain.agents.middleware import ToolCallRequest, ToolErrorMiddleware

from langchain_agent.tools.repository_errors import RepositoryToolError


def format_repository_tool_error(
    error: Exception,
    request: ToolCallRequest,
) -> str | None:
    """Return model-safe content for the repository tool error interface."""

    if not isinstance(error, RepositoryToolError):
        return None

    tool_name = request.tool_call["name"]
    return f"Repository tool `{tool_name}` failed: {error}"


def build_repository_tool_error_middleware() -> ToolErrorMiddleware:
    """Configure LangChain's tool-error mechanism with project policy."""

    return ToolErrorMiddleware(on_error=format_repository_tool_error)
