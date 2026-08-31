from collections.abc import Sequence

from anyio import BrokenResourceError, ClosedResourceError, EndOfStream
from httpx import NetworkError, TimeoutException
from langchain.agents.middleware import ToolRetryMiddleware
from langchain_core.tools import BaseTool

from langchain_agent.harness.permissions.registry import ToolPolicyRegistry

_TRANSIENT_TRANSPORT_ERRORS = (
    TimeoutError,
    ConnectionError,
    TimeoutException,
    NetworkError,
    BrokenResourceError,
    ClosedResourceError,
    EndOfStream,
)


class MCPRetryMiddleware(ToolRetryMiddleware):
    """Retry explicitly safe MCP tools."""


class MCPNoRetryFailureMiddleware(ToolRetryMiddleware):
    """Expose uncertain MCP transport failures without retrying."""


def is_transient_mcp_error(error: Exception) -> bool:
    """Return whether an MCP transport failure is safe to retry."""

    if isinstance(error, BaseExceptionGroup):
        nested_errors = error.exceptions

        return bool(nested_errors) and all(
            isinstance(nested_error, Exception)
            and is_transient_mcp_error(nested_error)
            for nested_error in nested_errors
        )

    return isinstance(error, _TRANSIENT_TRANSPORT_ERRORS)


def format_mcp_retry_failure(error: Exception) -> str:
    """Format exhausted MCP transport retries as a model-visible tool failure."""

    return (
        "MCP transport remained unavailable after automatic retries "
        f"({type(error).__name__}: {error}). "
        "Do not immediately repeat the same call; use another approach "
        "or tell the user that the external tool is unavailable."
    )


def format_mcp_no_retry_failure(error: Exception) -> str:
    """Format a transport failure when the call is not safe to retry."""

    return (
        "MCP transport failed and this call was not automatically retried "
        "because the tool is not explicitly classified as safe to retry "
        f"({type(error).__name__}: {error}). "
        "The external operation's outcome may be unknown. Do not immediately "
        "repeat the call; report the uncertainty and verify external state first."
    )


def build_mcp_retry_middleware(
    *,
    mcp_tools: Sequence[BaseTool],
    policy_registry: ToolPolicyRegistry,
    max_retries: int = 2,
    initial_delay: float = 0.5,
    backoff_factor: float = 2.0,
    max_delay: float = 4.0,
    jitter: bool = True,
) -> MCPRetryMiddleware | None:
    """Build retries only for explicitly safe, idempotent MCP tools."""

    retryable_tools = []

    for mcp_tool in mcp_tools:
        policy = policy_registry.get(mcp_tool.name)

        if policy is None:
            continue

        if policy.idempotent and not policy.side_effect:
            retryable_tools.append(mcp_tool)

    if not retryable_tools:
        return None

    return MCPRetryMiddleware(
        tools=retryable_tools,
        max_retries=max_retries,
        retry_on=is_transient_mcp_error,
        on_failure=format_mcp_retry_failure,
        initial_delay=initial_delay,
        backoff_factor=backoff_factor,
        max_delay=max_delay,
        jitter=jitter,
    )


def build_mcp_no_retry_failure_middleware(
    *,
    mcp_tools: Sequence[BaseTool],
    policy_registry: ToolPolicyRegistry,
) -> MCPNoRetryFailureMiddleware | None:
    """Expose unsafe or unclassified MCP transport failures without retrying."""

    no_retry_tools = []

    for mcp_tool in mcp_tools:
        policy = policy_registry.get(mcp_tool.name)

        if policy is None or not policy.idempotent or policy.side_effect:
            no_retry_tools.append(mcp_tool)

    if not no_retry_tools:
        return None

    return MCPNoRetryFailureMiddleware(
        tools=no_retry_tools,
        max_retries=0,
        retry_on=is_transient_mcp_error,
        on_failure=format_mcp_no_retry_failure,
    )
