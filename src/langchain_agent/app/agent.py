from collections.abc import Sequence

from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware
from langchain.agents import create_agent
from langchain.agents.middleware import SummarizationMiddleware, TodoListMiddleware
from langchain_core.language_models import BaseChatModel
from langchain_core.tools import BaseTool
from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_agent.app.config import AppConfig
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.middleware.git_audit import GitAuditMiddleware
from langchain_agent.harness.middleware.mcp_retry import (
    build_mcp_no_retry_failure_middleware,
    build_mcp_retry_middleware,
)
from langchain_agent.harness.middleware.tool_errors import (
    build_repository_tool_error_middleware,
)
from langchain_agent.harness.permissions.middleware import (
    PermissionEnforcementMiddleware,
    build_hitl_middleware,
)
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry
from langchain_agent.subagents.code_researcher import build_code_researcher
from langchain_agent.subagents.code_reviewer import build_code_reviewer
from langchain_agent.tools.repository import REPOSITORY_TOOLS
from langchain_agent.tools.repository_knowledge import search_repository_knowledge


NATIVE_TOOLS = (*REPOSITORY_TOOLS, search_repository_knowledge)

_SYSTEM_PROMPT = (
    "You are a repository analysis agent. "
    "Use repository tools when source-code information is required. "
    "Do not invent file contents."
)

_SUBAGENT_SYSTEM_PROMPT = """
Use subagents proactively when their specialization matches the task.

Delegation policy:
- For complex, multi-step, read-only repository investigation spanning
  multiple files, symbols, or modules, delegate to code-researcher.
- For review of an existing change set, working tree, git diff, or recently
  modified code, delegate to code-reviewer.
- Use repository tools directly for simple, targeted lookups that normally
  require only one or two tool calls, such as reading a known file or locating
  a known symbol.
- After a subagent returns, use direct tools only when targeted verification
  or follow-up is still needed.
""".strip()


def build_agent(
    *,
    model: BaseChatModel,
    summary_model: BaseChatModel,
    researcher_model: BaseChatModel,
    reviewer_model: BaseChatModel,
    native_tools: Sequence[BaseTool],
    mcp_tools: Sequence[BaseTool],
    policy_registry: ToolPolicyRegistry,
    checkpointer: BaseCheckpointSaver,
    config: AppConfig,
):
    tools = [*native_tools, *mcp_tools]
    protected_tool_names = {tool.name for tool in tools}

    subagent_middleware = SubAgentMiddleware(
        backend=StateBackend(),
        subagents=[
            build_code_researcher(researcher_model),
            build_code_reviewer(reviewer_model),
        ],
        system_prompt=_SUBAGENT_SYSTEM_PROMPT,
    )
    hitl_middleware = build_hitl_middleware(
        tools=tools,
        registry=policy_registry,
    )
    permission_enforcement_middleware = PermissionEnforcementMiddleware(
        registry=policy_registry,
        protected_tool_names=protected_tool_names,
    )
    mcp_retry_middleware = build_mcp_retry_middleware(
        mcp_tools=mcp_tools,
        policy_registry=policy_registry,
    )
    mcp_no_retry_failure_middleware = build_mcp_no_retry_failure_middleware(
        mcp_tools=mcp_tools,
        policy_registry=policy_registry,
    )
    middleware = [
        hitl_middleware,
        TodoListMiddleware(),
        subagent_middleware,
        SummarizationMiddleware(
            model=summary_model,
            trigger=("tokens", config.summarization_trigger_tokens),
            keep=("tokens", config.summarization_keep_tokens),
        ),
        permission_enforcement_middleware,
        *(
            [mcp_retry_middleware]
            if mcp_retry_middleware is not None
            else []
        ),
        *(
            [mcp_no_retry_failure_middleware]
            if mcp_no_retry_failure_middleware is not None
            else []
        ),
        build_repository_tool_error_middleware(),
        GitAuditMiddleware(),
    ]

    return create_agent(
        model=model,
        tools=tools,
        middleware=middleware,
        context_schema=AgentContext,
        system_prompt=_SYSTEM_PROMPT,
        checkpointer=checkpointer,
        name=config.agent_name,
    )
