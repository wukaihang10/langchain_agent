from __future__ import annotations

from collections.abc import AsyncIterator
from contextlib import asynccontextmanager
from dataclasses import dataclass
from typing import Any

from dotenv import load_dotenv
from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_agent.app.agent import NATIVE_TOOLS, build_agent
from langchain_agent.app.config import AppConfig
from langchain_agent.app.repository_knowledge import RepositoryKnowledgeProvider
from langchain_agent.app.session_continuation import SessionContinuation
from langchain_agent.harness.permissions.registry import build_tool_policy_registry
from langchain_agent.integrations.mcp.client import (
    MCPIntegration,
    load_mcp_integration,
)
from langchain_agent.integrations.mcp.config import load_mcp_config
from langchain_agent.integrations.model import create_model
from langchain_agent.persistence.checkpoints import open_checkpointer
from langchain_agent.persistence.sessions import SessionStore
from langchain_agent.repository_knowledge import (
    FallbackQueryExpander,
    IdentityQueryExpander,
    LLMQueryExpander,
    RepositoryKnowledgeConfig,
)
from langchain_agent.repository_knowledge.embedding import (
    SentenceTransformerEmbeddingClient,
)


@dataclass(frozen=True)
class Application:
    agent: Any
    checkpointer: BaseCheckpointSaver
    session_store: SessionStore
    repository_knowledge: RepositoryKnowledgeProvider
    mcp: MCPIntegration
    config: AppConfig
    continuation: SessionContinuation


@asynccontextmanager
async def bootstrap_application(
    config: AppConfig,
) -> AsyncIterator[Application]:
    config.paths.ensure_directories()
    load_dotenv(dotenv_path=config.paths.environment_path, override=False)
    session_store = SessionStore(config.paths.session_path)

    query_expander = FallbackQueryExpander(
        primary=LLMQueryExpander(model=create_model(thinking=False)),
        fallback=IdentityQueryExpander(),
    )
    repository_knowledge = RepositoryKnowledgeProvider(
        index_root=config.paths.index_root,
        embedding_client_factory=lambda: SentenceTransformerEmbeddingClient(
            model_name=config.embedding_model_name,
        ),
        query_expander=query_expander,
        config=RepositoryKnowledgeConfig(
            retrieval_mode=config.retrieval_mode,
            max_query_rewrites=config.max_query_rewrites,
        ),
    )

    mcp_config = load_mcp_config(config.paths.mcp_config_path)
    mcp = await load_mcp_integration(mcp_config)
    policy_registry = build_tool_policy_registry(
        local_tools=NATIVE_TOOLS,
        external_policy_overrides=mcp_config.tool_policies,
    )

    async with open_checkpointer(config.paths.checkpoint_path) as checkpointer:
        agent = build_agent(
            model=create_model(),
            summary_model=create_model(),
            researcher_model=create_model(),
            reviewer_model=create_model(),
            native_tools=NATIVE_TOOLS,
            mcp_tools=mcp.tools,
            policy_registry=policy_registry,
            checkpointer=checkpointer,
            config=config,
        )
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=policy_registry,
            session_store=session_store,
        )

        yield Application(
            agent=agent,
            checkpointer=checkpointer,
            session_store=session_store,
            repository_knowledge=repository_knowledge,
            mcp=mcp,
            config=config,
            continuation=continuation,
        )
