from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

from langgraph.checkpoint.base import BaseCheckpointSaver

from langchain_agent.app.config import AppConfig
from langchain_agent.app.context import AgentContext
from langchain_agent.app.repository_knowledge import RepositoryKnowledgeProvider
from langchain_agent.persistence.sessions import Session, SessionStore


@dataclass(frozen=True)
class SessionRuntime:
    session: Session
    context: AgentContext
    invoke_config: dict


def build_session_runtime(
    *,
    session: Session,
    repository_path: Path,
    session_store: SessionStore,
    repository_knowledge_provider: RepositoryKnowledgeProvider,
    config: AppConfig,
) -> SessionRuntime:
    resolved_path = repository_path.expanduser().resolve()

    if not resolved_path.is_dir():
        raise NotADirectoryError(f"Repository path does not exist: {resolved_path}")

    repository_knowledge = repository_knowledge_provider.get(resolved_path)
    context = AgentContext(
        repository_path=str(resolved_path),
        repository_knowledge=repository_knowledge,
        permission_mode=config.permission_mode,
    )
    invoke_config = {
        "configurable": {
            "thread_id": session.thread_id,
        },
        "run_name": config.agent_version,
        "tags": list(config.run_tags),
        "metadata": {
            "agent_version": config.agent_version,
            "thread_id": session.thread_id,
            "repository": resolved_path.name,
            "permission_mode": config.permission_mode.value,
        },
    }

    return SessionRuntime(
        session=session,
        context=context,
        invoke_config=invoke_config,
    )


async def delete_session(
    *,
    session: Session,
    session_store: SessionStore,
    checkpointer: BaseCheckpointSaver,
) -> bool:
    if session_store.get(session.thread_id) is None:
        return False

    await checkpointer.adelete_thread(session.thread_id)

    return session_store.delete(session.thread_id)
