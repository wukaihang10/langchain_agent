from dataclasses import dataclass

from langchain_agent.ragservice.repository_manager import RepositoryKnowledgeManager

from langchain_agent.permissions.types import PermissionMode


@dataclass(frozen=True)
class AgentContext:
    repository_path: str
    rag_manager: RepositoryKnowledgeManager
    permission_mode: PermissionMode = PermissionMode.DEFAULT
