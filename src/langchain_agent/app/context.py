from dataclasses import dataclass

from langchain_agent.harness.middleware.turn_recovery import TurnRecoveryPlan
from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.repository_knowledge import RepositoryKnowledgeService


@dataclass(frozen=True)
class AgentContext:
    repository_path: str
    repository_knowledge: RepositoryKnowledgeService
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    turn_recovery: TurnRecoveryPlan | None = None
