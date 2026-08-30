"""Public API for repository indexing and retrieval."""

from langchain_agent.repository_knowledge.config import (
    RepositoryKnowledgeConfig,
    RetrievalMode,
)
from langchain_agent.repository_knowledge.errors import (
    EmbeddingError,
    IndexBuildError,
    IndexNotReadyError,
    InvalidRepositoryError,
    RepositoryChangedDuringIndexingError,
    RepositoryKnowledgeError,
)
from langchain_agent.repository_knowledge.models import (
    Evidence,
    IndexReadyResult,
    RepositoryKnowledgeStatus,
    SearchResponse,
)
from langchain_agent.repository_knowledge.ports import EmbeddingClient
from langchain_agent.repository_knowledge.service import RepositoryKnowledgeService

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "Evidence",
    "IndexBuildError",
    "IndexNotReadyError",
    "IndexReadyResult",
    "InvalidRepositoryError",
    "RepositoryChangedDuringIndexingError",
    "RepositoryKnowledgeConfig",
    "RepositoryKnowledgeError",
    "RepositoryKnowledgeService",
    "RepositoryKnowledgeStatus",
    "RetrievalMode",
    "SearchResponse",
]
