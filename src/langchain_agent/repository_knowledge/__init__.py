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
    QueryExpansionError,
    RepositoryChangedDuringIndexingError,
    RepositoryKnowledgeError,
)
from langchain_agent.repository_knowledge.models import (
    Evidence,
    IndexReadyResult,
    RepositoryKnowledgeStatus,
    SearchResponse,
)
from langchain_agent.repository_knowledge.ports import EmbeddingClient, QueryExpander
from langchain_agent.repository_knowledge.query_expansion import (
    FallbackQueryExpander,
    IdentityQueryExpander,
    LLMQueryExpander,
)
from langchain_agent.repository_knowledge.service import RepositoryKnowledgeService

__all__ = [
    "EmbeddingClient",
    "EmbeddingError",
    "Evidence",
    "FallbackQueryExpander",
    "IndexBuildError",
    "IndexNotReadyError",
    "IndexReadyResult",
    "InvalidRepositoryError",
    "IdentityQueryExpander",
    "LLMQueryExpander",
    "QueryExpander",
    "QueryExpansionError",
    "RepositoryChangedDuringIndexingError",
    "RepositoryKnowledgeConfig",
    "RepositoryKnowledgeError",
    "RepositoryKnowledgeService",
    "RepositoryKnowledgeStatus",
    "RetrievalMode",
    "SearchResponse",
]
