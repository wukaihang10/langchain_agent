from __future__ import annotations

from collections.abc import Callable
from pathlib import Path

from langchain_agent.repository_knowledge import (
    EmbeddingClient,
    QueryExpander,
    RepositoryKnowledgeConfig,
    RepositoryKnowledgeService,
)
from langchain_agent.repository_knowledge.cache import repository_cache_key


class RepositoryKnowledgeProvider:
    """Create and cache one repository-knowledge service per repository."""

    def __init__(
        self,
        *,
        index_root: Path,
        embedding_client_factory: Callable[[], EmbeddingClient],
        query_expander: QueryExpander,
        config: RepositoryKnowledgeConfig,
    ) -> None:
        self._index_root = index_root
        self._embedding_client_factory = embedding_client_factory
        self._query_expander = query_expander
        self._config = config
        self._embedding_client: EmbeddingClient | None = None
        self._services: dict[Path, RepositoryKnowledgeService] = {}

    def get(self, repository_path: Path) -> RepositoryKnowledgeService:
        resolved_path = repository_path.resolve()
        existing_service = self._services.get(resolved_path)

        if existing_service is not None:
            return existing_service

        if self._embedding_client is None:
            self._embedding_client = self._embedding_client_factory()

        service = RepositoryKnowledgeService(
            repository_path=resolved_path,
            index_path=(self._index_root / repository_cache_key(resolved_path)),
            embedding_client=self._embedding_client,
            query_expander=self._query_expander,
            config=self._config,
        )
        self._services[resolved_path] = service

        return service
