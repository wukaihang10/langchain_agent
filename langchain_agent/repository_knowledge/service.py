from __future__ import annotations

from pathlib import Path
from threading import RLock

from langchain_agent.repository_knowledge.config import RepositoryKnowledgeConfig
from langchain_agent.repository_knowledge.errors import (
    IndexBuildError,
    IndexNotReadyError,
    InvalidRepositoryError,
    RepositoryKnowledgeError,
)
from langchain_agent.repository_knowledge.models import (
    Evidence,
    IndexReadyResult,
    RepositoryKnowledgeStatus,
    SearchResponse,
)
from langchain_agent.repository_knowledge.ports import EmbeddingClient


class RepositoryKnowledgeService:
    """Repository-scoped public facade for indexing and retrieval.

    The current implementation delegates to private Python-repository
    indexing and retrieval components. Callers depend only on this facade so
    those internals can be reorganized
    without changing Agent, CLI, or Web API code.
    """

    def __init__(
        self,
        *,
        repository_path: str | Path,
        index_path: str | Path,
        embedding_client: EmbeddingClient,
        config: RepositoryKnowledgeConfig | None = None,
    ) -> None:
        self.repository_path = self._resolve_repository(repository_path)
        self.index_path = Path(index_path).expanduser().resolve()
        self.embedding_client = embedding_client
        self.config = config or RepositoryKnowledgeConfig()
        self._backend = None
        self._prepared_snapshot = None
        self._last_ready_result: IndexReadyResult | None = None
        self._last_error: Exception | None = None
        self._status = RepositoryKnowledgeStatus.UNPREPARED
        self._lock = RLock()

    @property
    def status(self) -> RepositoryKnowledgeStatus:
        return self._status

    @property
    def is_ready(self) -> bool:
        return self._status is RepositoryKnowledgeStatus.READY

    @property
    def last_error(self) -> Exception | None:
        return self._last_error

    def prepare(self) -> IndexReadyResult:
        with self._lock:
            try:
                existing_result = self._ready_result_if_fresh()

                if existing_result is not None:
                    return existing_result

                self._status = RepositoryKnowledgeStatus.PREPARING
                candidate = self._create_backend()
                prepared_index = candidate.ensure_index(self.index_path)
                result = self._to_ready_result(prepared_index)

            except RepositoryKnowledgeError as error:
                self._mark_failed(error)
                raise
            except OSError as error:
                domain_error = IndexBuildError(
                    "Failed to prepare repository index: " f"{error}"
                )
                self._mark_failed(domain_error)
                raise domain_error from error
            except Exception as error:
                self._mark_failed(error)
                raise

            self._backend = candidate
            self._prepared_snapshot = prepared_index.repository_snapshot
            self._last_ready_result = result
            self._last_error = None
            self._status = RepositoryKnowledgeStatus.READY

            return result

    def search(
        self,
        query: str,
        *,
        top_k: int = 5,
    ) -> SearchResponse:
        with self._lock:
            normalized_query = self._validate_query(query)

            if not isinstance(top_k, int):
                raise TypeError("top_k must be an integer")
            if top_k <= 0:
                raise ValueError("top_k must be greater than 0")
            if not self.is_ready or self._backend is None:
                raise IndexNotReadyError(
                    "Repository knowledge is not ready. "
                    "Call prepare() before search()."
                )

            response = self._backend.search(
                query=normalized_query,
                top_k=top_k,
            )

            evidence = tuple(self._to_evidence(item) for item in response.sources)

            return SearchResponse(
                repository_path=self.repository_path,
                query=normalized_query,
                evidence=evidence,
                context=response.context.text,
                retrieved_count=len(response.search_results),
            )

    def _ready_result_if_fresh(self) -> IndexReadyResult | None:
        if self._status is not RepositoryKnowledgeStatus.READY:
            return None

        if (
            self._backend is None
            or self._prepared_snapshot is None
            or self._last_ready_result is None
        ):
            raise RuntimeError(
                "Ready repository knowledge is missing prepared state"
            )

        current_snapshot = self._backend.build_repository_snapshot()

        if current_snapshot != self._prepared_snapshot:
            return None

        return self._last_ready_result

    def _create_backend(self):
        # Keep implementation details behind the public facade. Import and
        # construct them lazily so selecting a session does not load local
        # models or an index.
        from langchain_agent.repository_knowledge._internal.backend import (
            PythonRepositoryKnowledgeBackend,
        )

        return PythonRepositoryKnowledgeBackend(
            repository_path=self.repository_path,
            embedding_client=self.embedding_client,
            max_chunk_characters=self.config.max_chunk_characters,
            overlap_lines=self.config.overlap_lines,
            max_context_characters=self.config.max_context_characters,
            max_context_items=self.config.max_context_items,
            retrieval_mode=self.config.retrieval_mode,
        )

    def _to_ready_result(self, prepared_index) -> IndexReadyResult:
        index = prepared_index.index

        if index.is_empty:
            raise IndexBuildError(
                "No indexable Python code was found in repository: "
                f"{self.repository_path}"
            )

        source = "loaded" if prepared_index.source == "disk" else "rebuilt"

        return IndexReadyResult(
            repository_path=self.repository_path,
            index_path=self.index_path,
            source=source,
            document_count=index.document_count,
            chunk_count=index.chunk_count,
            vector_dimension=index.vector_dimension,
            rebuild_reason=prepared_index.rebuild_reason,
        )

    def _mark_failed(self, error: Exception) -> None:
        self._backend = None
        self._prepared_snapshot = None
        self._last_ready_result = None
        self._last_error = error
        self._status = RepositoryKnowledgeStatus.FAILED

    @staticmethod
    def _resolve_repository(repository_path: str | Path) -> Path:
        if not isinstance(repository_path, (str, Path)):
            raise TypeError("repository_path must be a string or Path")

        path = Path(repository_path).expanduser().resolve()

        if not path.exists():
            raise InvalidRepositoryError(f"Repository does not exist: {path}")
        if not path.is_dir():
            raise InvalidRepositoryError(
                f"Repository path is not a directory: {path}"
            )

        return path

    @staticmethod
    def _validate_query(query: str) -> str:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            raise ValueError("query cannot be empty")

        return normalized_query

    @staticmethod
    def _to_evidence(item) -> Evidence:
        chunk = item.chunk
        metadata = chunk.metadata

        return Evidence(
            context_id=item.context_id,
            source=chunk.source,
            content=chunk.content,
            score=item.score,
            rank=item.retrieval_rank,
            symbol=RepositoryKnowledgeService._optional_string(
                metadata.get("symbol")
            ),
            symbol_type=RepositoryKnowledgeService._optional_string(
                metadata.get("symbol_type")
            ),
            start_line=RepositoryKnowledgeService._optional_int(
                metadata.get("start_line")
            ),
            end_line=RepositoryKnowledgeService._optional_int(
                metadata.get("end_line")
            ),
            part_index=RepositoryKnowledgeService._optional_int(
                metadata.get("part_index")
            ),
            part_count=RepositoryKnowledgeService._optional_int(
                metadata.get("part_count")
            ),
        )

    @staticmethod
    def _optional_string(value: object) -> str | None:
        return value if isinstance(value, str) and value else None

    @staticmethod
    def _optional_int(value: object) -> int | None:
        return value if isinstance(value, int) else None
