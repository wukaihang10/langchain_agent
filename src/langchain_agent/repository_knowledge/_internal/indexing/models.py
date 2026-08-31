from dataclasses import dataclass
from typing import Literal

from langchain_agent.repository_knowledge._internal.indexing.repository_snapshot import (
    RepositorySnapshot,
)


@dataclass
class IndexBuildStats:
    """Statistics produced by one index build or load."""

    source: str
    document_count: int
    chunk_count: int
    vector_dimension: int

    @property
    def is_empty(self) -> bool:
        return self.chunk_count == 0


@dataclass(frozen=True)
class PreparedIndex:
    """Internal result of loading or rebuilding an index."""

    index: IndexBuildStats
    repository_snapshot: RepositorySnapshot
    source: Literal["disk", "rebuilt"]
    rebuild_reason: str | None = None
