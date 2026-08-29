from dataclasses import dataclass
from pathlib import Path
from typing import Literal


@dataclass(frozen=True, slots=True)
class Evidence:
    """Stable, caller-facing evidence returned by repository search."""

    context_id: str
    source: str
    content: str
    score: float
    rank: int
    symbol: str | None = None
    symbol_type: str | None = None
    start_line: int | None = None
    end_line: int | None = None
    part_index: int | None = None
    part_count: int | None = None


@dataclass(frozen=True, slots=True)
class SearchResponse:
    """Structured retrieval output independent of Agent tool formatting."""

    repository_path: Path
    query: str
    evidence: tuple[Evidence, ...]
    context: str
    retrieved_count: int

    @property
    def context_character_count(self) -> int:
        return len(self.context)


@dataclass(frozen=True, slots=True)
class IndexReadyResult:
    """Result of loading or rebuilding a repository index."""

    repository_path: Path
    index_path: Path
    source: Literal["loaded", "rebuilt"]
    document_count: int
    chunk_count: int
    vector_dimension: int
    rebuild_reason: str | None = None

