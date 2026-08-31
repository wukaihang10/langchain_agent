from dataclasses import dataclass
from typing import Literal, TypeAlias


RetrievalMode: TypeAlias = Literal["fast", "quality"]


@dataclass(frozen=True, slots=True)
class RepositoryKnowledgeConfig:
    """Repository-specific indexing and retrieval policy."""

    retrieval_mode: RetrievalMode = "fast"
    max_chunk_characters: int = 2400
    overlap_lines: int = 8
    max_context_characters: int = 8000
    max_context_items: int = 5
    max_query_rewrites: int = 2

    def __post_init__(self) -> None:
        if self.retrieval_mode not in ("fast", "quality"):
            raise ValueError("retrieval_mode must be 'fast' or 'quality'")
        if self.max_chunk_characters <= 0:
            raise ValueError("max_chunk_characters must be greater than 0")
        if self.overlap_lines < 0:
            raise ValueError("overlap_lines cannot be negative")
        if self.max_context_characters <= 0:
            raise ValueError("max_context_characters must be greater than 0")
        if self.max_context_items <= 0:
            raise ValueError("max_context_items must be greater than 0")
        if self.max_query_rewrites <= 0:
            raise ValueError("max_query_rewrites must be greater than 0")
