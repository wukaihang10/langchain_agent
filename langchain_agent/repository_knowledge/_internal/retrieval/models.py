from dataclasses import dataclass

from langchain_agent.repository_knowledge._internal.source.models import Chunk


@dataclass
class SearchResult:
    """One ranked chunk returned by an internal retriever."""

    chunk: Chunk
    score: float
    rank: int


@dataclass
class SelectedContext:
    """One ranked chunk selected for the final retrieval context."""

    context_id: str
    chunk: Chunk
    score: float
    retrieval_rank: int


@dataclass
class RetrievedContext:
    """Budgeted text and structured items selected after retrieval."""

    text: str
    items: list[SelectedContext]
    character_count: int

    @property
    def is_empty(self) -> bool:
        return not self.items


@dataclass
class RetrievalResult:
    """Internal retrieval output before conversion to the public response."""

    query: str
    context: RetrievedContext
    search_results: list[SearchResult]

    @property
    def sources(self) -> list[SelectedContext]:
        return self.context.items
