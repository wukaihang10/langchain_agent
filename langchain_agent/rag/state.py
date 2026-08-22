from typing import NotRequired, TypedDict

from langchain_agent.ragservice.models import SearchResult


class RAGState(TypedDict):
    query: str
    top_k: int

    index_ready: NotRequired[bool]
    error: NotRequired[str]

    candidate_results: NotRequired[list[SearchResult]]
    final_results: NotRequired[list[SearchResult]]

    context: NotRequired[str]
    context_character_count: NotRequired[int]
