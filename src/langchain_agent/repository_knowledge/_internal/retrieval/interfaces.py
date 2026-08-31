from __future__ import annotations

from typing import Protocol

from langchain_agent.repository_knowledge._internal.retrieval.models import (
    SearchResult,
)
from langchain_agent.repository_knowledge._internal.source.models import Chunk
from langchain_agent.repository_knowledge.ports import FloatMatrix, FloatVector


class Retriever(Protocol):
    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]: ...


class VectorStore(Protocol):
    dimension: int

    def replace(
        self,
        chunks: list[Chunk],
        vectors: FloatMatrix,
    ) -> None: ...

    def search(
        self,
        query_vector: FloatVector,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]: ...


class Reranker(Protocol):
    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]: ...
