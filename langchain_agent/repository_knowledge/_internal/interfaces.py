from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_agent.repository_knowledge._internal.models import Chunk, Document, SearchResult
from langchain_agent.repository_knowledge.ports import FloatMatrix, FloatVector


class DocumentLoader(Protocol):
    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]: ...


class DocumentChunker(Protocol):
    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]: ...


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
