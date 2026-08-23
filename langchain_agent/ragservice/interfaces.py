from __future__ import annotations

from pathlib import Path
from typing import Protocol

import numpy as np
from numpy.typing import NDArray

from langchain_agent.ragservice.models import Chunk, Document, SearchResult

FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]


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


class EmbeddingClient(Protocol):
    dimension: int

    def embed_documents(
        self,
        texts: list[str],
    ) -> FloatMatrix: ...

    def embed_query(
        self,
        query: str,
    ) -> FloatVector: ...


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
