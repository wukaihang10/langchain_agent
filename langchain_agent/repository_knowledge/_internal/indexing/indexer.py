from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path

import numpy as np

from langchain_agent.repository_knowledge._internal.source.interfaces import (
    DocumentChunker,
    DocumentLoader,
)
from langchain_agent.repository_knowledge._internal.retrieval.interfaces import (
    VectorStore,
)

from langchain_agent.repository_knowledge._internal.indexing.models import (
    IndexBuildStats,
)
from langchain_agent.repository_knowledge._internal.source.models import Document
from langchain_agent.repository_knowledge.ports import EmbeddingClient


class RepositoryIndexer:
    """
    负责构建仓库知识索引。

    流程：
        Document
        -> Chunk
        -> Embedding
        -> VectorStore
    """

    def __init__(
        self,
        loader: DocumentLoader,
        chunker: DocumentChunker,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        if embedding_client.dimension != vector_store.dimension:
            raise ValueError(
                "Embedding client and vector store dimension do not match: "
                f"{embedding_client.dimension} != {vector_store.dimension}"
            )

        self.loader = loader
        self.chunker = chunker
        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def rebuild_directory(
        self,
        directory: str | Path,
    ) -> IndexBuildStats:
        """
        读取目录中的文档并重建整个索引。

        重建成功后，向量库只包含当前目录生成的 Chunk
        """

        root = Path(directory).resolve()

        documents = self.loader.load_directory(root)

        return self.rebuild_documents(
            documents=documents,
            source=root.as_posix(),
        )

    def rebuild_documents(
        self,
        documents: Sequence[Document],
        source: str = "documents",
    ) -> IndexBuildStats:
        """
        根据已经加载好的 Document 重建索引。
        """

        document_list = list(documents)

        chunks = self.chunker.split_documents(document_list)

        if chunks:
            vectors = self.embedding_client.embed_documents(
                [chunk.content_for_embedding for chunk in chunks]
            )

        else:
            vectors = np.empty(
                shape=(0, self.embedding_client.dimension),
                dtype=np.float32,
            )

        self.vector_store.replace(
            chunks=chunks,
            vectors=vectors,
        )

        return IndexBuildStats(
            source=source,
            document_count=len(document_list),
            chunk_count=len(chunks),
            vector_dimension=self.embedding_client.dimension,
        )
