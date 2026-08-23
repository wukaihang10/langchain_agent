from __future__ import annotations

from langchain_agent.ragservice.interfaces import (
    EmbeddingClient,
    VectorStore,
)


from langchain_agent.ragservice.models import SearchResult


class VectorRetriever:
    """
    讲文本查询转换成向量，
    再交给向量库执行 Top-k 检索。
    """

    def __init__(
        self,
        embedding_client: EmbeddingClient,
        vector_store: VectorStore,
    ) -> None:
        if embedding_client.dimension != vector_store.dimension:
            raise ValueError(
                f"Embedding client and vector store dimension do not match: {embedding_client.dimension} != {vector_store.dimension}"
            )

        self.embedding_client = embedding_client
        self.vector_store = vector_store

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        """
        根据自然语言查询返回最相关的 Chunk。

        流程：
            查询文本
            -> 查询向量
            -> 向量相似度搜索
            -> SearchResult 列表
        """

        query_vector = self.embedding_client.embed_query(query)

        return self.vector_store.search(
            query_vector=query_vector,
            top_k=top_k,
            minimum_score=minimum_score,
        )
