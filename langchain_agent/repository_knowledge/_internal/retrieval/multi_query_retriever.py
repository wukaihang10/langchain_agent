from __future__ import annotations

import math

from langchain_agent.repository_knowledge._internal.retrieval.interfaces import Retriever
from langchain_agent.repository_knowledge._internal.retrieval.models import (
    SearchResult,
)
from langchain_agent.repository_knowledge._internal.source.models import Chunk
from langchain_agent.repository_knowledge.ports import QueryExpander


class MultiQueryRetriever:
    """
    对原始 query 和多个 rewrite 分别执行检索，
    再通过 RRF 融合多个 query 的结果。

    它不负责：
    - Dense/BM25 融合
    - Query generation
    - Result diversity

    这些分别由其他组件负责。
    """

    def __init__(
        self,
        *,
        base_retriever: Retriever,
        query_expander: QueryExpander,
        max_query_rewrites: int = 2,
        rrf_k: int = 60,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")
        if max_query_rewrites <= 0:
            raise ValueError("max_query_rewrites must be greater than 0")

        self.base_retriever = base_retriever

        self.query_expander = query_expander
        self.max_query_rewrites = max_query_rewrites

        self.rrf_k = rrf_k

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if minimum_score is not None and not math.isfinite(minimum_score):
            raise ValueError("minimum_score must be finite")

        original_query = query.strip()

        if not original_query:
            return []

        queries = self._build_queries(original_query)

        fused_scores: dict[
            str,
            float,
        ] = {}

        chunks_by_id: dict[
            str,
            Chunk,
        ] = {}

        best_rank: dict[
            str,
            int,
        ] = {}

        for retrieval_query in queries:
            results = self.base_retriever.retrieve(
                query=retrieval_query,
                top_k=top_k,
                minimum_score=None,
            )

            for result in results:
                chunk_id = result.chunk.id

                score = 1.0 / (self.rrf_k + result.rank)

                fused_scores[chunk_id] = (
                    fused_scores.get(
                        chunk_id,
                        0.0,
                    )
                    + score
                )

                chunks_by_id.setdefault(
                    chunk_id,
                    result.chunk,
                )

                previous_rank = best_rank.get(chunk_id)

                if previous_rank is None or result.rank < previous_rank:
                    best_rank[chunk_id] = result.rank

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                best_rank[chunk_id],
                chunks_by_id[chunk_id].source,
                chunk_id,
            ),
        )

        output: list[SearchResult] = []

        for chunk_id in ranked_chunk_ids:
            score = fused_scores[chunk_id]

            if minimum_score is not None and score < minimum_score:
                continue

            output.append(
                SearchResult(
                    chunk=chunks_by_id[chunk_id],
                    score=score,
                    rank=len(output) + 1,
                )
            )

            if len(output) >= top_k:
                break

        return output

    def _build_queries(
        self,
        original_query: str,
    ) -> list[str]:
        """Combine the required original query with optional rewrites."""

        rewrites = self.query_expander.expand(
            original_query,
            limit=self.max_query_rewrites,
        )

        if isinstance(rewrites, str):
            raise TypeError("query expander must return a sequence of strings")

        rewrites = tuple(rewrites)[: self.max_query_rewrites]

        queries = [
            original_query,
            *rewrites,
        ]

        return self._deduplicate_queries(queries)

    @staticmethod
    def _deduplicate_queries(
        queries: list[str],
    ) -> list[str]:
        result: list[str] = []
        seen: set[str] = set()

        for query in queries:
            normalized = query.strip()

            if not normalized:
                continue

            key = normalized.casefold()

            if key in seen:
                continue

            seen.add(key)
            result.append(normalized)

        return result
