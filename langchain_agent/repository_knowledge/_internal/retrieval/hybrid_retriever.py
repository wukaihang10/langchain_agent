from __future__ import annotations

import math

from langchain_agent.repository_knowledge._internal.retrieval.interfaces import Retriever
from langchain_agent.repository_knowledge._internal.retrieval.models import SearchResult
from langchain_agent.repository_knowledge._internal.source.models import Chunk


class HybridRetriever:
    """
    使用 Reciprocal Rank Fusion 融合多个 Retriever。

    当前融合：

    Dense Vector Retriever
    BM25 Retriever

    不直接比较向量分数和 BM25 分数，
    而是比较各自在结果列表中的排名。
    """

    def __init__(
        self,
        dense_retriever: Retriever,
        lexical_retriever: Retriever,
        rrf_k: int = 60,
        candidate_multiplier: int = 3,
        dense_weight: float = 1.0,
        lexical_weight: float = 1.0,
    ) -> None:
        if rrf_k <= 0:
            raise ValueError("rrf_k must be greater than 0")

        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier " "must be greater than 0")

        if dense_weight < 0:
            raise ValueError("dense_weight cannot be negative")

        if lexical_weight < 0:
            raise ValueError("lexical_weight cannot be negative")

        if dense_weight == 0 and lexical_weight == 0:
            raise ValueError("At least one retrieval weight " "must be greater than 0")

        self.dense_retriever = dense_retriever
        self.lexical_retriever = lexical_retriever

        self.rrf_k = rrf_k
        self.candidate_multiplier = candidate_multiplier
        self.dense_weight = float(dense_weight)
        self.lexical_weight = float(lexical_weight)

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

        if not query.strip():
            return []

        candidate_k = top_k * self.candidate_multiplier

        dense_results = self.dense_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            minimum_score=None,
        )

        lexical_results = self.lexical_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            minimum_score=None,
        )

        fused_scores: dict[str, float] = {}
        chunks_by_id: dict[str, Chunk] = {}
        best_channel_rank: dict[str, int] = {}

        self._accumulate_results(
            results=dense_results,
            weight=self.dense_weight,
            fused_scores=fused_scores,
            chunks_by_id=chunks_by_id,
            best_channel_rank=(best_channel_rank),
        )

        self._accumulate_results(
            results=lexical_results,
            weight=self.lexical_weight,
            fused_scores=fused_scores,
            chunks_by_id=chunks_by_id,
            best_channel_rank=(best_channel_rank),
        )

        ranked_chunk_ids = sorted(
            fused_scores,
            key=lambda chunk_id: (
                -fused_scores[chunk_id],
                best_channel_rank[chunk_id],
                chunks_by_id[chunk_id].source,
                chunk_id,
            ),
        )

        results: list[SearchResult] = []

        for chunk_id in ranked_chunk_ids:
            score = fused_scores[chunk_id]

            if minimum_score is not None and score < minimum_score:
                continue

            results.append(
                SearchResult(
                    chunk=chunks_by_id[chunk_id],
                    score=score,
                    rank=len(results) + 1,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def _accumulate_results(
        self,
        *,
        results: list[SearchResult],
        weight: float,
        fused_scores: dict[str, float],
        chunks_by_id: dict[str, Chunk],
        best_channel_rank: dict[str, int],
    ) -> None:
        if weight == 0:
            return

        for result in results:
            chunk_id = result.chunk.id

            reciprocal_rank_score = weight / (self.rrf_k + result.rank)

            fused_scores[chunk_id] = (
                fused_scores.get(
                    chunk_id,
                    0.0,
                )
                + reciprocal_rank_score
            )

            chunks_by_id.setdefault(
                chunk_id,
                result.chunk,
            )

            previous_best_rank = best_channel_rank.get(chunk_id)

            if previous_best_rank is None or result.rank < previous_best_rank:
                best_channel_rank[chunk_id] = result.rank
