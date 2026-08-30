from __future__ import annotations

import math
from typing import TYPE_CHECKING

import numpy as np

if TYPE_CHECKING:
    from sentence_transformers import CrossEncoder

from langchain_agent.repository_knowledge._internal.interfaces import (
    Reranker,
    Retriever,
)
from langchain_agent.repository_knowledge._internal.models import (
    Chunk,
    SearchResult,
)


class CrossEncoderReranker:
    """
    使用 Cross-Encoder 对候选 Chunk
    与原始用户 Query 做逐对相关性评分。

    Candidate Retrieval 已经负责“找全”。

    本类只负责：
        query + candidate
            ↓
        relevance score
            ↓
        reranking
    """

    def __init__(
        self,
        model_name: str = ("BAAI/bge-reranker-base"),
        *,
        batch_size: int = 8,
        max_length: int = 512,
        device: str | None = None,
        show_progress_bar: bool = False,
    ) -> None:
        if batch_size <= 0:
            raise ValueError("batch_size must be " "greater than 0")

        if max_length <= 0:
            raise ValueError("max_length must be " "greater than 0")

        self.model_name = model_name
        self.batch_size = batch_size
        self.max_length = max_length
        self.device = device
        self.show_progress_bar = show_progress_bar

        # Lazy loading:
        # Repository knowledge backend 初始化时
        # 不立即占用 reranker 内存。
        self._model: CrossEncoder | None = None

    def rerank(
        self,
        query: str,
        results: list[SearchResult],
    ) -> list[SearchResult]:
        if not isinstance(query, str):
            raise TypeError("query must be a string")

        normalized_query = query.strip()

        if not normalized_query:
            return []

        if not results:
            return []

        pairs = [
            (
                normalized_query,
                self._build_passage(result.chunk),
            )
            for result in results
        ]

        model = self._get_model()

        raw_scores = model.predict(
            pairs,
            batch_size=self.batch_size,
            show_progress_bar=(self.show_progress_bar),
            convert_to_numpy=True,
        )

        scores = np.asarray(
            raw_scores,
            dtype=np.float32,
        ).reshape(-1)

        if len(scores) != len(results):
            raise RuntimeError("Reranker returned an " "unexpected number of scores")

        if not np.isfinite(scores).all():
            raise RuntimeError("Reranker returned " "non-finite scores")

        scored_results = [
            (
                result,
                float(score),
            )
            for result, score in zip(
                results,
                scores,
                strict=True,
            )
        ]

        scored_results.sort(
            key=lambda item: (
                -item[1],
                # 分数相同时保留原始
                # retrieval rank 作为稳定 tie-breaker。
                item[0].rank,
                item[0].chunk.id,
            )
        )

        return [
            SearchResult(
                chunk=result.chunk,
                score=score,
                rank=index + 1,
            )
            for index, (
                result,
                score,
            ) in enumerate(scored_results)
        ]

    def _get_model(
        self,
    ) -> CrossEncoder:
        if self._model is None:
            from sentence_transformers import CrossEncoder

            self._model = CrossEncoder(
                model_name_or_path=(self.model_name),
                max_length=self.max_length,
                device=self.device,
            )

        return self._model

    @staticmethod
    def _build_passage(
        chunk: Chunk,
    ) -> str:
        """
        Reranker 使用与 Dense Retrieval
        相同的 enriched Chunk representation。

        embedding_content 存在时其中包含：
        file / symbol / symbol type / code 等信息。

        若不存在，则自动退化到原始 content。
        """

        return chunk.content_for_embedding


class RerankingRetriever:
    """
    Retriever 装饰器。

    先从基础 Retriever 获取较大的候选池，
    再用 Reranker 重排，
    最后返回调用方要求的 Top-k。
    """

    def __init__(
        self,
        *,
        base_retriever: Retriever,
        reranker: Reranker,
        candidate_count: int = 30,
    ) -> None:
        if candidate_count <= 0:
            raise ValueError("candidate_count must be " "greater than 0")

        self.base_retriever = base_retriever

        self.reranker = reranker

        self.candidate_count = candidate_count

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if minimum_score is not None and not math.isfinite(minimum_score):
            raise ValueError("minimum_score must be finite")

        candidate_k = max(
            top_k,
            self.candidate_count,
        )

        candidates = self.base_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            # 原 Retriever 的 score
            # 和 CrossEncoder score
            # 语义不同，所以不要提前过滤。
            minimum_score=None,
        )

        reranked = self.reranker.rerank(
            query=query,
            results=candidates,
        )

        selected: list[SearchResult] = []

        for result in reranked:
            if minimum_score is not None and result.score < minimum_score:
                continue

            selected.append(result)

            if len(selected) >= top_k:
                break

        return selected
