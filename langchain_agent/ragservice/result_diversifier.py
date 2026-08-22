from __future__ import annotations

from collections.abc import Sequence

from langgraph_agent.ragservice.models import SearchResult

from langgraph_agent.ragservice.interfaces import Retriever


class ResultDiversifier:
    """
    对已经按相关性排序的检索结果进行多样性选择。

    当前规则：

    同一个 source + symbol
    最多保留 max_results_per_symbol 条。

    它不重新计算相关性分数，
    只对已有排序结果进行选择。
    """

    def __init__(
        self,
        max_results_per_symbol: int = 2,
    ) -> None:
        if max_results_per_symbol <= 0:
            raise ValueError("max_results_per_symbol " "must be greater than 0")

        self.max_results_per_symbol = max_results_per_symbol

    def diversify(
        self,
        results: Sequence[SearchResult],
        top_k: int,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        selected: list[SearchResult] = []

        group_counts: dict[
            tuple[str, str],
            int,
        ] = {}

        seen_chunk_ids: set[str] = set()

        for result in results:
            chunk = result.chunk

            if chunk.id in seen_chunk_ids:
                continue

            group_key = self._build_group_key(result)

            if group_key is not None:
                current_count = group_counts.get(
                    group_key,
                    0,
                )

                if current_count >= self.max_results_per_symbol:
                    continue

            selected.append(result)
            seen_chunk_ids.add(chunk.id)

            if group_key is not None:
                group_counts[group_key] = (
                    group_counts.get(
                        group_key,
                        0,
                    )
                    + 1
                )

            if len(selected) >= top_k:
                break

        return [
            SearchResult(
                chunk=result.chunk,
                score=result.score,
                rank=index + 1,
            )
            for index, result in enumerate(selected)
        ]

    @staticmethod
    def _build_group_key(
        result: SearchResult,
    ) -> tuple[str, str] | None:
        chunk = result.chunk

        symbol = chunk.metadata.get("symbol")

        if not isinstance(symbol, str):
            return None

        symbol = symbol.strip()

        if not symbol:
            return None

        return (
            chunk.source,
            symbol,
        )


class DiversifiedRetriever:
    """
    Retriever 装饰器。

    内部先从基础 Retriever 获取更大的候选集，
    再通过 ResultDiversifier 选出最终 Top-k。
    """

    def __init__(
        self,
        base_retriever: Retriever,
        diversifier: ResultDiversifier,
        candidate_multiplier: int = 3,
        candidate_count: int | None = None,
    ) -> None:
        if candidate_multiplier <= 0:
            raise ValueError("candidate_multiplier " "must be greater than 0")

        if candidate_count is not None and candidate_count <= 0:
            raise ValueError("candidate_count must be " "greater than 0")

        self.base_retriever = base_retriever
        self.diversifier = diversifier
        self.candidate_multiplier = candidate_multiplier
        self.candidate_count = candidate_count

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if self.candidate_count is None:
            candidate_k = top_k * self.candidate_multiplier

        else:
            candidate_k = max(
                top_k,
                self.candidate_count,
            )

        candidates = self.base_retriever.retrieve(
            query=query,
            top_k=candidate_k,
            minimum_score=minimum_score,
        )

        return self.diversifier.diversify(
            results=candidates,
            top_k=top_k,
        )
