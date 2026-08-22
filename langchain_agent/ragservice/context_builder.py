from __future__ import annotations

from collections.abc import Sequence

from langgraph_agent.ragservice.models import (
    BuiltContext,
    ContextItem,
    SearchResult,
)


class ContextBuilder:
    """
    将检索结果构造成适合交给 LLM 的资料上下文。

    当前负责：
    1. 按 Chunk ID 去重；
    2. 保持检索排序；
    3. 限制最大资料数量；
    4. 限制 Context 总字符数；
    5. 限制同一代码符号占用的资料数量；
    6. 添加资料编号、来源和位置。
    """

    def __init__(
        self,
        max_context_characters: int = 6000,
        max_items: int = 5,
        max_items_per_symbol: int = 2,
        include_scores: bool = False,
    ) -> None:
        if max_context_characters <= 0:
            raise ValueError("max_context_characters " "must be greater than 0")

        if max_items <= 0:
            raise ValueError("max_items must be greater than 0")

        if max_items_per_symbol <= 0:
            raise ValueError("max_items_per_symbol " "must be greater than 0")

        self.max_context_characters = max_context_characters
        self.max_items = max_items
        self.max_items_per_symbol = max_items_per_symbol
        self.include_scores = include_scores

    def build(
        self,
        results: Sequence[SearchResult],
    ) -> BuiltContext:
        unique_results = self._deduplicate(results)

        selected_items: list[ContextItem] = []
        sections: list[str] = []
        used_characters = 0

        symbol_counts: dict[
            tuple[str, str],
            int,
        ] = {}

        for result in unique_results:
            if len(selected_items) >= self.max_items:
                break

            group_key = self._build_group_key(result)

            if group_key is not None:
                current_count = symbol_counts.get(
                    group_key,
                    0,
                )

                if current_count >= self.max_items_per_symbol:
                    continue

            context_id = f"source_{len(selected_items) + 1}"

            item = ContextItem(
                context_id=context_id,
                chunk=result.chunk,
                score=result.score,
                retrieval_rank=result.rank,
            )

            section = self._format_item(item)

            separator_size = 2 if sections else 0

            projected_size = used_characters + separator_size + len(section)

            if projected_size > self.max_context_characters:
                continue

            selected_items.append(item)
            sections.append(section)
            used_characters = projected_size

            if group_key is not None:
                symbol_counts[group_key] = (
                    symbol_counts.get(
                        group_key,
                        0,
                    )
                    + 1
                )

        text = "\n\n".join(sections)

        return BuiltContext(
            text=text,
            items=selected_items,
            character_count=len(text),
        )

    @staticmethod
    def _deduplicate(
        results: Sequence[SearchResult],
    ) -> list[SearchResult]:
        seen_ids: set[str] = set()
        unique_results: list[SearchResult] = []

        for result in results:
            chunk_id = result.chunk.id

            if chunk_id in seen_ids:
                continue

            seen_ids.add(chunk_id)
            unique_results.append(result)

        return unique_results

    @staticmethod
    def _build_group_key(
        result: SearchResult,
    ) -> tuple[str, str] | None:
        chunk = result.chunk

        symbol = chunk.metadata.get("symbol")

        if not isinstance(symbol, str):
            return None

        stripped_symbol = symbol.strip()

        if not stripped_symbol:
            return None

        return (
            chunk.source,
            stripped_symbol,
        )

    def _format_item(
        self,
        item: ContextItem,
    ) -> str:
        chunk = item.chunk
        metadata = chunk.metadata

        header_lines = [
            f"[{item.context_id}]",
            f"来源：{chunk.source}",
        ]

        symbol = metadata.get("symbol")

        if symbol:
            header_lines.append(f"符号：{symbol}")

        symbol_type = metadata.get("symbol_type")

        if symbol_type:
            header_lines.append(f"类型：{symbol_type}")

        start_line = metadata.get("start_line")
        end_line = metadata.get("end_line")

        if isinstance(start_line, int) and isinstance(end_line, int):
            header_lines.append(f"行号：{start_line}-{end_line}")
        else:
            header_lines.append(
                "位置：字符 " f"{chunk.start_char}-" f"{chunk.end_char}"
            )

        part_index = metadata.get("part_index")
        part_count = metadata.get("part_count")

        if (
            isinstance(part_index, int)
            and isinstance(part_count, int)
            and part_count > 1
        ):
            header_lines.append("片段：" f"{part_index + 1}/" f"{part_count}")

        symbol_start = metadata.get("symbol_start_line")
        symbol_end = metadata.get("symbol_end_line")

        if isinstance(symbol_start, int) and isinstance(symbol_end, int):
            header_lines.append("完整符号范围：" f"{symbol_start}-" f"{symbol_end}")

        if self.include_scores:
            header_lines.extend(
                [
                    ("检索排名：" f"{item.retrieval_rank}"),
                    ("检索分数：" f"{item.score:.6f}"),
                ]
            )

        header = "\n".join(header_lines)

        return f"{header}\n\n" f"{chunk.content}"
