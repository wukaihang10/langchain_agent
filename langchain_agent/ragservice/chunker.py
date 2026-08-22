from __future__ import annotations

import hashlib
from collections.abc import Iterable

from langgraph_agent.ragservice.models import Chunk, Document


class TextChunker:
    """
    一个带自然边界感知的文本切分器。
    它不会严格在chunk_size 位置切断，而是优先向前寻找：
    - 段落边界
    - 换行
    - 句号
    - 逗号
    - 空格

    如果找不到合适边界，才按固定字符位置切分。
    """

    DEFAULT_SEPARATORS = {
        "\n\n",
        "\n",
        "。",
        "！",
        "？",
        "；",
        "，" ".",
        "!",
        "?",
        ";",
        ",",
        " ",
    }

    def __init__(
        self,
        chunk_size: int = 800,
        chunk_overlap: int = 100,
        minimum_chunk_ratio: float = 0.5,
        separators: Iterable[str] | None = None,
    ) -> None:
        if chunk_size <= 0:
            raise ValueError("chunk_size must be greater than 0")

        if chunk_overlap < 0:
            raise ValueError("chunk_overlap cannot be negative")

        if chunk_overlap >= chunk_size:
            raise ValueError("chunk_overlap must be smaller than chunk_size")

        if not 0 < minimum_chunk_ratio <= 1:
            raise ValueError("minimum_chunk_ratio must be within (0, 1]")

        self.chunk_size = chunk_size
        self.chunk_overlap = chunk_overlap
        self.minimum_chunk_ratio = minimum_chunk_ratio

        self.separators = tuple(
            separators if separators is not None else self.DEFAULT_SEPARATORS
        )

    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]:
        chunks: list[Chunk] = []

        for document in documents:
            chunks.extend(self.split_document(document))

        return chunks

    def split_document(
        self,
        document: Document,
    ) -> list[Chunk]:
        text = document.content

        if not text.strip():
            return []

        chunks: list[Chunk] = []
        start = 0
        chunk_index = 0

        while start < len(text):
            hard_end = min(start + self.chunk_size, len(text))

            split_end = self._find_split_end(
                text=text,
                start=start,
                hard_end=hard_end,
            )

            content_start, content_end = self._trim_span(
                text=text,
                start=start,
                end=split_end,
            )

            if content_start < content_end:
                content = text[content_start:content_end]

                chunk = self._create_chunk(
                    document=document,
                    content=content,
                    index=chunk_index,
                    start_char=content_start,
                    end_char=content_end,
                )

                chunks.append(chunk)
                chunk_index += 1

            if split_end >= len(text):
                break

            next_start = split_end - self.chunk_overlap

            start = max(next_start, start + 1)

        return chunks

    def _find_split_end(
        self,
        text: str,
        start: int,
        hard_end: int,
    ) -> int:
        """
        在不超过 hard_end 的范围内寻找合适的结束位置。
        不允许为了寻找边界而让 Chunk 过短。
        """

        if hard_end >= len(text):
            return len(text)

        minimum_size = max(1, int(self.chunk_size * self.minimum_chunk_ratio))

        search_start = min(start + minimum_size, hard_end)

        for separator in self.separators:
            position = text.rfind(
                separator,
                search_start,
                hard_end,
            )

            if position == -1:
                continue

            return position + len(separator)

        return hard_end

    @staticmethod
    def _trim_span(
        text: str,
        start: int,
        end: int,
    ) -> tuple[int, int]:
        """去除Chunk两端空白"""

        while start < end and text[start].isspace():
            start += 1

        while start < end and text[end - 1].isspace():
            end -= 1

        return start, end

    def _create_chunk(
        self,
        document: Document,
        content: str,
        index: int,
        start_char: int,
        end_char: int,
    ) -> Chunk:
        chunk_id = self._build_chunk_id(
            document_id=document.id,
            content=content,
            start_char=start_char,
            end_char=end_char,
        )

        metadata = {
            **document.metadata,
            "chunk_index": index,
            "start_char": start_char,
            "end_char": end_char,
            "character_count": len(content),
        }

        return Chunk(
            id=chunk_id,
            document_id=document.id,
            content=content,
            source=document.source,
            index=index,
            start_char=start_char,
            end_char=end_char,
            metadata=metadata,
        )

    @staticmethod
    def _build_chunk_id(
        document_id: str,
        content: str,
        start_char: int,
        end_char: int,
    ) -> str:
        """
        这里不使用index来构造id是因为文档里的chunk index会随着开头加入的文字而改变。

        现在使用sha256能够保证同样文档和切分结果得到同样ID，内容发生变化后ID会变。
        """
        raw_id = f"{document_id}\0{start_char}\0{end_char}\0{content}"

        digest = hashlib.sha256(raw_id.encode("utf-8")).hexdigest()[:16]

        return f"{document_id}:chunk:{digest}"
