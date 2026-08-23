from __future__ import annotations

import math
from collections import Counter, defaultdict
from collections.abc import Sequence

from langchain_agent.ragservice.code_tokenizer import CodeTokenizer
from langchain_agent.ragservice.models import Chunk, SearchResult


class InMemoryBM25Index:
    """
    内存 BM25 倒排索引。

    当前保存：

    chunks
    document lengths
    term -> {document_position: term_frequency}
    average document length
    """

    def __init__(
        self,
        tokenizer: CodeTokenizer,
        k1: float = 1.5,
        b: float = 0.75,
    ) -> None:
        if k1 <= 0:
            raise ValueError("k1 must be greater than 0")

        if not 0 <= b <= 1:
            raise ValueError("b must be between 0 and 1")

        self.tokenizer = tokenizer
        self.k1 = float(k1)
        self.b = float(b)

        self._chunks: list[Chunk] = []

        self._document_lengths: list[int] = []

        self._postings: dict[
            str,
            dict[int, int],
        ] = {}

        self._average_document_length = 0.0

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        return tuple(self._chunks)

    def replace(self, chunks: Sequence[Chunk]) -> None:
        """
        使用新的 Chunk 集合整体替换 BM25 索引。

        所有状态先在局部变量中建立，
        完成后再替换当前对象状态。
        """

        chunk_list = list(chunks)

        self._validate_chunk_ids(chunk_list)

        new_document_lengths: list[int] = []

        new_postings: dict[
            str,
            dict[int, int],
        ] = defaultdict(dict)

        for position, chunk in enumerate(chunk_list):
            searchable_text = self._build_searchable_text(chunk)

            tokens = self.tokenizer.tokenize(searchable_text)

            term_frequencies = Counter(tokens)

            new_document_lengths.append(len(tokens))

            for term, frequency in term_frequencies.items():
                new_postings[term][position] = frequency

        if chunk_list:
            average_document_length = sum(new_document_lengths) / len(chunk_list)
        else:
            average_document_length = 0.0

        self._chunks = chunk_list
        self._document_lengths = new_document_lengths
        self._postings = {
            term: dict(postings) for term, postings in new_postings.items()
        }
        self._average_document_length = average_document_length

    def search(
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

        if self.is_empty:
            return []

        if self._average_document_length <= 0:
            return []

        query_tokens = self.tokenizer.tokenize(query)

        if not query_tokens:
            return []

        query_term_frequencies = Counter(query_tokens)

        scores: dict[int, float] = defaultdict(float)

        document_count = len(self._chunks)

        for term, query_frequency in query_term_frequencies.items():
            postings = self._postings.get(term)

            if not postings:
                continue

            document_frequency = len(postings)

            inverse_document_frequency = math.log(
                1.0
                + (document_count - document_frequency + 0.5)
                / (document_frequency + 0.5)
            )

            for position, term_frequency in postings.items():
                document_length = self._document_lengths[position]

                length_normalization = (
                    1.0
                    - self.b
                    + self.b * document_length / self._average_document_length
                )

                denominator = term_frequency + self.k1 * length_normalization

                term_score = (
                    inverse_document_frequency
                    * (term_frequency * (self.k1 + 1.0))
                    / denominator
                )

                scores[position] += query_frequency * term_score

        ranked_positions = sorted(
            scores,
            key=lambda position: (
                -scores[position],
                position,
            ),
        )

        results: list[SearchResult] = []

        for position in ranked_positions:
            score = float(scores[position])

            if minimum_score is not None and score < minimum_score:
                continue

            results.append(
                SearchResult(
                    chunk=self._chunks[position],
                    score=score,
                    rank=len(results) + 1,
                )
            )

            if len(results) >= top_k:
                break

        return results

    def clear(self) -> None:
        self._chunks.clear()
        self._document_lengths.clear()
        self._postings.clear()
        self._average_document_length = 0.0

    @staticmethod
    def _build_searchable_text(
        chunk: Chunk,
    ) -> str:
        """
        为 BM25 构造代码检索文本。

        BM25 需要精确词项，因此显式加入：
        文件路径、符号、类型、父符号和源码。

        不直接使用 embedding_content，
        避免把为向量模型准备的中文标签重复加入。
        """

        metadata = chunk.metadata

        fields = [
            chunk.source,
        ]

        for key in (
            "symbol",
            "symbol_type",
            "parent_symbol",
        ):
            value = metadata.get(key)

            if isinstance(value, str) and value:
                fields.append(value)

        fields.append(chunk.content)

        return "\n".join(fields)

    @staticmethod
    def _validate_chunk_ids(
        chunks: Sequence[Chunk],
    ) -> None:
        chunk_ids = [chunk.id for chunk in chunks]

        if len(chunk_ids) != len(set(chunk_ids)):
            raise ValueError("The BM25 index contains " "duplicate Chunk IDs")


class BM25Retriever:
    """
    将文本查询交给 BM25 索引。
    """

    def __init__(
        self,
        index: InMemoryBM25Index,
    ) -> None:
        self.index = index

    def retrieve(
        self,
        query: str,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        return self.index.search(
            query=query,
            top_k=top_k,
            minimum_score=minimum_score,
        )
