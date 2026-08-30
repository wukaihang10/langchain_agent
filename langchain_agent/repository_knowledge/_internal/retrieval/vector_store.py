from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
from numpy.typing import NDArray

from langchain_agent.repository_knowledge._internal.models import Chunk, SearchResult

FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]


class InMemoryVectorStore:
    """
    使用 Numpy 在内存中保存 Chunk向量，
    并通过余弦相似度执行精确 Top-k 检索。

    当前实现会在加入向量时自动归一化文档向量，
    在查询时自动归一化查询向量。
    """

    def __init__(
        self,
        dimension: int,
    ) -> None:
        if dimension <= 0:
            raise ValueError("dimension must be greater than 0")

        self.dimension = dimension

        self._chunks: list[Chunk] = []

        self._vectors = np.empty(
            shape=(0, dimension),
            dtype=np.float32,
        )

        self._id_to_position: dict[str, int] = {}

    def __len__(self) -> int:
        return len(self._chunks)

    @property
    def is_empty(self) -> bool:
        return not self._chunks

    @property
    def chunks(self) -> tuple[Chunk, ...]:
        """
        返回只读形式的 Chunk 集合。

        tuple 可以避免调用者直接对内部 list 执行append。
        """

        return tuple(self._chunks)

    def add(
        self,
        chunks: Sequence[Chunk],
        vectors: FloatMatrix,
    ) -> None:
        """
        批量加入 Chunk 和对应向量。

        chunks[i] 必须对应 vectors[i]。
        """

        if not chunks:
            matrix = np.asarray(
                vectors,
                dtype=np.float32,
            )

            if matrix.size != 0:
                raise ValueError("vectors must be empty when chunks is empty")

            return

        matrix = self._validate_matrix(vectors=vectors, expected_rows=len(chunks))

        self._validate_new_chunk_ids(chunks)

        normalized_vectors = self._normalize_matrix(matrix)

        self._chunks.extend(chunks)

        self._vectors = np.concatenate(
            [
                self._vectors,
                normalized_vectors,
            ],
            axis=0,
        )

        self._rebuild_id_positions()

    def replace(
        self,
        chunks: Sequence[Chunk],
        vectors: FloatMatrix,
    ) -> None:
        """
        使用一组全新的 Chunk 和向量替换当前索引。

        所有输入会先完成验证和归一化，然后才更新内部状态。
        """

        if not chunks:
            matrix = np.asarray(
                vectors,
                dtype=np.float32,
            )

            if matrix.size != 0:
                raise ValueError("vectors must be empty when chunks is empty")

            self.clear()
            return

        matrix = self._validate_matrix(
            vectors=vectors,
            expected_rows=len(chunks),
        )

        self._validate_incoming_chunk_ids(chunks)

        normalize_vectors = self._normalize_matrix(matrix)

        new_chunks = list(chunks)

        new_vectors = np.array(
            normalize_vectors,
            dtype=np.float32,
            copy=True,
        )

        new_id_to_position = {
            chunk.id: position for position, chunk in enumerate(new_chunks)
        }

        self._chunks = new_chunks
        self._vectors = new_vectors
        self._id_to_position = new_id_to_position

    def search(
        self,
        query_vector: FloatVector,
        top_k: int = 5,
        minimum_score: float | None = None,
    ) -> list[SearchResult]:
        """
        根据查询向量返回余弦相似度最高的 Top-k Chunk。
        """

        if top_k <= 0:
            raise ValueError("top_k must be greater than 0")

        if minimum_score is not None and not np.isfinite(minimum_score):
            raise ValueError("minimum_score must be finite")

        if self.is_empty:
            return []

        normalized_query = self._normalize_query_vector(query_vector)

        # _vectors 的形状：（Chunk 数量，dimension）
        # normalized_query：（dimension，）

        # 结果 scores：（Chunk 数量，）

        scores = self._vectors @ normalized_query

        sorted_positions = np.argsort(
            -scores,
            kind="stable",
        )

        results: list[SearchResult] = []

        for position in sorted_positions:
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

    def get(
        self,
        chunk_id: str,
    ) -> Chunk | None:
        """
        根据 Chunk ID 查找 Chunk。

        找不到属于正常查询结果，因此返回 None。
        """

        position = self._id_to_position.get(chunk_id)

        if position is None:
            return None

        return self._chunks[position]

    def delete(
        self,
        chunk_ids: Iterable[str],
    ) -> int:
        """
        删除指定 Chunk。

        返回实际删除数量。
        不存在的 ID 会被忽略。
        """

        ids_to_delete = set(chunk_ids)

        if not ids_to_delete:
            return 0

        keep_positions = [
            position
            for position, chunk in enumerate(self._chunks)
            if chunk.id not in ids_to_delete
        ]

        deleted_count = len(self._chunks) - len(keep_positions)

        if deleted_count == 0:
            return 0

        self._chunks = [self._chunks[position] for position in keep_positions]

        if keep_positions:
            self._vectors = self._vectors[
                np.asarray(
                    keep_positions,
                    dtype=np.int64,
                )
            ]

        else:
            self._vectors = np.empty(
                shape=(0, self.dimension),
                dtype=np.float32,
            )

        self._rebuild_id_positions()

        return deleted_count

    def clear(self) -> None:
        """
        清空全部 Chunk 和向量。
        """

        self._chunks.clear()

        self._vectors = np.empty(
            shape=(0, self.dimension),
            dtype=np.float32,
        )

        self._id_to_position.clear()

    def snapshot(
        self,
    ) -> tuple[list[Chunk], FloatMatrix]:
        """
        返回当前 Chunk 和向量的副本。

        用于索引持久化，避免持久化层直接访问
        _chunks 和 _vectors 私有属性。
        """

        chunks = list(self._chunks)

        vectors = np.array(
            self._vectors,
            dtype=np.float32,
            copy=True,
        )

        return chunks, vectors

    def _validate_matrix(
        self,
        vectors: FloatMatrix,
        expected_rows: int,
    ) -> FloatMatrix:
        matrix = np.asarray(
            vectors,
            dtype=np.float32,
        )

        if matrix.ndim != 2:
            raise ValueError("vectors must be a two-dimensional matrix")

        if matrix.shape[0] != expected_rows:
            raise ValueError(
                "Chunk and vector counts do not match: "
                f"{expected_rows} chunks, {matrix.shape[0]} vectors"
            )

        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Unexpected vector dimension: expected {self.dimension}, got {matrix.shape[1]}"
            )

        if not np.all(np.isfinite(matrix)):
            raise ValueError("vectors contain NaN or infinity")

        return matrix

    @staticmethod
    def _validate_incoming_chunk_ids(
        chunks: Sequence[Chunk],
    ) -> None:
        incoming_ids = [chunk.id for chunk in chunks]

        if len(incoming_ids) != len(set(incoming_ids)):
            raise ValueError("The input contains duplicate chunk IDs")

    def _validate_new_chunk_ids(
        self,
        chunks: Sequence[Chunk],
    ) -> None:
        """
        验证 add() 输入。
        除了要求本次输入内部没有重复ID，
        还要求这些ID没有存在于当前向量库。
        """

        self._validate_incoming_chunk_ids(chunks)

        incoming_ids = [chunk.id for chunk in chunks]

        existing_duplicates = set(incoming_ids) & self._id_to_position.keys()

        if existing_duplicates:
            duplicate_text = ", ".join(sorted(existing_duplicates))

            raise ValueError(f"Chunk IDs already exist: {duplicate_text}")

    def _normalize_matrix(
        self,
        matrix: FloatMatrix,
    ) -> FloatMatrix:
        norms = np.linalg.norm(
            matrix,
            axis=1,
            keepdims=True,
        )

        if np.any(norms == 0):
            raise ValueError("Document vectors cannot be zero vectors")

        normalized = matrix / norms

        return np.asarray(
            normalized,
            dtype=np.float32,
        )

    def _normalize_query_vector(
        self,
        vector: FloatVector,
    ) -> FloatVector:
        query = np.asarray(
            vector,
            dtype=np.float32,
        )

        if query.ndim != 1:
            raise ValueError("query_vector must be one-dimensional")

        if query.shape[0] != self.dimension:
            raise ValueError(
                "Unexpected query vector dimension: "
                f"expected {self.dimension}, got {query.shape[0]}"
            )

        if not np.all(np.isfinite(query)):
            raise ValueError("query_vector contains NaN or infinity")

        norm = np.linalg.norm(query)

        if norm == 0:
            raise ValueError("query_vector cannot be a zero vector")

        normalized = query / norm

        return np.asarray(
            normalized,
            dtype=np.float32,
        )

    def _rebuild_id_positions(self) -> None:
        self._id_to_position = {
            chunk.id: position for position, chunk in enumerate(self._chunks)
        }
