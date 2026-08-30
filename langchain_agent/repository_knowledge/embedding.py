from __future__ import annotations

from collections.abc import Sequence
from threading import RLock
from typing import TYPE_CHECKING

import numpy as np
from numpy.typing import NDArray

from langchain_agent.repository_knowledge.errors import EmbeddingError

if TYPE_CHECKING:
    from sentence_transformers import SentenceTransformer

FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]


class SentenceTransformerEmbeddingClient:
    """
    使用Sentence Transformers生成文本向量。

    文档：
        直接编码原始文本。

    查询：
        先添加查询指令，再进行编码。
    """

    DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"

    DEFAULT_QUERY_INSTRUCTION = "为这个句子生成表示以用于检索相关文章："

    def __init__(
        self,
        model_name: str = DEFAULT_MODEL_NAME,
        query_instruction: str | None = DEFAULT_QUERY_INSTRUCTION,
        batch_size: int = 32,
        normalize_embeddings: bool = True,
        show_progress_bar: bool = False,
        device: str | None = None,
    ) -> None:
        if not model_name.strip():
            raise ValueError("model_name cannot be empty")

        if batch_size <= 0:
            raise ValueError("batch_size must be greater than 0")

        self.model_name = model_name
        self.query_instruction = query_instruction
        self.batch_size = batch_size
        self.normalize_embeddings = normalize_embeddings
        self.show_progress_bar = show_progress_bar
        self.device = device
        self._model: SentenceTransformer | None = None
        self._dimension: int | None = None
        self._lock = RLock()

    @property
    def model_id(self) -> str:
        return self.model_name

    @property
    def dimension(self) -> int:
        with self._lock:
            if self._dimension is None:
                try:
                    dimension = self._get_model().get_embedding_dimension()
                except EmbeddingError:
                    raise
                except Exception as error:
                    raise EmbeddingError(
                        "Embedding model failed to report its dimension: "
                        f"{error}"
                    ) from error

                if dimension is None:
                    raise EmbeddingError(
                        "The embedding model did not provide an embedding dimension"
                    )

                self._dimension = int(dimension)

            return self._dimension

    def embed_documents(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix:
        """
        批量生成文档向量。

        返回：
            （文档数量，向量维度）
        """

        validated_texts = self._validate_texts(
            texts=texts,
            name="documents",
        )

        if not validated_texts:
            return np.empty(
                shape=(0, self.dimension),
                dtype=np.float32,
            )

        return self._encode(validated_texts)

    def embed_query(
        self,
        query: str,
    ) -> FloatVector:
        """
        生成某个查询向量。

        返回：
            （向量维度，）
        """

        validated_query = self._validate_text(
            text=query,
            name="query",
        )

        model_input = self._build_query_input(validated_query)

        embeddings = self._encode([model_input])

        return embeddings[0]

    def _encode(
        self,
        texts: Sequence[str],
    ) -> FloatMatrix:
        with self._lock:
            try:
                embeddings = self._get_model().encode(
                    list(texts),
                    batch_size=self.batch_size,
                    show_progress_bar=self.show_progress_bar,
                    convert_to_numpy=True,
                    normalize_embeddings=self.normalize_embeddings,
                )
            except EmbeddingError:
                raise
            except Exception as error:
                raise EmbeddingError(
                    f"Embedding model failed to encode text: {error}"
                ) from error

            matrix = np.asarray(
                embeddings,
                dtype=np.float32,
            )

            if matrix.ndim == 1:
                matrix = matrix.reshape(1, -1)

            if matrix.ndim != 2:
                raise EmbeddingError(
                    "Embedding result must be a two-dimensional matrix"
                )

            if matrix.shape[1] != self.dimension:
                raise EmbeddingError(
                    "Unexpected embedding dimension: "
                    f"expected {self.dimension}, got {matrix.shape[1]}"
                )

            return matrix

    def _get_model(self) -> SentenceTransformer:
        with self._lock:
            if self._model is None:
                try:
                    from sentence_transformers import SentenceTransformer

                    self._model = SentenceTransformer(
                        self.model_name,
                        device=self.device,
                    )
                except Exception as error:
                    raise EmbeddingError(
                        f"Failed to load embedding model {self.model_name!r}: {error}"
                    ) from error

            return self._model

    def _build_query_input(
        self,
        query: str,
    ) -> str:
        if not self.query_instruction:
            return query

        return f"{self.query_instruction}{query}"

    @classmethod
    def _validate_texts(
        cls,
        texts: Sequence[str],
        name: str,
    ) -> list[str]:
        validated: list[str] = []

        for index, text in enumerate(texts):
            validated.append(
                cls._validate_text(
                    text=text,
                    name=f"{name}[{index}]",
                )
            )

        return validated

    @staticmethod
    def _validate_text(
        text: str,
        name: str,
    ) -> str:
        if not isinstance(text, str):
            raise TypeError(f"{text} must be a string")

        stripped = text.strip()

        if not stripped:
            raise ValueError(f"{name} cannot be empty")

        return stripped
