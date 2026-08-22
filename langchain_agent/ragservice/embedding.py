from __future__ import annotations

from collections.abc import Sequence

import numpy as np
from numpy.typing import NDArray
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

    # DEFAULT_MODEL_NAME = "BAAI/bge-small-zh-v1.5"
    DEFAULT_MODEL_NAME = (
        r"C:\Users\31972\.cache\huggingface\hub\models--BAAI--bge-small-zh-v1.5"
    )

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

        self.model = SentenceTransformer(
            model_name,
            device=device,
        )

        dimension = self.model.get_embedding_dimension()

        if dimension is None:
            raise ValueError(
                "The embedding model did not provide an embedding dimension"
            )

        self.dimension = int(dimension)

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
        embeddings = self.model.encode(
            list(texts),
            batch_size=self.batch_size,
            show_progress_bar=self.show_progress_bar,
            convert_to_numpy=True,
            normalize_embeddings=self.normalize_embeddings,
        )

        matrix = np.asarray(
            embeddings,
            dtype=np.float32,
        )

        if matrix.ndim == 1:
            matrix = matrix.reshape(1, -1)

        if matrix.ndim != 2:
            raise ValueError("Embedding result must be a two-dimensional matrix")

        if matrix.shape[1] != self.dimension:
            raise ValueError(
                f"Unexpected embedding dimension: expected {self.dimension}, got {matrix.shape[1]}"
            )

        return matrix

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
