from __future__ import annotations

from collections.abc import Sequence
from typing import Protocol

import numpy as np
from numpy.typing import NDArray


FloatVector = NDArray[np.float32]
FloatMatrix = NDArray[np.float32]


class EmbeddingClient(Protocol):
    """Embedding dependency supplied by the application composition root."""

    model_id: str
    dimension: int

    def embed_documents(self, texts: Sequence[str]) -> FloatMatrix: ...

    def embed_query(self, query: str) -> FloatVector: ...

