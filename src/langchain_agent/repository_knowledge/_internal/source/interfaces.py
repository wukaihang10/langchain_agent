from __future__ import annotations

from pathlib import Path
from typing import Protocol

from langchain_agent.repository_knowledge._internal.source.models import (
    Chunk,
    Document,
)


class DocumentLoader(Protocol):
    def load_directory(
        self,
        directory: str | Path,
    ) -> list[Document]: ...


class DocumentChunker(Protocol):
    def split_documents(
        self,
        documents: list[Document],
    ) -> list[Chunk]: ...
