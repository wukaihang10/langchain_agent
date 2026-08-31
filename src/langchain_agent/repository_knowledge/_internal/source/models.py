from dataclasses import dataclass, field
from typing import Any


Metadata = dict[str, Any]


@dataclass
class Document:
    """One source file loaded from a repository."""

    id: str
    content: str
    source: str
    metadata: Metadata = field(default_factory=dict)


@dataclass
class Chunk:
    """One indexable unit derived from a source document."""

    id: str
    document_id: str
    content: str
    source: str
    index: int
    start_char: int
    end_char: int
    metadata: Metadata = field(default_factory=dict)
    embedding_content: str | None = None

    @property
    def content_for_embedding(self) -> str:
        return (
            self.embedding_content
            if self.embedding_content is not None
            else self.content
        )
