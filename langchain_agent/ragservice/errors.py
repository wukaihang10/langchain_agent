"""Compatibility imports for the former repository-knowledge package."""

from langchain_agent.repository_knowledge.errors import (
    RepositoryChangedDuringIndexingError,
    RepositoryKnowledgeError,
)

__all__ = [
    "RepositoryChangedDuringIndexingError",
    "RepositoryKnowledgeError",
]
