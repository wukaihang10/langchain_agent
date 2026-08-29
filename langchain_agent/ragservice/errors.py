class RepositoryKnowledgeError(RuntimeError):
    """Expected repository-knowledge failure that is safe to show to the agent."""


class RepositoryChangedDuringIndexingError(RepositoryKnowledgeError):
    """The repository kept changing while a stable index was being built."""
