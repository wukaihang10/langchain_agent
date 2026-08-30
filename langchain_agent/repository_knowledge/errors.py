class RepositoryKnowledgeError(RuntimeError):
    """Base class for expected repository-knowledge failures."""


class InvalidRepositoryError(RepositoryKnowledgeError):
    """The configured repository path cannot be indexed."""


class IndexNotReadyError(RepositoryKnowledgeError):
    """Search was requested before the repository index was prepared."""


class IndexBuildError(RepositoryKnowledgeError):
    """A repository index could not be prepared."""


class EmbeddingError(RepositoryKnowledgeError):
    """The configured embedding adapter could not encode repository text."""


class QueryExpansionError(RepositoryKnowledgeError):
    """The configured query expander could not produce supplemental queries."""


class RepositoryChangedDuringIndexingError(IndexBuildError):
    """The repository kept changing while a stable index was being built."""
