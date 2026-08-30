# Repository knowledge package layout

`langchain_agent.repository_knowledge` is organized by responsibility and
stability, not by Python construct type.

## Public layer

- `service.py` is the use-case facade: prepare an index and search it.
- `models.py` contains stable values returned to callers.
- `errors.py` contains the failures callers are expected to handle.
- `config.py` contains repository-knowledge policy.
- `ports.py` contains required external dependency protocols.
- `embedding.py` provides one concrete embedding adapter for application
  composition.

Agent tools, CLI code, and future Web adapters may depend on this layer. They
must not import `_internal` modules.

## Internal layer

- `_internal/backend.py` composes the indexing and retrieval pipeline.
- `_internal/source/` owns source models and protocols, loads repository files,
  and creates Python-aware chunks.
- `_internal/indexing/` owns index-result models and builds, persists,
  validates, and reloads indexes.
- `_internal/retrieval/` implements dense and lexical retrieval, fusion,
  optional reranking, and context selection. Its retrieval models and protocols
  stay in the same subpackage.

## Placement rules

When adding code, decide where it belongs by asking who owns the behavior:

1. A type that callers exchange with the capability belongs in the public
   layer.
2. A framework or model adapter chosen by the application belongs beside the
   public port it implements.
3. A component used only to implement indexing or retrieval belongs in the
   matching `_internal` subpackage.
4. Agent-specific formatting and tool runtime behavior belong in
   `langchain_agent.tools`, not in repository knowledge.
5. Answer generation belongs in a separate future capability that consumes
   `SearchResponse`; it must not be added to `RepositoryKnowledgeService`.

Avoid generic catch-all modules such as `utils.py`, `common.py`, or a single
project-wide `errors.py`. A shared module is justified only when the concepts
have the same owner and change for the same reason.
