# ADR 0001: Repository knowledge boundary

- Status: Accepted
- Date: 2026-08-29

## Context

Repository RAG code was split between an Agent-facing graph, a manager, and a
large `ragservice` package. The graph and manager both orchestrated retrieval,
and the graph reached through the manager into retriever, reranker, and context
builder internals. Index files were also written into the repository being
analyzed even though they are derived application data.

The repository knowledge capability must be reusable by Agent tools, the CLI,
and future Web APIs without depending on `AgentContext`, `ToolRuntime`, or
`ToolMessage`.

## Decision

- One `RepositoryKnowledgeService` instance is bound to one repository.
- The application creates and injects shared embedding clients.
- The embedding dependency is required; the service does not create a default.
- `prepare()` owns index loading or rebuilding; `search()` requires a ready
  index and does not implicitly prepare it.
- Indexes are application-owned caches. Their paths are supplied by the
  composition root and should live below `.agent/indexes/` for the CLI app.
- `RepositoryKnowledgeService` owns indexing and retrieval only.
- Answer generation belongs to a separate service that composes repository
  knowledge with a chat model.
- Agent-specific validation, truncation, formatting, and error translation stay
  in the Agent tool adapter.
- The current linear RAG Graph will leave the production call path. A graph may
  be reintroduced behind the service if branching, parallelism, persistence,
  interrupts, or node-level streaming becomes a real requirement.

## Public contract

The stable public surface consists of:

- `RepositoryKnowledgeConfig`
- `RepositoryKnowledgeService`
- `EmbeddingClient`
- `IndexReadyResult`
- `SearchResponse`
- `Evidence`
- expected repository-knowledge exceptions

Internal `Document`, `Chunk`, retriever, vector-store, reranker, graph-state,
and index-storage types are not part of this contract.

## Consequences

- Repository state is isolated per service instance.
- Expensive model clients can be shared across repositories.
- Agent, CLI, and Web adapters can format the same structured results
  differently.
- Existing RAG internals can be moved incrementally behind the facade.
- Application code must explicitly choose an index cache path and prepare the
  service before searching.

