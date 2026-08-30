# ADR 0002: Repository knowledge lifecycle

- Status: Accepted
- Date: 2026-08-30

## Context

A repository-scoped service is shared by all sessions that use the same
repository. Index preparation mutates in-memory dense and lexical indexes and
also writes an application-owned disk cache. Publishing a backend before
preparation completed allowed a failed build to appear ready. Preparing only
before the first search also allowed later repository edits to leave the Agent
using stale evidence.

## Decision

- The service exposes `UNPREPARED`, `PREPARING`, `READY`, and `FAILED` states.
- `is_ready` is derived from the service state, not from partially populated
  backend data structures.
- Preparation uses a candidate backend. The service publishes it only after
  indexing, validation, and cache persistence all succeed.
- A prepared index carries the exact repository snapshot it represents.
- `prepare()` is idempotent: it reuses a ready backend when the source snapshot
  is unchanged and rebuilds through a new candidate when it changes.
- The Agent tool calls `prepare()` before every search so freshness is checked.
- A failed initial build or refresh invalidates the active backend. Searches
  fail closed until a later preparation succeeds.
- A per-service reentrant lock serializes preparation and search for one
  repository. Different repository services retain independent locks.
- The shared sentence-transformer adapter serializes lazy model loading and
  encoding because it can be used by multiple repository services.
- Expected storage and embedding failures become repository-knowledge domain
  errors. Unexpected invariant violations retain their original exception
  type after the service cleans up its state.

## Consequences

- A failed candidate cannot become searchable even if it populated internal
  vector structures before failing.
- Repository edits are detected before each Agent search.
- Repeated searches pay the cost of hashing Python source files, but avoid
  embedding and rebuilding when the snapshot is unchanged.
- Concurrent operations for one repository are safe but serialized.
- A stale last-known-good index is not served after refresh failure.
- A future large-repository application may replace per-search hashing with a
  TTL or file watcher, but it must preserve the same freshness contract.
