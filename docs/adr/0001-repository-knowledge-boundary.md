# Repository knowledge boundary

Repository knowledge is an independent, repository-scoped capability exposed through a stable service boundary rather than through an Agent-specific RAG graph. Agent and interface layers may adapt its structured results, but they must not own or reach into indexing and retrieval internals; embedding and cache locations are supplied by application composition, and derived indexes remain application-owned data rather than repository contents.

We chose this boundary so repository retrieval can be reused by Agents, CLI, evaluations, and future interfaces without depending on LangChain runtime types, while keeping the option to reintroduce an internal graph later if retrieval genuinely requires branching, persistence, interrupts, or other graph-specific behavior.
