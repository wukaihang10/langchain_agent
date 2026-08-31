# Agent Learning State

## Frozen stages

- Core LangGraph / create_agent loop
- Runtime context
- Tool execution
- Permission + HITL
- SQLite persistence / sessions
- Summarization
- MCP basics
- Subagents
  - code-researcher
  - code-reviewer

## Current stage

Harness Robustness

Completed in this stage:

- Expected repository tool failures have one model-visible error boundary.
- MCP transport retries are limited to explicitly safe tools.
- Runtime composition, CLI, harness, integrations, persistence, and Agent tool
  adapters have explicit package boundaries.
- The production Agent can be constructed from injected dependencies for
  deterministic evaluation without starting the CLI.
