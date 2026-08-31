# ADR 0003: Application package boundaries

- Status: Accepted
- Date: 2026-08-31

## Context

The executable entry point previously owned command-line interaction, session
management, model and MCP construction, repository-knowledge service caching,
middleware construction, agent creation, checkpoint lifetime, and the main run
loop. This made the Agent definition difficult to reuse in evaluations and
made unrelated runtime responsibilities change together.

The repository-knowledge capability already has an accepted public boundary.
The application restructure must preserve that boundary and must not turn its
Agent tool adapter into part of the capability implementation.

## Decision

- `app/` owns application configuration, runtime contracts, dependency
  composition, Agent construction, and session runtime construction.
- `cli/` owns terminal input, output, slash commands, and HITL interaction.
- `harness/` owns cross-cutting Agent policies such as permission enforcement,
  expected tool-error translation, MCP retry behavior, and Git audit state.
- `tools/` remains the LangChain tool-adapter layer. ToolRuntime validation and
  model-facing formatting stay outside domain capabilities.
- `integrations/` owns external implementations for models, MCP, and Git
  process execution.
- `persistence/` owns session metadata and LangGraph checkpoints.
- `repository_knowledge/` keeps its existing public import path and internal
  architecture unchanged.
- The repository-root `main.py` is only an executable shim for `cli.app.main`.

The composition root is `app/bootstrap.py`. `app/agent.py` receives constructed
dependencies and defines the standard tools, middleware order, subagents,
prompt, context schema, and checkpointer used by the production Agent.

## Dependency rules

- Repository knowledge must not import the Agent runtime, CLI, harness, or tool
  adapters.
- Harness and tool adapters must not import the CLI or application bootstrap.
- Tool adapters may depend on `app.context`, which is the shared runtime
  contract injected by LangChain.
- CLI code may invoke application services but does not construct models,
  middleware, MCP clients, or checkpoint implementations directly.
- Git tools depend on the Git integration, not on Git audit middleware.

These rules are enforced by architecture tests.

## Consequences

- Evaluations can construct the production Agent with fake models and an
  in-memory checkpointer without starting the CLI or reading MCP configuration.
- Runtime resource ownership is visible in one composition root.
- Failure, retry, permission, and persistence semantics retain their existing
  owners while becoming easier to test independently.
- Moving files changes internal import paths, but persisted sessions,
  checkpoints, repository indexes, tool names, and Agent behavior do not
  require migration.
