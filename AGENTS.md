# AGENTS.md

## Project purpose

This repository is primarily a learning project for modern LLM/Agent
engineering.

The goal is not merely to finish features quickly. Development should help
the user understand mainstream Agent architecture, framework capabilities,
responsibility boundaries, and engineering tradeoffs.

## Development principles

- Prefer current official / mainstream framework capabilities and extension
  points.
- Do not invent custom abstractions when standard framework mechanisms are
  sufficient.
- If a custom abstraction is necessary, explain why the standard mechanism is
  insufficient.
- Pay attention to responsibility boundaries and ownership.
- When responsibilities become mixed, prefer fixing the architecture instead
  of adding another local patch.
- Treat observability, failure handling, permissions, persistence, context
  management, and evaluation as first-class Agent concerns.

## Teaching workflow

When encountering an important new concept:

1. Explain the concept first.
2. Explain common industry / framework practice.
3. Inspect how the current repository handles it.
4. Discuss architectural choices and responsibility boundaries.
5. Agree on the design before making significant changes.
6. Implement.
7. Review the diff and explain important changes.
8. Validate behavior with runtime output / LangSmith traces when appropriate.

Do not immediately implement architecture-level decisions without discussion.

## Coding participation

Use different levels of assistance depending on the task:

- Architecture and important design decisions:
  Ask the user to reason about the design first, then review together.

- Small but educational implementation:
  Let the user write a first version when useful, then review it.

- Mechanical, repetitive, or boilerplate implementation:
  Implement it directly.

The user does not need to memorize APIs or minor project details.
Prioritize the ability to:
requirements -> decomposition -> architecture -> documentation -> implementation.

## Current project direction

Prefer modern LangChain / LangGraph standard architecture.

The Agent should remain general-purpose rather than becoming tightly coupled
to one repository-analysis task.