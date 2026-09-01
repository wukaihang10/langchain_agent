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
- When responsibilities become materially mixed and cause repeated coupling or confusion, prefer fixing the underlying boundary instead of accumulating local patches.
- Treat observability, failure handling, permissions, persistence, context
  management, and evaluation as first-class Agent concerns.
- Prefer evidence-driven changes once evaluation exists.
  When behavior can be measured, establish or reproduce the failure before
  changing architecture or prompts, then compare the result after the change.
  Avoid tuning Agent behavior purely from intuition when an eval or trace can
  provide evidence.

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

## Scope discipline

Do not continuously refactor working architecture merely because a cleaner
abstraction is possible.

Prefer targeted improvements justified by:

- an observed failure mode,
- an evaluation result,
- a concrete maintenance problem,
- or a capability that the current design cannot support cleanly.

Once a subsystem has a clear boundary and is working reliably, prefer moving
forward to the next learning objective over polishing it indefinitely.

## Skill usage

Installed engineering skills are supporting workflows, not replacements for
the learning process above.

For architecture, domain modeling, responsibility boundaries, state ownership,
failure semantics, or other important tradeoffs, do not use a skill to skip
direct discussion with the user.

Skills may proactively handle research, mechanical implementation, validation,
testing, and review. Important design conclusions should still be surfaced to
the user and understood before implementation.

## Coding participation

Use different levels of assistance depending on the task:

- Architecture and important design decisions:
  Ask the user to reason first when the decision involves ownership,
  responsibility boundaries, state/context semantics, persistence, failure
  semantics, permissions, public interfaces, orchestration, or meaningful
  tradeoffs.

  Do not require user participation for minor naming, syntax, API lookup, or
  routine implementation choices.

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

## Agent skills

### Issue tracker

Issues are tracked in this repository's GitHub Issues. See `docs/agents/issue-tracker.md`.

### Triage labels

The default five-role triage label vocabulary is used. See `docs/agents/triage-labels.md`.

### Domain docs

Domain documentation uses a single-context layout. See `docs/agents/domain.md`.
