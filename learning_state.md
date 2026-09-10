# Agent Learning State

## Frozen stages

- Core LangGraph / `create_agent` loop
- Runtime context
- Tool execution
- Permission + HITL
- SQLite persistence / sessions
- Summarization
- MCP basics
- Subagents
  - code-researcher
  - code-reviewer
- Harness robustness foundations
  - Expected repository tool failures have one model-visible error boundary.
  - MCP transport retries are limited to explicitly replay-safe tools.
  - Runtime composition, CLI, harness, integrations, persistence, and Agent tool
    adapters have explicit package boundaries.
  - The application uses the standard `src/langchain_agent` package layout and
    exposes a `langchain-agent` CLI entry point.
  - The production Agent can be constructed from injected dependencies without
    starting the CLI.
- State and runtime boundaries
  - `sessions.json` owns user-visible session metadata, while LangGraph SQLite
    checkpoints own durable graph execution state.
  - Invocation context and active-execution task handles remain process-local;
    restart recovery never treats them as durable truth.
  - `SessionContinuation` classifies checkpoints as `EMPTY`, `READY`,
    `WAITING_HUMAN`, `RESUMABLE`, `OUTCOME_UNKNOWN`, or `NEEDS_REPAIR` and
    validates the corresponding continuation action.
  - Recovery preserves tool-call/result pairing, never silently replays an
    uncertain side effect, and keeps malformed state fail-closed.
  - `/stop` owns user-requested active-turn finalization; application `aclose()`
    only cancels and awaits process-local work before the checkpointer closes.
  - One active invocation is allowed per LangGraph thread, and restart derives
    recovery behavior from the latest checkpoint rather than a second run-state
    model.

## Deferred topics

These are intentionally outside the current scope and should be reopened only
when a concrete product requirement or observed failure justifies them:

- Persist call-time replay-policy evidence, policy versions, or idempotency keys.
  Reopen when a real side-effecting tool offers a durable idempotency contract or
  cross-version recovery becomes a requirement.
- Add an operation ledger, background supervisor, worker lease, heartbeat, or
  cross-process cancellation. Reopen if the application moves beyond a local
  foreground process.
- Generalize HITL recovery classification for arbitrary custom `StateGraph` node
  names. The official `create_agent` plus `HumanInTheLoopMiddleware` path works;
  reopen when custom graph support becomes an application requirement.
- Split permission, immediate retry, and historical recovery into richer policy
  models. Reopen when the current conservative rule causes an observed incorrect
  decision or a required capability cannot be represented safely.

## Current stage

Evaluation and Observability

### Learning objective

Build a small evidence loop that distinguishes process success from task success,
measures important Agent behavior through stable interfaces, and uses traces to
explain failures before prompts or architecture are changed.

### First milestone

- Distinguish unit tests, integration tests, behavioral evaluations, and trace
  inspection by the question each one answers.
- Define a minimal set of task-level success criteria before creating a dataset
  or evaluator.
- Reuse the injectable production Agent seam instead of building a parallel eval
  application.
- Start with a small representative baseline and defer prompt tuning until its
  failures are observable and classified.

### Learning roadmap

1. Define success before choosing evaluation tools.
   - Classify the Agent's representative task families.
   - Separate process success, execution success, task success, and degraded
     completion.
   - Write a small evaluation contract for the first task family.
2. Choose the evaluation surface that owns each quality question.
   - Use component evaluations for retrieval, permission, and error-boundary
     behavior.
   - Use trajectory evaluations for tool selection, arguments, and delegation.
   - Use final-response and end-to-end evaluations for user-visible task success.
3. Build a small, discriminating dataset.
   - Begin with 5-10 representative examples for one capability slice.
   - Include ordinary success, plausible-but-wrong answers, tool failures,
     insufficient evidence, unsafe actions, and multi-step investigation cases.
   - Record required facts, acceptable variation, forbidden behavior, and
     diagnostic labels for each example.
4. Design evaluators from the most reliable evidence available.
   - Prefer deterministic rules and structured trajectory checks.
   - Add rubric-based or LLM-as-judge evaluation only where semantic judgment is
     necessary, and calibrate it against human review.
5. Run a baseline through the injectable production Agent.
   - Reuse `build_agent(...)` with injected models, tools, policies, and
     checkpointer rather than constructing a parallel evaluation application.
   - Measure task quality, variance, latency, cost, and degraded completion as
     separate signals.
6. Diagnose evaluation failures with LangSmith traces.
   - Classify failures by task understanding, tool choice, arguments, retrieval,
     tool results, permission/retry behavior, delegation, answer synthesis, or
     external infrastructure.
   - Change prompts or architecture only after the failing boundary is supported
     by trace evidence.
7. Establish regression and production feedback loops.
   - Turn confirmed production failures into curated regression examples.
   - Compare experiments before and after changes and add stable deterministic
     checks to CI where appropriate.
   - Add sampled online evaluation and monitoring only after the offline contract
     is trustworthy.

The first capability slice will be repository-fact answers that are correct and
grounded in code evidence. This slice exercises retrieval, tool use, evidence
fidelity, and final-answer quality without beginning with the broader ambiguity
of full code review.
