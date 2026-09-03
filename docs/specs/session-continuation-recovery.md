# Session Continuation Recovery

Status: ready for implementation. Canonical tracker record:
https://github.com/wukaihang10/langchain_agent/issues/1

> Follow-up: issue #2 extends the `OUTCOME_UNKNOWN` baseline described here
> with explicit `RESOLVE_AND_CONTINUE` recovery. Statements below that say
> `OUTCOME_UNKNOWN` has no action describe issue #1's original scope.

## Problem Statement

The application persists LangGraph checkpoints and lets a user select a saved
session, but it currently treats every selected session as ready for a new human
message. If the previous Agent turn stopped before its graph reached a terminal
checkpoint, a fresh human message can be appended to unfinished execution state.
This is especially dangerous when an AI message contains tool calls whose
matching tool results were never persisted.

The user needs session recovery to distinguish a completed conversation from an
unfinished graph run, a pending human-in-the-loop decision, a safely replayable
operation, an operation whose external outcome is unknown, and malformed state
that requires explicit repair. Selecting a session must not itself restart work.
The application must decide which continuation operation is valid before the
CLI can send anything to LangGraph.

The immediate target is a local foreground Agent. The application does not yet
have a durable run registry, background supervisor, worker lease, heartbeat, or
operation ledger. The design must therefore provide truthful local recovery
without pretending to offer cloud-style automatic job recovery or exactly-once
external side effects.

## Solution

Add a session-continuation Module owned by the Application layer. Its Interface
is the single authoritative Seam for inspecting a session and executing any
new-turn or continuation request. The CLI presents the result and collects user
input; it does not inspect raw checkpoints or select LangGraph input types.

LangGraph's persisted `StateSnapshot` remains the only stored execution truth.
The Module derives one non-persisted `ContinuationStatus` from the latest
snapshot, unresolved tool calls, active interrupts, pending tasks, and injected
tool policy metadata. No separate persisted `RunState` is introduced.

The supported statuses are:

| Status | Meaning | Allowed continuation action |
| --- | --- | --- |
| `EMPTY` | The session has no checkpoint | Start the first turn |
| `READY` | The previous graph execution is complete and message history is valid | Start a new turn |
| `WAITING_HUMAN` | A persisted human-in-the-loop interrupt awaits a decision | Answer the interrupt |
| `RESUMABLE` | The previous graph execution has pending work and every pending operation is safe to replay | Continue the existing graph execution |
| `OUTCOME_UNKNOWN` | A pending tool may have produced an external effect whose result is not durable or verifiable | No automatic graph action in this version |
| `NEEDS_REPAIR` | The persisted history is structurally invalid or has no legal continuation path | No automatic graph action in this version |

The executable continuation actions in this version are:

| Action | Valid status | LangGraph input |
| --- | --- | --- |
| `START_TURN` | `EMPTY`, `READY` | A new human message input |
| `ANSWER_INTERRUPT` | `WAITING_HUMAN` | A `Command` carrying the HITL decision |
| `CONTINUE` | `RESUMABLE` | `None`, using the same thread configuration |

Session selection and session creation remain CLI/session-management actions,
not continuation actions. Selecting a session only inspects it. Creating a new
session remains an explicit user choice.

The Module returns a semantic inspection containing the status, observed
checkpoint identity, pending node names, pending interrupts, unresolved tool
calls, allowed actions, and a user-presentable reason. Before executing an
action, the Module rechecks the latest checkpoint identity. If the checkpoint
has changed since inspection, it rejects the stale request and requires a fresh
inspection.

Only `EMPTY` and `READY` accept a new human message. All other statuses block
ordinary chat input until the user performs an allowed continuation action or
chooses another session.

## User Stories

1. As a CLI user, I want selecting a saved session to be read-only, so that opening a conversation does not unexpectedly consume time, tokens, or tool calls.
2. As a CLI user, I want to see whether the selected session is ready, waiting for me, safely resumable, uncertain, or damaged, so that I understand what happened before acting.
3. As a CLI user, I want to begin the first turn in an empty session, so that newly created sessions continue to behave normally.
4. As a CLI user, I want to send a new message after the previous turn completed, so that ordinary multi-turn conversation remains unchanged.
5. As a CLI user, I want a new message rejected when the previous graph run is unfinished, so that my input is not appended to incompatible checkpoint state.
6. As a CLI user, I want a persisted HITL approval request to be shown again after reopening a session, so that I can make the decision that the graph is actually waiting for.
7. As a CLI user, I want the application to preserve the original HITL choices and tool arguments, so that recovery does not silently change what I am approving.
8. As a CLI user, I want approving or rejecting an HITL request to resume the same thread, so that the prior execution continues from its persisted interrupt.
9. As a CLI user, I want an unfinished non-tool node to be identified as resumable, so that I can explicitly continue the prior task without inventing a new message.
10. As a CLI user, I want a safely replayable unfinished tool step to be identified as resumable, so that I can continue the prior task without inspecting framework internals.
11. As a CLI user, I want continuation to require an explicit action after reopening a session, so that merely browsing sessions never restarts old work.
12. As a CLI user, I want existing filesystem and Git changes to remain untouched during session inspection, so that inspection cannot masquerade as rollback or recovery.
13. As a CLI user, I want a side-effecting unfinished tool to be reported as outcome unknown, so that the application does not duplicate an operation that may already have succeeded.
14. As a CLI user, I want an unclassified tool to be treated conservatively, so that missing policy metadata never grants automatic replay authority.
15. As a CLI user, I want a non-idempotent unfinished tool to be blocked from automatic continuation, so that nondeterministic work is not repeated silently.
16. As a CLI user, I want mixed safe and unsafe pending tool calls to be classified by the unsafe call, so that a safe sibling does not hide an external-risk condition.
17. As a CLI user, I want a malformed tool-call history with no normal continuation path to be reported as needing repair, so that the application does not send invalid history to the model provider.
18. As a CLI user, I want outcome uncertainty distinguished from structural repair, so that message validity is not confused with knowledge of external effects.
19. As a CLI user, I want the application to refuse automatic synthetic tool results, so that model-visible history does not claim an external fact that was never verified.
20. As a CLI user, I want to create or select another session even when the current one is blocked, so that an uncertain old task does not prevent unrelated work.
21. As a CLI user, I want the selected session's last-used time to change only when I perform real work, so that inspecting a session does not make it appear recently active.
22. As an Application caller, I want one Interface to inspect and execute continuation requests, so that CLI, future UI, and tests share the same safety rules.
23. As an Application caller, I want invalid actions rejected even if the caller ignores the advertised allowed actions, so that safety does not depend on a well-behaved CLI.
24. As an Application caller, I want stale inspection results rejected, so that an action cannot run against a checkpoint different from the one the user reviewed.
25. As a maintainer, I want the classifier to use task identities, interrupts, and tool-call IDs rather than message counts, so that parallel or partially completed tool work is interpreted correctly.
26. As a maintainer, I want tool replay policy injected into the continuation Module, so that the classifier uses the same policy truth as permission and retry middleware.
27. As a maintainer, I want the CLI isolated from LangGraph continuation input types, so that framework recovery details remain local to the Application Module.
28. As a maintainer, I want checkpoint state to remain the sole persisted execution truth, so that this local foreground application does not acquire two state machines that can drift.
29. As a maintainer, I want unsafe and damaged sessions to fail closed in the first version, so that explicit future repair work is not hidden inside ordinary resume behavior.
30. As a future background-Agent developer, I want this local continuation model not to imply a durable job owner, so that a later supervisor and run registry can be added deliberately rather than reverse-engineered from session state.

## Implementation Decisions

- The session-continuation Module belongs to the Application layer because it
  coordinates persisted graph state, tool policy, session semantics, and valid
  LangGraph invocation. The CLI remains an Adapter responsible for prompts,
  rendering, and collecting user decisions.
- The Module has one external Interface with two capabilities: inspect the
  current continuation condition and execute a typed continuation request.
  Classification helpers may exist inside its Implementation but are not
  separate caller-facing Seams.
- The persisted LangGraph checkpoint is the execution source of truth.
  `ContinuationStatus` is recalculated and never written to the session catalog
  or checkpoint database.
- No `RunState`, run table, worker record, lease, heartbeat, PID, or supervisor
  is introduced.
- Inspection returns a semantic result rather than a raw `StateSnapshot`.
  Callers receive only the information required to render the state and submit
  an allowed action.
- Inspection records the checkpoint identity it observed. Execution compares
  that identity with the latest checkpoint and rejects a stale request before
  invoking the graph.
- The classifier examines checkpoint existence, scheduled next nodes, tasks and
  task errors, top-level and task-level interrupts, message history, tool-call
  IDs, tool-result IDs, and tool policy metadata.
- Tool calls and tool results are matched by tool-call identity, not by count or
  position alone. Multiple tool calls and partial completion are supported.
- A tool call represented by an active HITL interrupt is considered not yet
  authorized and is classified as waiting for human input rather than outcome
  unknown.
- For pending tool work not protected by an active pre-execution interrupt,
  automatic continuation is safe only when every pending tool has known policy,
  is idempotent, and has no side effect.
- A missing tool policy fails closed and contributes to `OUTCOME_UNKNOWN`.
- Protocol and checkpoint structure are prerequisites for every executable
  continuation action. A structurally invalid state is `NEEDS_REPAIR`, even
  when an unresolved tool call may also have an unknown external outcome; that
  uncertainty remains visible in the inspection diagnostics.
- `NEEDS_REPAIR` is also used when an unresolved tool call has no pending tool
  execution path, including malformed history whose only pending node is the
  model.
- `EMPTY` means no checkpoint exists. It allows only `START_TURN`.
- `READY` requires an empty next-node tuple, no pending interrupts, no unresolved
  tool calls, and protocol-valid message history. It allows only `START_TURN`.
- `WAITING_HUMAN` requires a persisted interrupt after excluding any independent
  unknown-outcome operation. It allows only `ANSWER_INTERRUPT`.
- `RESUMABLE` requires pending graph work, no unresolved interrupt, no unknown
  external outcome, and a normal LangGraph continuation path. It allows only
  `CONTINUE`.
- `OUTCOME_UNKNOWN` permits no graph execution in this version. The CLI explains
  which tool calls are uncertain and tells the user that external state must be
  verified before any later repair or risk-bearing retry.
- `NEEDS_REPAIR` permits no graph execution in this version. The CLI explains
  the protocol or checkpoint inconsistency without modifying history.
- `START_TURN` sends a new human message through the same Agent and thread
  configuration already used by normal CLI conversation.
- `ANSWER_INTERRUPT` sends the collected decision through LangGraph's resume
  command with the same thread configuration. Re-presenting the interrupt is UI
  work; approving, rejecting, or editing it is always a user action.
- `CONTINUE` invokes the graph with `None` and the same thread configuration.
  It never manufactures a new human message.
- Session selection performs inspection but does not invoke the graph and does
  not update last-used metadata.
- Session last-used metadata is updated only after an accepted action starts
  real execution.
- Ordinary CLI input is routed through the continuation Interface. It is never
  sent directly to the Agent based solely on the presence of an active session.
- Existing HITL prompting remains a CLI responsibility, but the CLI submits the
  resulting decision to the Application Interface instead of directly invoking
  the Agent.
- The tool policy registry is injected into the continuation Module during
  application composition. It is not exposed to the CLI.
- Unexpected checkpoint read failures and programming defects remain errors;
  they are not normalized into a continuation status. The statuses describe
  validly inspected domain conditions, not arbitrary infrastructure failures.
- The feature does not automatically inspect or reconcile Git, filesystem, or
  remote-system state. It preserves existing work and reports uncertainty
  truthfully.
- The first version does not provide a generic abandon operation. Rejecting an
  HITL tool request, terminating unresolved tool calls with synthetic errors,
  and forking from a prior checkpoint have materially different semantics and
  must not be hidden behind one ambiguous command.
- The existing package namespace and public import compatibility remain
  unchanged.

## Testing Decisions

- The primary test Seam is the Application session-continuation Interface. Tests
  submit observable checkpoint situations and assert the returned status,
  allowed actions, and externally visible graph invocation. They do not test
  private classifier helpers directly.
- Tests use injected fake Agent/state adapters and the existing tool policy
  registry model. They do not require a real model provider or external MCP
  server.
- An empty session returns `EMPTY`, allows `START_TURN`, and rejects continuation
  or HITL answers.
- A terminal valid checkpoint returns `READY`, allows `START_TURN`, and rejects
  continuation or HITL answers.
- A checkpoint with a persisted HITL interrupt returns `WAITING_HUMAN`, preserves
  the original action request, and permits only `ANSWER_INTERRUPT`.
- Answering an interrupt invokes the graph with the expected resume command and
  the original thread configuration.
- A pending non-tool node with no blocker returns `RESUMABLE` and permits only
  `CONTINUE`.
- Pending tools that are all known, idempotent, and side-effect free return
  `RESUMABLE`; continuing invokes the graph with `None`.
- A structurally valid pending tool that is side-effecting, non-idempotent, or
  unclassified returns `OUTCOME_UNKNOWN` and does not invoke the graph.
- A mixed pending tool batch containing both safe and unsafe tools returns
  `OUTCOME_UNKNOWN`.
- A tool call gated by the currently persisted HITL interrupt returns
  `WAITING_HUMAN`, not `OUTCOME_UNKNOWN`.
- An unanswered tool call with no pending tool task, no interrupt, and no normal
  continuation path returns `NEEDS_REPAIR`.
- An unanswered tool call followed by a non-tool message whose only pending node
  is the model returns `NEEDS_REPAIR`; it is not offered tool-call recovery.
- A structurally invalid history with a pending unsafe tool returns
  `NEEDS_REPAIR`, exposes the possible unknown external outcome in diagnostics,
  and permits no continuation action.
- Tool calls and results are paired by ID in tests that vary ordering, include
  several calls in one AI message, and preserve only part of a parallel batch.
- A normal new human message is rejected in every status except `EMPTY` and
  `READY`.
- A stale checkpoint identity causes execution to fail before the Agent is
  invoked and returns an error that instructs the caller to inspect again.
- Selecting a session and inspecting it does not touch last-used metadata.
  Starting, continuing, or answering an interrupt touches it exactly once after
  the request is accepted.
- CLI-level tests are deliberately thin: they verify that resume renders the
  semantic status, ordinary input is blocked when the Interface disallows it,
  continuation is routed through the Interface, and HITL decisions are collected
  and submitted. They do not duplicate the Application classification matrix.
- Existing Application-runtime tests and HITL tests provide prior art for
  injected runtime construction and interrupt decision collection. Existing
  retry tests provide prior art for tool policy combinations and safe versus
  unsafe replay behavior.
- The full existing test suite must continue to pass. Verification also includes
  bytecode compilation, editable-package/import behavior where already covered,
  CLI help/startup smoke checks, and whitespace validation.
- Tests must use the repository's configured Agent Python environment. Tooling
  that is not installed must not be reported as passing.

## Out of Scope

- Persisted run or job lifecycle state.
- Automatic worker crash recovery.
- Background Agent execution, supervisors, worker leases, and heartbeats.
- Exactly-once guarantees for filesystem, shell, MCP, network, or other external
  effects.
- A generic reconciliation Interface for external tools.
- Automatic replay of side-effecting, non-idempotent, or unclassified tools.
- Automatically generated success, failure, or unknown `ToolMessage` repairs.
- Automatic deletion or rewriting of unanswered AI tool calls.
- A generic abandon, rollback, rewind, or checkpoint-fork command.
- Starting a new human turn in a blocked session.
- Automatically creating a replacement session.
- Evaluation datasets, new evaluation scripts, or prompt tuning.
- Unrelated refactoring of the Agent, repository knowledge capability,
  permissions, retry middleware, persistence adapters, or package layout.

## Further Notes

### Acceptance criteria

- Selecting any existing session returns and renders exactly one semantic
  continuation status without executing the Agent.
- Only `EMPTY` and `READY` sessions accept an ordinary human message.
- A persisted HITL interrupt can be answered after process restart using the
  same thread, without adding a new human message.
- A safely replayable pending graph execution can be continued explicitly using
  the same thread and a `None` graph input.
- Unsafe, unclassified, and structurally broken states are blocked before any
  Agent invocation and include actionable diagnostics.
- No new persisted state machine or run registry is added.
- The CLI has no direct responsibility for interpreting raw checkpoint fields or
  choosing LangGraph continuation input types.
- Existing behavior for new sessions, completed sessions, permission decisions,
  retries, and tool errors remains compatible.
- Relevant automated tests and repository verification commands pass in the
  configured Agent Python environment.

### Handoff guidance

The implementing Agent should treat this specification as the decision record
for the first safe recovery slice. It should inspect current framework types and
installed versions before depending on field shapes, but it should not reopen
the settled product decisions. If implementation reveals that the framework
cannot distinguish an active pre-tool HITL interrupt from an already-started
unsafe tool call, fail closed as `OUTCOME_UNKNOWN` and report the concrete
evidence rather than weakening replay safety.

Supporting research already exists in the repository under the titles
"Production coding-agent interruption and recovery" and "LangGraph
interrupted-run recovery". Those notes contain official-source links and the
reasoning behind the local-versus-cloud recovery distinction.

This specification is published as GitHub issue #1 with the `ready-for-agent`
triage label. Do not split it into child tickets unless the implementation later
proves too large for one coherent change.
