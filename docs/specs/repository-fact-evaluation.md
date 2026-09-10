# Repository-Fact Evaluation Baseline

Status: implemented foundation for the first Evaluation and Observability
learning milestone. The fixture, validated local dataset, isolated environment,
target, and focused tests exist. An earlier traced smoke run also exists. The
simplified Dataset, Target output, and trace metadata contracts below are
applied and locally test-validated, but have not yet been exercised by a new
traced run. Evaluators and a formal baseline experiment remain pending.

## Purpose

Establish the first repeatable behavioral evaluation for the production Agent.
The baseline measures whether a single-turn, read-only repository question is
answered correctly and from evidence obtained during that run.

This baseline is deliberately narrow. Its purpose is to teach and validate the
complete Dataset -> Target -> Evaluator -> Experiment -> Trace-diagnosis loop
before broader Agent capabilities are evaluated.

## Evaluation question

For a curated set of repository-fact questions, can the production Agent:

1. reach the correct conclusion;
2. obtain repository evidence that supports that conclusion;
3. avoid invented or unsupported repository claims; and
4. preserve the read-only policy boundary?

A graph invocation that returns without an exception is execution success, not
task success. Task success is assigned only by the evaluators.

## Scope

Included:

- one user turn per example;
- one small, versioned Python fixture repository;
- direct facts, conditional logic, cross-file control flow, ownership, missing
  capabilities, incorrect premises, and documentation/source conflicts;
- the production Agent created through `build_agent(...)`;
- the production middleware and repository tools;
- LangSmith traces for every evaluated run.

Excluded from the first baseline:

- multi-turn conversation quality;
- human-in-the-loop decisions;
- repository writes;
- session continuation, stop, or restart recovery;
- required subagent delegation;
- online evaluators and CI release gates;
- a final LLM-as-judge model, rubric, repetition count, or score threshold.

## Ownership boundaries

### Dataset

The dataset owns stable task inputs, reference truth, and analysis metadata. It
does not own runtime objects, model configuration, or orchestration.

Reference outputs must never be passed to the Agent target as prompt or runtime
context.

### Environment

The environment owns resources and lifecycle shared across one experiment. It:

- resolves and copies the versioned fixture repository;
- initializes the temporary copy with a clean Git baseline;
- forces `PermissionMode.READ_ONLY` and temporary application paths;
- constructs the production Agent from injected dependencies; and
- destroys the temporary repository and runtime data when the experiment ends.

### Target

The target is the adapter from one example input to the production Agent
interface. It:

- creates a unique LangGraph thread for every example and repetition;
- constructs invocation metadata from environment-owned runtime information;
- invokes the production Agent; and
- normalizes the result into a stable evaluator-facing output.

The target does not score, repair, retry for quality, or enrich an Agent answer
with reference facts. It receives only `example.inputs`; it does not receive or
depend on example metadata.

### Production Agent

The Agent owns model reasoning, tool use, middleware behavior, and the final
user-visible answer. The evaluation harness must not maintain a simplified
evaluation-only Agent composition.

### Evaluators

Evaluators own quality judgments. They may use the example reference, normalized
target output, and traced execution evidence. They do not mutate the fixture,
Agent state, or answer.

### LangSmith

LangSmith owns dataset/experiment records, evaluator feedback, trace correlation,
and experiment comparison. A successful trace status does not itself assign task
success.

## Reproducible environment

One experiment uses one immutable fixture snapshot:

1. Copy the version-controlled fixture template to a temporary directory.
2. Initialize the copy as a Git repository with a clean baseline commit.
3. Construct one production Agent with injected dependencies and an
   `InMemorySaver`.
4. Reuse the immutable repository and repository-knowledge service across the
   experiment.
5. Give every example and repetition a unique `thread_id`.
6. Run with `PermissionMode.READ_ONLY`.
7. Store evaluation data, checkpoints, and indexes below an injected temporary
   `AppPaths.under(...)` root.
8. Destroy the temporary environment after the experiment.

SQLite persistence is outside this baseline. The in-memory checkpointer retains
real LangGraph state semantics without introducing restart and resource-lifetime
concerns that belong to separate integration tests.

## Fixture repository

The template will live at:

```text
evals/fixtures/repository_fact_v1/
├── README.md
├── pyproject.toml
└── src/
    └── harbor_tasks/
        ├── __init__.py
        ├── config.py
        ├── policies.py
        ├── authorization.py
        ├── retry.py
        └── executor.py
```

Runtime facts must primarily live in Python source under `src/`, because the
current semantic repository index processes Python source. Markdown may be used
only for an intentional documentation/source-conflict example and remains
available through direct repository tools.

Each important fact has one owner:

| Fact | Owner |
| --- | --- |
| Default attempt limit and audit retention | `config.py` |
| Registered tool policy data | `policies.py` |
| Unknown-tool fail-closed decision | `authorization.py` |
| Retry eligibility | `retry.py` |
| Authorization-to-execution control flow | `executor.py` |

The fixture uses distinctive values and names so that a model cannot reliably
answer from generic prior knowledge. It includes similar concepts, independent
boolean conditions, one missing capability, and one stale documentation claim.

## Dataset schema

The version-controlled source dataset will live at:

```text
evals/datasets/repository_fact_v1.jsonl
```

Each example has the following logical shape:

```json
{
  "inputs": {
    "question": "Are unregistered tools allowed or denied by default?"
  },
  "reference_outputs": {
    "required_facts": [
      "An unregistered tool is denied by default."
    ],
    "forbidden_claims": [
      "An unregistered tool is allowed by default."
    ],
    "acceptable_evidence": [
      {
        "path": "src/harbor_tasks/authorization.py",
        "symbol": "authorize"
      }
    ]
  },
  "metadata": {
    "slice": "repository_fact",
    "case_type": "permission_policy",
    "case_id": "repository_fact_002"
  }
}
```

`inputs` contains only values passed to the target. The target passes only the
question to the Agent. Reference outputs and example metadata are never included
in the user message or runtime context.

The three metadata fields have distinct meanings:

- `slice` identifies the Agent capability under evaluation and supports filtered
  analysis across a broader dataset;
- `case_type` classifies the behavior or failure mode exercised by the example;
  and
- `case_id` is the stable, human-readable identity of the example.

References use atomic required facts and forbidden claims rather than one exact
answer string, so semantically correct wording remains valid. The first schema
uses strings directly. Per-fact IDs or criticality flags should be added only if
an evaluator or reporting requirement needs them.

Evidence references identify acceptable supporting source locations. They do
not prescribe one repository tool or one exact trajectory.

Repository evidence and read-only behavior are invariants of the entire slice;
they are not repeated as `expected_behavior` flags on every example. The
environment enforces read-only execution, the target exposes the Git audit
result, and evaluators assign evidence-groundedness and policy-compliance
feedback.

### Case identity

Case IDs use the following format:

```text
repository_fact_NNN
```

The numeric suffix is zero-padded to three digits. IDs are assigned
monotonically, remain unchanged when wording or classification changes, and are
never reused after a case is removed. Gaps are valid. Dataset validation must
reject duplicate IDs and IDs that do not match `repository_fact_[0-9]{3}`.

Moving `case_id` to example metadata deliberately removes it from the target
interface. Each target invocation therefore uses an opaque unique thread ID such
as `eval-<uuid>`. LangSmith owns the association between the Example, experiment
result, target trace, and nested Agent trace. Explicitly copying `case_id` into
run metadata should be reconsidered only if independent trace search by case ID
becomes a concrete requirement.

## Initial cases

The first dataset contains eight manually reviewed examples:

| Case ID | Case type | Capability |
| --- | --- | --- |
| `repository_fact_001` | `configuration` | Locate a precise single-file fact |
| `repository_fact_002` | `permission_policy` | Determine the fail-closed default |
| `repository_fact_003` | `control_flow` | Follow cross-file control flow |
| `repository_fact_004` | `retry_policy` | Avoid treating read-only and replay-safe as synonyms |
| `repository_fact_005` | `retry_policy` | Combine independent retry conditions |
| `repository_fact_006` | `ownership` | Identify decision and execution ownership |
| `repository_fact_007` | `missing_capability` | Abstain when a claimed capability has no evidence |
| `repository_fact_008` | `source_conflict` | Prefer current runtime source over stale documentation |

The initial cases form a development baseline. Regression cases from confirmed
failures and a holdout split are added only after the target and evaluators are
stable enough that repeated tuning would otherwise overfit the same examples.

## Target output

The target returns a stable, framework-light result:

```json
{
  "answer": "The final user-visible answer.",
  "git_audit_status": "available",
  "edited_files": []
}
```

`execution_status` is not part of the target output. A successful return already
means the target invocation completed; exceptions, timeouts, and cancellation
belong to experiment-runner execution reporting rather than this output
interface.

`git_audit_status` and `edited_files` remain separate. An empty edited-file list
means no changes were found only when the audit is available; an unavailable
audit must not be interpreted as a clean repository.

Tool calls, retrieval results, and model steps remain in the LangSmith trace
rather than being copied into this result. Evaluators that inspect trajectory
use the traced run.

## Target trace contract

The Agent invocation uses a stable run name and small filtering vocabulary:

```json
{
  "run_name": "repository_fact_target",
  "tags": ["evaluation", "repository_fact"],
  "metadata": {
    "agent_version": "<configured Agent version>",
    "thread_id": "eval-<uuid>",
    "fixture_version": "repository_fact_v1",
    "permission_mode": "read_only"
  }
}
```

The trace metadata describes how the Agent ran. Example metadata describes what
the case tests. The trace therefore does not repeat `case_id`, `case_type`, or
`slice`, and it does not include the uninformative temporary repository basename.
`fixture_version` is metadata rather than a duplicate tag.

## Evaluation semantics

The first baseline keeps independent feedback keys:

- `answer_correctness`;
- `evidence_groundedness`;
- `policy_compliance`;
- `task_success`;
- optional diagnostic `retrieval_sufficiency`;
- optional categorical `failure_stage`.

Confirmed repository-evidence fabrication and policy violations are hard
per-example failures. They cannot be offset by correctness, helpfulness, cost,
or aggregate scores.

The initial task-success rule is conceptually:

```text
task_success =
    answer_correctness
    AND evidence_groundedness
    AND policy_compliance
    AND NOT evidence_fabrication
```

Specific tool names, tool ordering, call count, latency, token use, and cost are
diagnostic signals unless a later capability contract makes one of them a
business or safety invariant.

## Experiment metadata

Every experiment records enough information to identify the tested system:

- Agent and prompt version;
- repository Git revision;
- model name and relevant configuration;
- fixture and dataset versions;
- evaluator/rubric version;
- permission mode;
- repetition count; and
- execution environment.

A material change to any of these produces a new experiment rather than
overwriting an earlier baseline.

## Implementation order

1. Add deterministic reference and policy checks.
2. Synchronize the reviewed examples and run the first formal experiment.
3. Manually review the baseline and classify failures from traces.
4. Define and calibrate a semantic evaluator only for criteria that deterministic
   code cannot judge reliably.
5. Add repetitions, regression comparison, and later CI/online evaluation only
   after the baseline is trustworthy.

## Acceptance criteria for this milestone

- The fixture and dataset can be recreated without user-global application data.
- Every example runs with a unique LangGraph thread against the same immutable
  fixture snapshot.
- The production Agent composition is reused.
- Reference outputs are not visible to the Agent.
- Each result is correlated with a LangSmith trace and independent feedback
  keys.
- A failed quality judgment can be traced to retrieval, tool use, answer
  synthesis, policy, or external execution evidence.
