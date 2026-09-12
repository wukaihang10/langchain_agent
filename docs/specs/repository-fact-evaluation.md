# Repository-Fact Evaluation Baseline

Status: the original `v1` policy baseline completed on 2026-09-10, the renamed
`v0` policy baseline completed on 2026-09-11, and the clean automated `v0`
baseline using the complete evaluator bundle completed on 2026-09-12. The next
step is a repeated-run stability Experiment before candidate comparisons.

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
evals/fixtures/repository_fact_v0/
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
evals/datasets/repository_fact_v0.jsonl
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

The invocation `thread_id` is a LangGraph checkpoint and continuation key. It is
placed under `configurable`, not duplicated in user-authored trace metadata.
LangSmith may still expose framework-provided thread information on the Run.

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
  "tags": ["langchain-agent", "evaluation", "repository_fact"],
  "metadata": {
    "agent_version": "<configured Agent version>",
    "fixture_version": "repository-fact-v0",
    "permission_mode": "read_only"
  }
}
```

The trace metadata describes how the Agent ran. Example metadata describes what
the case tests. The trace therefore does not repeat `case_id`, `case_type`, or
`slice`, and it does not include the uninformative temporary repository basename.
`fixture_version` is metadata rather than a duplicate tag.

The observability fields have separate jobs:

- `run_name` identifies the operation being traced and remains stable across
  implementation versions;
- tags are a small, low-cardinality filtering vocabulary for the application,
  evaluation context, and capability slice; and
- metadata carries exact comparison dimensions such as Agent, fixture, and
  permission versions or modes.

`agent_name` names the compiled production Agent, while `agent_version` names
the behavior/configuration revision being evaluated. A version must not be used
as a substitute for every other artifact version: the Agent, Dataset, fixture,
and evaluator evolve independently.

### Trace evidence projection

`project_tool_evidence` in `evals/evaluators/trace_evidence.py` is a pure
structural adapter from a loaded LangSmith Run tree to the stable evaluator
shape:

```json
{
  "status": "available",
  "evidence": [
    {
      "tool_name": "read_file",
      "inputs": {"file_path": "src/harbor_tasks/config.py"},
      "output": "DEFAULT_MAX_ATTEMPTS = 4",
      "error": null
    }
  ]
}
```

The projection recursively visits nested children, selects `run_type="tool"`
without hard-coding a repository-tool allowlist, and orders siblings by
`dotted_order` with a deterministic time-and-ID fallback. It unwraps the common
`{"output": ...}` tool result, renders structured values as stable JSON text,
and keeps tool errors. It does not copy model messages, middleware state, Run
IDs, timing, token use, or case metadata into the judge context.

The adapter accepts both Run representations used by LangSmith: live
`aevaluate()` callbacks provide a `RunTree` with attached `child_runs`, while
historical read APIs provide `schemas.Run` values that may additionally declare
`child_run_ids`. Missing-declared-child detection applies only when that second
field exists; a live `RunTree` is traversed through its attached children.

Projection completeness has three states:

| Status | Meaning |
| --- | --- |
| `available` | Nested Runs were loaded and the projection retained all evidence. |
| `partial` | Declared descendants were missing, duplicate/cyclic nodes were detected, or content exceeded an internal projection budget. |
| `unavailable` | The root Trace is absent or its root children were not loaded. |

An available empty evidence list means the observed execution obtained no tool
evidence. An unavailable Trace means the evaluator cannot know whether evidence
was obtained. A partial projection exposes only an incomplete observation and
must not be treated as complete merely because some evidence is present.

The adapter limits each output or error to 12,000 characters and the complete
evidence payload to 60,000 characters. Truncation is marked in the retained text
and changes the projection status to `partial`. These are internal prompt-safety
budgets, not Agent behavior or Experiment comparison dimensions.

## Evaluator interface and lifecycle

The evaluation harness uses LangSmith row-level custom evaluators directly. It
does not introduce a general-purpose evaluator framework. An evaluator declares
only the framework arguments it needs:

- `outputs` for normalized Target output;
- `reference_outputs` for Dataset-owned reference truth;
- `run` when actual child runs or tool evidence are required; and
- `example` only when example metadata such as `case_id` or `case_type` is
  required for diagnostics.

Parameter names are part of the LangSmith evaluator interface. Reference output
and example metadata flow directly from the Example to evaluators and never
through the Target.

For a new experiment, each row follows this sequence:

```text
Target completes and forms the root Run tree
-> row evaluators receive that Run and its Example
-> feedback is attached to the evaluated root Run
```

Trace-dependent evaluators recursively traverse `run.child_runs`; they must not
assume that repository tool runs are direct children of the root. When adding a
trace-dependent evaluator to an existing experiment under the currently pinned
LangSmith behavior, nested runs must be loaded explicitly with
`load_nested=True`.

Each gating result uses an explicit LangSmith feedback shape:

```json
{
  "key": "policy_compliance",
  "value": "pass",
  "comment": "Git audit was available and found no edited files."
}
```

The stable fields are:

- `key`: the feedback identity used for comparison and filtering;
- `value`: the explicit verdict `pass`, `fail`, or `unknown`; and
- `comment`: a concise reason grounded in the evaluated data.

Gating feedback is categorical rather than switching one feedback key between a
boolean score and a categorical unknown. A release or regression gate passes
only an explicit `pass`; `unknown` is non-passing without being mislabeled as a
confirmed behavior failure. A later summary evaluator may calculate numeric pass
rates across the experiment without changing the row-level verdict contract.

Malformed evaluator inputs and implementation defects raise an evaluator error.
They are not converted to a business `fail` or `unknown`. Expected absence of
evidence, such as an unavailable Git audit or missing nested Trace data, produces
an explicit business `unknown` with a comment. Target execution errors are kept
with experiment `error_handling="log"` so they remain diagnosable rather than
being omitted.

## Evaluation semantics

The first baseline keeps independent feedback keys:

- `answer_correctness`;
- `evidence_groundedness`;
- `policy_compliance`;
- `task_success`;
- optional diagnostic `retrieval_sufficiency`;
- optional categorical `failure_stage`.

The feedback contracts are:

| Feedback key | Initial owner | Required inputs | Gating |
| --- | --- | --- | --- |
| `answer_correctness` | Human reviewer, later a calibrated semantic evaluator | Target answer and reference outputs | Yes |
| `evidence_groundedness` | Human reviewer, later a calibrated trace-aware semantic evaluator | Target answer and recursively collected tool evidence | Yes |
| `policy_compliance` | Deterministic code | Git audit status and edited files from Target output | Yes |
| `task_success` | Deterministic three-valued rule over the three gating verdicts | The three gating verdicts | Yes |
| `retrieval_sufficiency` | Optional diagnostic | Child tool runs and acceptable evidence | No |
| `failure_stage` | Human diagnosis initially | Run error and the other verdicts | No |

`answer_correctness` is `pass` only when the response conveys every required
fact and does not endorse any forbidden claim. Semantic equivalents are valid;
exact wording, ordering, formatting, and verbosity are not graded. A forbidden
claim quoted only to reject or correct it is not an endorsement. Missing a
required fact, contradicting the reference, accepting an incorrect premise, or
claiming an unsupported missing capability is `fail`.

`evidence_groundedness` is `pass` only when every material repository claim in
the answer is supported by evidence actually obtained during that same run. A
correct answer produced without repository evidence is therefore not grounded.
Fabricated paths, symbols, quotations, or source claims are `fail`, as is an
answer that conflicts with the evidence it retrieved. Evidence equivalent to,
but located outside, `acceptable_evidence` may still pass after review because
the Dataset does not prescribe one exact retrieval trajectory. Missing or
incomplete nested Trace data is `unknown`, not proof of fabrication.

Groundedness judges whether the retrieved evidence is sufficient for the
material conclusion, not whether the Agent literally opened every repository
file. An unopened file does not by itself cause `fail` when the relevant
execution paths were inspected, the material claims are supported, the omitted
content is not reasonably capable of changing the conclusion, and external
uncertainty is stated. A claim about the specific contents of an unread source,
or a conclusion that depends on that source, still fails. General model
knowledge about what a file such as `pyproject.toml` usually contains is not a
substitute for run evidence; it is the source's demonstrated irrelevance to the
material conclusion that permits a pass.

`policy_compliance` follows this truth table:

| Git audit status | Edited files | Verdict |
| --- | --- | --- |
| `available` | empty | `pass` |
| `available` | non-empty | `fail` |
| `unavailable` | any value | `unknown` |

An unavailable audit is never interpreted as a clean repository. Malformed or
missing Target audit fields violate the Target interface and raise an evaluator
error instead of producing a policy verdict.

`task_success` uses three-valued conjunction:

```text
if any gating verdict is fail:
    task_success = fail
else if every gating verdict is pass:
    task_success = pass
else:
    task_success = unknown
```

This preserves uncertainty while remaining fail-closed for gates. Confirmed
repository-evidence fabrication and policy violations are hard per-example
failures; they cannot be offset by correctness, helpfulness, cost, or aggregate
scores. Evidence fabrication is recorded as the reason for a failed
`evidence_groundedness` verdict rather than creating another top-level metric.

`retrieval_sufficiency` and `failure_stage` remain diagnostic in the first
baseline. They must not affect `task_success`. The first experiment uses manual
Trace review to learn whether the current `acceptable_evidence` shape supports a
reliable deterministic retrieval diagnostic before one is automated.

Specific tool names, tool ordering, call count, latency, token use, and cost are
diagnostic signals unless a later capability contract makes one of them a
business or safety invariant.

## Evaluator composition and calibration

LangSmith invokes row evaluators independently; a later evaluator cannot rely on
the execution order to read feedback emitted by an earlier evaluator. A weighted
Composite evaluator also does not represent the hard logical-AND semantics of
`task_success`.

The implementation follows two stages:

1. The first formal experiment runs the deterministic policy evaluator and is
   manually reviewed for answer correctness, evidence groundedness, task success,
   and failure stage. These eight reviewed runs become calibration examples.
2. The trace-aware semantic evaluator computes `answer_correctness` and
   `evidence_groundedness` in one judge invocation, reuses the deterministic
   policy rule, applies the three-valued task-success rule in the same evaluator
   invocation, and returns independent semantic and task feedback keys. The
   standalone policy evaluator remains the only writer of `policy_compliance`.

This avoids duplicate judge calls without collapsing distinct feedback keys or
depending on evaluator execution order. The semantic evaluator receives its
judge model as a dependency; it does not construct a global model internally.
Its model and evaluator-bundle version are recorded as experiment metadata.

The first structured-output failure is retried once inside the semantic
evaluator. The retry adds a trusted instruction to emit both required judgment
properties and records `judge_attempt=2` on the judge Trace. Normal semantic
verdicts and provider exceptions are not retried at this layer. A second invalid
result remains an evaluator error; it is never repaired heuristically or mapped
to a business `fail` or `unknown`. Safe parser details from
`raw.invalid_tool_calls` are retained in that final error without copying the
complete model response. This bounded-retry behavior is evaluator bundle
`repository-fact-evaluator-v1`; the rubric and feedback keys remain unchanged
from `v0`.

The evidence projection was verified against all eight v0 root Runs: it retained
48 tool Runs across five tool names while keeping every Trace `available`. It
recursively extracts only the tool name, inputs, outputs, and errors needed for
evaluation. It does not copy Trace data into Target output or make a particular
tool sequence part of task success.

This interface follows the official LangSmith documentation for
[custom evaluators](https://docs.langchain.com/langsmith/code-evaluator-sdk),
[intermediate-step evaluation](https://docs.langchain.com/langsmith/evaluate-on-intermediate-steps),
and [multiple feedback results](https://docs.langchain.com/langsmith/multiple-scores).

## Experiment metadata

Every experiment records enough information to identify the tested system:

- configured Agent version;
- repository Git revision and dirty state;
- model name and relevant configuration;
- fixture version;
- evaluator or rubric version;
- permission mode;
- repetition count; and
- execution environment.

LangSmith owns the Dataset identity and the immutable Dataset version used by an
Experiment. Custom metadata must not reuse the reserved `dataset_version` key
for a human-readable Dataset name because the SDK may replace it with the actual
LangSmith Dataset version. If an additional label becomes necessary, use an
unambiguous key such as `dataset_name` or `dataset_contract_version`.

The SDK also records Git provenance. Additional source hashes are useful only
when they identify the complete behavior-changing artifact; hashing one source
file must not be presented as full Agent reproducibility.

A material change to any of these produces a new experiment rather than
overwriting an earlier baseline.

The next full evaluator Experiment uses
`evaluator_version="repository-fact-evaluator-v0"`. This single bundle version
identifies the deterministic policy rule, semantic rubric and structured schema,
and task-success composition together; separate prompt-version metadata is not
added. The first judge uses the same recorded model configuration as the Agent.
A distinct `judge_model_name` becomes necessary only when the composition root
actually injects a different model.

## Historical v1 baseline

The first formal policy-scored baseline completed on 2026-09-10 before the
subsequent naming cleanup:

- Dataset: `repository_fact_v1`
- Dataset ID: `4a4c9814-c8e4-4b3b-8dea-10d482225151`
- Experiment: `repository-fact-baseline-4938563d`
- Experiment ID: `75db3638-1205-4781-9274-80efd00ada27`
- Agent version: `repository-fact-eval-v1`
- Fixture version: `repository_fact_v1`
- Evaluator version: `policy-compliance-v1`
- Repetitions: 1
- Root runs: 8
- Total nested runs: 536
- Runs with errors: 0
- `policy_compliance`: 8 `pass`, 0 `fail`, 0 `unknown`

The experiment is available in the
[LangSmith comparison view](https://smith.langchain.com/o/d5981144-0eb8-48d9-bbe1-2e1e6ae5763c/datasets/4a4c9814-c8e4-4b3b-8dea-10d482225151/compare?selectedSessions=75db3638-1205-4781-9274-80efd00ada27).

This is an execution and policy baseline, not yet a task-success baseline.
`answer_correctness`, `evidence_groundedness`, `task_success`, and
`failure_stage` still require the planned manual review. The observed traces
contain nested repository-tool runs rather than one fixed trajectory; examples
used `read_file`, `search_code`, `search_repository_knowledge`,
`summarize_repository`, and `list_files` in different combinations.

Dataset synchronization uses stable example UUIDs and distinguishes create,
update, and unchanged examples. Repeating synchronization with unchanged local
data performs no example writes and leaves remote example modification times
unchanged. It does not delete remote examples automatically.

The current source-level names `langchain-agent-v0`, `repository-fact-v0`, and
`policy-compliance-v0` do not retroactively rename this historical Dataset or
Experiment.

## Current v0 policy baseline

The renamed policy-scored baseline completed on 2026-09-11:

- Dataset: `repository-fact-v0`
- Dataset ID: `182c5b98-0c39-4ea2-8b4e-b101b75fde5d`
- Experiment: `repository-fact-baseline-a36ce648`
- Experiment ID: `58613711-f185-4232-bfdc-8c5b0818401c`
- Agent version: `langchain-agent-v0`
- Fixture version: `repository-fact-v0`
- Evaluator version: `policy-compliance-v0`
- Repetitions: 1
- Root runs: 8
- Total runs including roots: 528
- Runs with errors: 0
- `policy_compliance`: 8 `pass`, 0 `fail`, 0 `unknown`

The experiment is available in the
[LangSmith comparison view](https://smith.langchain.com/o/d5981144-0eb8-48d9-bbe1-2e1e6ae5763c/datasets/182c5b98-0c39-4ea2-8b4e-b101b75fde5d/compare?selectedSessions=58613711-f185-4232-bfdc-8c5b0818401c).

LangSmith projected the three Example metadata fields onto each root Run as
`ls_example_case_id`, `ls_example_case_type`, and `ls_example_slice`. This makes
case-level Trace filtering available without adding `case_id` to Target inputs
or manually duplicating Example metadata in the Target trace contract.

The first v0 synchronization attempt created the Dataset but stopped before
creating Examples because the empty-Dataset Example lookup returned HTTP 404.
No Experiment was created by that attempt. The synchronization contract now
treats an Example lookup `LangSmithNotFoundError` as an empty match set after the
Dataset itself has been resolved, then creates the missing stable-ID Examples.

### Manual review result

The eight v0 root Runs were manually reviewed on 2026-09-11. The review was
persisted as Run feedback with review version `repository-fact-manual-v0`; it
did not duplicate or replace the deterministic `policy_compliance` feedback.
The aggregate result is:

| Feedback key | Result |
| --- | --- |
| `answer_correctness` | 8 `pass` |
| `evidence_groundedness` | 8 `pass` |
| `policy_compliance` | 8 `pass` |
| `task_success` | 8 `pass` |
| `failure_stage` | 8 `none` |

Two passing cases retain calibration notes rather than creating additional
top-level metrics:

- `repository_fact_003` described a defensive branch too absolutely as
  unreachable. The imprecision was non-material and did not change the answer's
  conclusion or conflict with the retrieved evidence.
- `repository_fact_007` did not open `pyproject.toml` and overstated that every
  repository file had been read. It still inspected the relevant Python
  execution paths, supported the rollback conclusion, and bounded uncertainty
  about external callers. The inspection-scope wording is non-material, so
  `evidence_groundedness` remains `pass` under the sufficiency rule above.

### Semantic evaluator calibration fixture

The Agent Dataset contains eight real positive Runs, so it cannot by itself
show that a semantic evaluator distinguishes `pass`, `fail`, and `unknown`. A
separate evaluator fixture therefore lives at
`tests/fixtures/evaluators/repository_fact/semantic_v0.jsonl`. It tests the
evaluator rather than invoking or grading the Agent.

Each calibration case mirrors the stable evaluator boundaries:

- `inputs` contains the original question;
- `outputs` contains the Target answer;
- `reference_outputs` contains the required facts, forbidden claims, and
  acceptable evidence;
- `trace` contains an explicit availability status and the already-projected
  tool evidence;
- `expected_feedback` contains the human labels for `answer_correctness` and
  `evidence_groundedness`;
- `metadata` contains only `slice`, `case_type`, and `case_id`; and
- `rationale` explains the expected labels for calibration failures.

The eleven cases cover a missing required fact, an endorsed forbidden claim,
an available Trace with no evidence, fabricated evidence, an unavailable
Trace, a non-material imprecision, an approximate but materially correct source
location, an answer that conflicts with retrieved evidence, and the three
`partial` outcomes: sufficient evidence is `pass`, insufficient evidence is
`unknown`, and directly conflicting evidence is `fail`.

This fixture does not duplicate deterministic judgments. `policy_compliance`
continues to come from the Git audit, and `task_success` continues to be derived
from the three gating verdicts. The `failure_stage` vocabulary remains outside
this first semantic fixture until real Agent failures establish useful stage
categories rather than synthetic guesses.

### Semantic evaluator calibration result

The first local calibration completed on 2026-09-12 with judge tracing disabled
and without uploading automatic feedback:

- synthetic semantic cases: 11 of 11 exact matches;
- manually reviewed v0 root Runs: 8 of 8 exact matches for
  `answer_correctness`, `evidence_groundedness`, and `task_success`; and
- the non-material-imprecision decisions for `repository_fact_003` and
  `repository_fact_007` remained `pass`.

The initial judge attempt returned `unknown` groundedness for the available,
empty-evidence case. The rubric was clarified so that `unknown` is not valid for
a complete Trace: insufficient evidence is `fail`. After that rubric correction,
all eleven synthetic branches matched their human labels. An unavailable Trace
is deterministically forced to `unknown`; a partial Trace preserves the judge's
calibrated `pass`, `fail`, or `unknown` sufficiency decision.

## Automated v0 baseline

The first clean Experiment using the complete evaluator bundle completed on
2026-09-12:

- Dataset: `repository-fact-v0`
- Dataset ID: `182c5b98-0c39-4ea2-8b4e-b101b75fde5d`
- Experiment: `repository-fact-baseline-173a79c7`
- Experiment ID: `930d9486-c9eb-44a9-86ad-0f8419be29fc`
- Git revision: `e06a765`
- Agent version: `langchain-agent-v0`
- Fixture version: `repository-fact-v0`
- Evaluator version: `repository-fact-evaluator-v0`
- Repetitions: 1
- Root Runs: 8
- Root Run errors: 0
- Feedback records: 32
- `answer_correctness`: 8 `pass`
- `evidence_groundedness`: 8 `pass`
- `policy_compliance`: 8 `pass`
- `task_success`: 8 `pass`

The experiment is available in the
[LangSmith comparison view](https://smith.langchain.com/o/d5981144-0eb8-48d9-bbe1-2e1e6ae5763c/datasets/182c5b98-0c39-4ea2-8b4e-b101b75fde5d/compare?selectedSessions=930d9486-c9eb-44a9-86ad-0f8419be29fc).

The preceding Experiment `repository-fact-baseline-eb90adf1` is not a model
regression or a valid baseline. Its eight root Runs completed, and its
deterministic policy feedback passed, but all semantic feedback was null because
the Trace adapter accessed `child_run_ids` on the live `RunTree` representation.
The adapter now supports both live `RunTree` values and historical
`schemas.Run` values, with a real-`RunTree` regression test.

The Experiment entry point accepts `--repetitions N`, defaults to one, and
records the selected value in Experiment metadata. Repetition changes sampling
only; it does not create a new Dataset or change the evaluator contract.

The first three-repetition stability attempt,
`repository-fact-baseline-91245b45`, ran 24 target rows from dirty revision
`e06a765-dirty`. All target Runs and deterministic policy evaluations succeeded.
The semantic judge succeeded for 23 rows, while one repetition of
`repository_fact_001` produced an invalid tool-call argument object: the second
judgment object omitted its `evidence_groundedness` property name. That single
judge failure created three null semantic/task feedback records and is evaluator
execution evidence rather than an Agent failure. The bounded structured-output
retry in evaluator `v1` addresses that failure mode; a clean repeated Experiment
is still required.

## Implementation order

1. Manually label answer correctness, evidence groundedness, task success, and
   failure stage for all eight Runs. Complete.
2. Establish positive, negative, and unknown semantic evaluator calibration
   cases without adding them to the Agent Dataset. Complete.
3. Implement and test the recursive evidence projection against real and
   synthetic nested Trace shapes. Complete.
4. Define and calibrate a semantic evaluator only for criteria that
   deterministic code cannot judge reliably. Complete.
5. Run a clean one-repetition Experiment with the complete evaluator bundle.
   Complete.
6. Run a repeated stability Experiment and diagnose any per-case or judge
   variance before changing Agent behavior.
7. Add candidate regression comparison and later CI/online evaluation only
   after the repeated baseline is trustworthy.

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
