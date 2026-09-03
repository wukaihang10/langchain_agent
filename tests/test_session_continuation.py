import unittest
from types import SimpleNamespace

from langchain_core.messages import AIMessage, HumanMessage, ToolMessage
from langgraph.types import Command, Interrupt, StateSnapshot

from langchain_agent.app.session_continuation import (
    ContinuationAction,
    ContinuationRequest,
    ContinuationStatus,
    InvalidContinuationAction,
    SessionContinuation,
    StaleContinuationError,
)
from langchain_agent.app.context import AgentContext
from langchain_agent.harness.permissions.models import (
    ToolCategory,
    ToolPolicy,
    ToolRisk,
)
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry
from langchain_agent.harness.middleware.turn_recovery import (
    RecoveryDirectiveKind,
    TurnRecoveryMode,
)

SAFE_POLICY = ToolPolicy(
    category=ToolCategory.READ,
    idempotent=True,
    side_effect=False,
    risk=ToolRisk.LOW,
)
UNSAFE_POLICY = ToolPolicy(
    category=ToolCategory.WRITE,
    idempotent=False,
    side_effect=True,
    risk=ToolRisk.HIGH,
)
SIDE_EFFECT_POLICY = ToolPolicy(
    category=ToolCategory.WRITE,
    idempotent=True,
    side_effect=True,
    risk=ToolRisk.HIGH,
)
NON_IDEMPOTENT_POLICY = ToolPolicy(
    category=ToolCategory.READ,
    idempotent=False,
    side_effect=False,
    risk=ToolRisk.MEDIUM,
)


def snapshot(
    *,
    checkpoint_id="checkpoint-1",
    messages=(),
    next_nodes=(),
    tasks=(),
    interrupts=(),
):
    configurable = {"thread_id": "thread-1", "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id

    return StateSnapshot(
        values={"messages": list(messages)} if messages else {},
        next=tuple(next_nodes),
        config={"configurable": configurable},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=tuple(tasks),
        interrupts=tuple(interrupts),
    )


def ai_call(*calls):
    return AIMessage(
        content="",
        tool_calls=[
            {
                "name": name,
                "args": args,
                "id": call_id,
                "type": "tool_call",
            }
            for call_id, name, args in calls
        ],
    )


class FakeAgent:
    def __init__(self, state, *, state_after_invoke=None):
        self.state = state
        self.state_after_invoke = state_after_invoke
        self.invocations = []

    async def aget_state(self, config):
        self.state_reads = getattr(self, "state_reads", [])
        self.state_reads.append(config)
        return self.state

    async def ainvoke(self, input_value, *, config, context):
        self.invocations.append((input_value, config, context))
        if self.state_after_invoke is not None:
            self.state = self.state_after_invoke
        return {"messages": [AIMessage(content="done")]}


class FakeSessionStore:
    def __init__(self):
        self.touched = []

    def touch(self, thread_id):
        self.touched.append(thread_id)


def runtime():
    return SimpleNamespace(
        session=SimpleNamespace(thread_id="thread-1"),
        invoke_config={"configurable": {"thread_id": "thread-1"}},
        context=AgentContext(
            repository_path="C:/repository",
            repository_knowledge=object(),
        ),
    )


def service(state, *, policies=None, state_after_invoke=None):
    agent = FakeAgent(state, state_after_invoke=state_after_invoke)
    store = FakeSessionStore()
    continuation = SessionContinuation(
        agent=agent,
        policy_registry=ToolPolicyRegistry(policies),
        session_store=store,
    )
    return continuation, agent, store


class SessionContinuationInspectionTests(unittest.IsolatedAsyncioTestCase):
    async def test_unfinished_sessions_offer_explicit_turn_termination(self):
        cases = (
            (
                snapshot(
                    messages=[ai_call(("call-1", "write", {}))],
                    next_nodes=["model"],
                    interrupts=[
                        Interrupt(
                            value={
                                "action_requests": [
                                    {"name": "write", "args": {}}
                                ],
                                "review_configs": [
                                    {"allowed_decisions": ["approve", "reject"]}
                                ],
                            },
                            id="interrupt-1",
                        )
                    ],
                ),
                {"write": UNSAFE_POLICY},
                {
                    ContinuationAction.ANSWER_INTERRUPT,
                    ContinuationAction.TERMINATE_TURN,
                },
            ),
            (
                snapshot(next_nodes=["model"]),
                None,
                {
                    ContinuationAction.CONTINUE,
                    ContinuationAction.TERMINATE_TURN,
                },
            ),
            (
                snapshot(
                    messages=[ai_call(("call-1", "write", {}))],
                    next_nodes=["tools"],
                ),
                {"write": UNSAFE_POLICY},
                {
                    ContinuationAction.CONTINUE,
                    ContinuationAction.TERMINATE_TURN,
                },
            ),
        )

        for state, policies, expected in cases:
            with self.subTest(expected=expected):
                continuation, _, _ = service(state, policies=policies)
                inspection = await continuation.inspect(runtime())
                self.assertEqual(inspection.allowed_actions, expected)

    async def test_empty_and_ready_sessions_allow_only_start_turn(self):
        continuation, _, _ = service(snapshot(checkpoint_id=None))
        empty = await continuation.inspect(runtime())
        self.assertEqual(empty.status, ContinuationStatus.EMPTY)
        self.assertEqual(
            empty.allowed_actions,
            {ContinuationAction.START_TURN},
        )

        continuation, _, _ = service(snapshot(messages=[HumanMessage(content="hello")]))
        ready = await continuation.inspect(runtime())
        self.assertEqual(ready.status, ContinuationStatus.READY)
        self.assertEqual(
            ready.allowed_actions,
            {ContinuationAction.START_TURN},
        )

    async def test_hitl_interrupt_preserves_request_and_overrides_unsafe_policy(self):
        action = {"name": "write_file", "args": {"path": "a.txt"}}
        interrupt = Interrupt(
            value={
                "action_requests": [action],
                "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
            },
            id="interrupt-1",
        )
        continuation, _, _ = service(
            snapshot(
                messages=[ai_call(("call-1", "write_file", {"path": "a.txt"}))],
                next_nodes=["model"],
                interrupts=[interrupt],
            ),
            policies={"write_file": UNSAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.WAITING_HUMAN)
        self.assertEqual(
            inspection.allowed_actions,
            {
                ContinuationAction.ANSWER_INTERRUPT,
                ContinuationAction.TERMINATE_TURN,
            },
        )
        self.assertEqual(inspection.interrupts[0].value["action_requests"], [action])

    async def test_pending_non_tool_and_safe_tool_work_are_resumable(self):
        continuation, _, _ = service(
            snapshot(
                next_nodes=["model"],
                tasks=[
                    SimpleNamespace(
                        id="task-1",
                        name="model",
                        error=RuntimeError("provider disconnected"),
                        interrupts=(),
                    )
                ],
            )
        )
        non_tool = await continuation.inspect(runtime())
        self.assertEqual(non_tool.status, ContinuationStatus.RESUMABLE)
        self.assertEqual(non_tool.pending_nodes, ("model",))
        self.assertIn("task-1", non_tool.reason)
        self.assertIn("provider disconnected", non_tool.reason)

        continuation, _, _ = service(
            snapshot(
                messages=[ai_call(("call-1", "search", {"q": "agent"}))],
                next_nodes=["tools"],
            ),
            policies={"search": SAFE_POLICY},
        )
        safe_tool = await continuation.inspect(runtime())
        self.assertEqual(safe_tool.status, ContinuationStatus.RESUMABLE)
        self.assertEqual(
            safe_tool.allowed_actions,
            {
                ContinuationAction.CONTINUE,
                ContinuationAction.TERMINATE_TURN,
            },
        )

    async def test_unsafe_non_idempotent_and_unclassified_tools_fail_closed(self):
        cases = [
            ("side_effect", {"side_effect": SIDE_EFFECT_POLICY}),
            (
                "non_idempotent",
                {"non_idempotent": NON_IDEMPOTENT_POLICY},
            ),
            ("unknown", {}),
        ]
        for tool_name, policies in cases:
            with self.subTest(tool_name=tool_name):
                continuation, _, _ = service(
                    snapshot(
                        messages=[ai_call(("call-1", tool_name, {}))],
                        next_nodes=["tools"],
                    ),
                    policies=policies,
                )
                inspection = await continuation.inspect(runtime())
                self.assertEqual(
                    inspection.status,
                    ContinuationStatus.OUTCOME_UNKNOWN,
                )
                self.assertEqual(
                    inspection.allowed_actions,
                    {
                        ContinuationAction.CONTINUE,
                        ContinuationAction.TERMINATE_TURN,
                    },
                )

    async def test_independent_unsafe_call_overrides_hitl_interrupt(self):
        interrupt = Interrupt(
            value={
                "action_requests": [{"name": "reviewed", "args": {}}],
                "review_configs": [{"allowed_decisions": ["approve", "reject"]}],
            },
            id="interrupt-1",
        )
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ai_call(
                        ("reviewed-id", "reviewed", {}),
                        ("independent-id", "write", {}),
                    )
                ],
                next_nodes=["tools"],
                interrupts=[interrupt],
            ),
            policies={
                "reviewed": UNSAFE_POLICY,
                "write": UNSAFE_POLICY,
            },
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.OUTCOME_UNKNOWN)
        self.assertIn("independent-id", inspection.reason)
        self.assertEqual(
            [
                call.id
                for call in inspection.unresolved_tool_calls
                if call.outcome_unknown
            ],
            ["independent-id"],
        )

    async def test_terminal_tool_history_is_ready_and_orphan_result_needs_repair(self):
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ai_call(("call-1", "search", {"q": "x"})),
                    ToolMessage(content="found", tool_call_id="call-1"),
                    AIMessage(content="answer"),
                ]
            ),
            policies={"search": SAFE_POLICY},
        )
        ready = await continuation.inspect(runtime())
        self.assertEqual(ready.status, ContinuationStatus.READY)

        continuation, _, _ = service(
            snapshot(messages=[ToolMessage(content="orphan", tool_call_id="missing")])
        )
        damaged = await continuation.inspect(runtime())
        self.assertEqual(damaged.status, ContinuationStatus.NEEDS_REPAIR)

    async def test_mixed_batch_uses_ids_and_unsafe_call_takes_precedence(self):
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ai_call(
                        ("safe-id", "search", {"q": "x"}),
                        ("unsafe-id", "write", {"value": 1}),
                    ),
                    ToolMessage(content="found", tool_call_id="safe-id"),
                ],
                next_nodes=["tools"],
            ),
            policies={"search": SAFE_POLICY, "write": UNSAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.OUTCOME_UNKNOWN)
        self.assertEqual(
            inspection.allowed_actions,
            {
                ContinuationAction.CONTINUE,
                ContinuationAction.TERMINATE_TURN,
            },
        )
        self.assertEqual(
            [call.id for call in inspection.unresolved_tool_calls],
            ["unsafe-id"],
        )

    async def test_parallel_results_are_paired_by_id_not_position(self):
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ai_call(
                        ("first", "search", {"q": "one"}),
                        ("second", "search", {"q": "two"}),
                    ),
                    ToolMessage(content="two", tool_call_id="second"),
                ],
                next_nodes=["tools"],
            ),
            policies={"search": SAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.RESUMABLE)
        self.assertEqual(
            [call.id for call in inspection.unresolved_tool_calls],
            ["first"],
        )

    async def test_unanswered_call_without_graph_path_needs_repair(self):
        continuation, _, _ = service(
            snapshot(messages=[ai_call(("call-1", "search", {}))]),
            policies={"search": SAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.NEEDS_REPAIR)
        self.assertFalse(inspection.allowed_actions)

        continuation, _, _ = service(
            snapshot(messages=[ai_call(("unsafe-id", "write", {}))]),
            policies={"write": UNSAFE_POLICY},
        )
        unsafe = await continuation.inspect(runtime())
        self.assertEqual(unsafe.status, ContinuationStatus.NEEDS_REPAIR)
        self.assertIn("external outcome may be unknown", unsafe.reason.lower())

    async def test_tool_call_bypassed_by_human_and_pending_model_needs_repair(self):
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ai_call(("unsafe-id", "task", {})),
                    HumanMessage(content="hello"),
                ],
                next_nodes=["model"],
            ),
            policies={"task": UNSAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.NEEDS_REPAIR)
        self.assertFalse(inspection.allowed_actions)
        self.assertIn("structurally inconsistent", inspection.reason.lower())

    async def test_protocol_repair_blocks_unknown_outcome_continuation(self):
        continuation, _, _ = service(
            snapshot(
                messages=[
                    ToolMessage(content="orphan", tool_call_id="missing"),
                    ai_call(("unsafe-id", "write", {})),
                ],
                next_nodes=["tools"],
            ),
            policies={"write": UNSAFE_POLICY},
        )

        inspection = await continuation.inspect(runtime())

        self.assertEqual(inspection.status, ContinuationStatus.NEEDS_REPAIR)
        self.assertFalse(inspection.allowed_actions)
        self.assertIn("external outcome may be unknown", inspection.reason.lower())


class SessionContinuationExecutionTests(unittest.IsolatedAsyncioTestCase):
    async def test_waiting_termination_rejects_all_and_cancels_unprotected(
        self,
    ):
        calls = ai_call(
            ("write-1", "write", {"value": 1}),
            ("write-2", "write", {"value": 2}),
            ("safe-1", "search", {"q": "agent"}),
        )
        waiting = snapshot(
            messages=[calls],
            next_nodes=["tools"],
            interrupts=[
                Interrupt(
                    value={
                        "action_requests": [
                            {"name": "write", "args": {"value": 1}},
                            {"name": "write", "args": {"value": 2}},
                        ],
                        "review_configs": [
                            {"allowed_decisions": ["approve", "reject"]},
                            {"allowed_decisions": ["approve", "reject"]},
                        ],
                    },
                    id="interrupt-1",
                )
            ],
        )
        continuation, agent, _ = service(
            waiting,
            policies={"write": UNSAFE_POLICY, "search": SAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.TERMINATE_TURN,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        input_value, _, invocation_context = agent.invocations[0]
        self.assertIsInstance(input_value, Command)
        self.assertEqual(
            input_value.resume,
            {
                "decisions": [
                    {"type": "reject", "message": "User terminated the turn."},
                    {"type": "reject", "message": "User terminated the turn."},
                ]
            },
        )
        self.assertEqual(
            set(invocation_context.turn_recovery.directives),
            {"safe-1"},
        )
        self.assertEqual(
            invocation_context.turn_recovery.directives[
                "safe-1"
            ].kind,
            RecoveryDirectiveKind.CANCELLED_BY_TERMINATION,
        )

    async def test_outcome_unknown_termination_records_each_unprotected_risk(self):
        pending = snapshot(
            messages=[
                ai_call(
                    ("safe-id", "search", {"q": "agent"}),
                    ("unsafe-id", "write", {"value": 1}),
                    ("unknown-id", "unregistered", {}),
                )
            ],
            next_nodes=["tools"],
        )
        continuation, agent, _ = service(
            pending,
            policies={"search": SAFE_POLICY, "write": UNSAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.TERMINATE_TURN,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        directives = agent.invocations[0][2].turn_recovery.directives
        self.assertEqual(
            {
                call_id: directive.kind
                for call_id, directive in directives.items()
            },
            {
                "safe-id": RecoveryDirectiveKind.CANCELLED_BY_TERMINATION,
                "unsafe-id": RecoveryDirectiveKind.OUTCOME_UNKNOWN_AT_TERMINATION,
                "unknown-id": RecoveryDirectiveKind.OUTCOME_UNKNOWN_AT_TERMINATION,
            },
        )

    async def test_outcome_unknown_termination_also_rejects_embedded_hitl(self):
        pending = snapshot(
            messages=[
                ai_call(
                    ("protected-id", "write", {"value": 1}),
                    ("independent-id", "create", {"value": 2}),
                )
            ],
            next_nodes=["tools"],
            interrupts=[
                Interrupt(
                    value={
                        "action_requests": [
                            {"name": "write", "args": {"value": 1}}
                        ],
                        "review_configs": [
                            {"allowed_decisions": ["approve", "reject"]}
                        ],
                    },
                    id="interrupt-1",
                )
            ],
        )
        continuation, agent, _ = service(
            pending,
            policies={"write": UNSAFE_POLICY, "create": UNSAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )
        inspection = await continuation.inspect(runtime())
        self.assertEqual(inspection.status, ContinuationStatus.OUTCOME_UNKNOWN)

        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.TERMINATE_TURN,
                observed_checkpoint_id=inspection.checkpoint_id,
            ),
        )

        input_value, _, invocation_context = agent.invocations[0]
        self.assertIsInstance(input_value, Command)
        self.assertEqual(
            input_value.resume,
            {
                "decisions": [
                    {"type": "reject", "message": "User terminated the turn."}
                ]
            },
        )
        self.assertEqual(
            set(invocation_context.turn_recovery.directives),
            {"independent-id"},
        )

    async def test_termination_payload_and_stale_checkpoint_are_rejected_without_work(
        self,
    ):
        pending = snapshot(next_nodes=["model"])
        invalid_payloads = (
            {"message": "new work"},
            {"decisions": ({"type": "reject"},)},
        )

        for payload in invalid_payloads:
            with self.subTest(payload=payload):
                continuation, agent, store = service(pending)
                with self.assertRaisesRegex(
                    InvalidContinuationAction,
                    "does not accept",
                ):
                    await continuation.execute(
                        runtime(),
                        ContinuationRequest(
                            action=ContinuationAction.TERMINATE_TURN,
                            observed_checkpoint_id="checkpoint-1",
                            **payload,
                        ),
                    )
                self.assertFalse(store.touched)
                self.assertFalse(agent.invocations)

        continuation, agent, store = service(
            snapshot(checkpoint_id="checkpoint-2", next_nodes=["model"])
        )
        with self.assertRaises(StaleContinuationError):
            await continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.TERMINATE_TURN,
                    observed_checkpoint_id="checkpoint-1",
                ),
            )
        self.assertFalse(store.touched)
        self.assertFalse(agent.invocations)

    async def test_resumable_termination_builds_cancellation_plan_without_new_input(
        self,
    ):
        pending = snapshot(
            messages=[ai_call(("safe-id", "search", {"q": "agent"}))],
            next_nodes=["tools"],
        )
        continuation, agent, store = service(
            pending,
            policies={"search": SAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        result = await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.TERMINATE_TURN,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        input_value, _, invocation_context = agent.invocations[0]
        self.assertIsNone(input_value)
        self.assertEqual(
            invocation_context.turn_recovery.mode,
            TurnRecoveryMode.TERMINATE,
        )
        directive = invocation_context.turn_recovery.directives["safe-id"]
        self.assertEqual(
            directive.kind,
            RecoveryDirectiveKind.CANCELLED_BY_TERMINATION,
        )
        self.assertEqual(store.touched, ["thread-1"])
        self.assertEqual(result.inspection.status, ContinuationStatus.READY)

    async def test_termination_reports_pending_state_instead_of_claiming_success(self):
        continuation, _, _ = service(
            snapshot(next_nodes=["model"]),
            state_after_invoke=snapshot(
                checkpoint_id="checkpoint-2",
                next_nodes=["model"],
            ),
        )

        result = await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.TERMINATE_TURN,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        self.assertEqual(result.inspection.status, ContinuationStatus.RESUMABLE)
        self.assertIn(
            ContinuationAction.TERMINATE_TURN,
            result.inspection.allowed_actions,
        )

    async def test_unknown_outcome_is_automatically_translated_to_recovery_plan(self):
        pending = snapshot(
            messages=[ai_call(("write-id", "write", {"value": 1}))],
            next_nodes=["tools"],
        )
        continuation, agent, store = service(
            pending,
            policies={"write": UNSAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.CONTINUE,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        invocation_context = agent.invocations[0][2]
        self.assertEqual(
            invocation_context.turn_recovery.mode,
            TurnRecoveryMode.CONTINUE,
        )
        directive = invocation_context.turn_recovery.directives["write-id"]
        self.assertEqual(directive.kind, RecoveryDirectiveKind.OUTCOME_UNKNOWN)
        self.assertEqual(store.touched, ["thread-1"])

    async def test_outcome_unknown_plan_covers_only_replay_unsafe_calls(self):
        pending = snapshot(
            messages=[
                ai_call(
                    ("unsafe-1", "write", {"value": 1}),
                    ("unsafe-2", "write", {"value": 2}),
                    ("safe-1", "search", {"q": "x"}),
                )
            ],
            next_nodes=["tools"],
        )
        continuation, agent, store = service(
            pending,
            policies={"write": UNSAFE_POLICY, "search": SAFE_POLICY},
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.CONTINUE,
                observed_checkpoint_id="checkpoint-1",
            ),
        )

        plan = agent.invocations[0][2].turn_recovery
        self.assertEqual(
            set(plan.directives),
            {"unsafe-1", "unsafe-2"},
        )
        self.assertTrue(
            all(
                directive.kind == RecoveryDirectiveKind.OUTCOME_UNKNOWN
                for directive in plan.directives.values()
            )
        )
        self.assertEqual(store.touched, ["thread-1"])

    async def test_stale_recovery_plan_is_rejected_before_plan_validation(self):
        continuation, agent, store = service(
            snapshot(
                checkpoint_id="checkpoint-2",
                messages=[ai_call(("unsafe-1", "write", {}))],
                next_nodes=["tools"],
            ),
            policies={"write": UNSAFE_POLICY},
        )

        with self.assertRaises(StaleContinuationError):
            await continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.CONTINUE,
                    observed_checkpoint_id="checkpoint-1",
                ),
            )

        self.assertFalse(store.touched)
        self.assertFalse(agent.invocations)

    async def test_start_turn_validates_then_touches_and_invokes_human_input(self):
        continuation, agent, store = service(
            snapshot(checkpoint_id=None),
            state_after_invoke=snapshot(checkpoint_id="checkpoint-2"),
        )

        result = await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.START_TURN,
                observed_checkpoint_id=None,
                message="hello",
            ),
        )

        self.assertEqual(store.touched, ["thread-1"])
        self.assertEqual(
            agent.invocations[0][0],
            {"messages": [{"role": "user", "content": "hello"}]},
        )
        self.assertEqual(result.inspection.status, ContinuationStatus.READY)

    async def test_continue_uses_none_and_answer_uses_langgraph_command(self):
        pending = snapshot(next_nodes=["model"])
        complete = snapshot(checkpoint_id="checkpoint-2")
        continuation, agent, store = service(
            pending,
            state_after_invoke=complete,
        )
        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.CONTINUE,
                observed_checkpoint_id="checkpoint-1",
            ),
        )
        self.assertIsNone(agent.invocations[0][0])

        action = {"name": "write", "args": {}}
        waiting = snapshot(
            messages=[ai_call(("call-1", "write", {}))],
            next_nodes=["model"],
            interrupts=[
                Interrupt(
                    value={
                        "action_requests": [action],
                        "review_configs": [
                            {"allowed_decisions": ["approve", "reject"]}
                        ],
                    },
                    id="interrupt-1",
                )
            ],
        )
        continuation, agent, store = service(
            waiting,
            policies={"write": UNSAFE_POLICY},
            state_after_invoke=complete,
        )
        await continuation.execute(
            runtime(),
            ContinuationRequest(
                action=ContinuationAction.ANSWER_INTERRUPT,
                observed_checkpoint_id="checkpoint-1",
                decisions=({"type": "reject", "message": "not now"},),
            ),
        )
        command = agent.invocations[0][0]
        self.assertIsInstance(command, Command)
        self.assertEqual(
            command.resume,
            {"decisions": [{"type": "reject", "message": "not now"}]},
        )
        self.assertEqual(store.touched, ["thread-1"])

    async def test_invalid_action_and_stale_checkpoint_never_touch_or_invoke(self):
        continuation, agent, store = service(snapshot(next_nodes=["model"]))
        with self.assertRaises(InvalidContinuationAction):
            await continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id="checkpoint-1",
                    message="unsafe new turn",
                ),
            )
        self.assertFalse(store.touched)
        self.assertFalse(agent.invocations)

        inspection = await continuation.inspect(runtime())
        agent.state = snapshot(checkpoint_id="checkpoint-2", next_nodes=["model"])
        with self.assertRaisesRegex(StaleContinuationError, "inspect.*again"):
            await continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.CONTINUE,
                    observed_checkpoint_id=inspection.checkpoint_id,
                ),
            )
        self.assertFalse(store.touched)
        self.assertFalse(agent.invocations)


if __name__ == "__main__":
    unittest.main()
