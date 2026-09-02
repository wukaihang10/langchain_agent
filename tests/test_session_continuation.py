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
from langchain_agent.harness.permissions.models import (
    ToolCategory,
    ToolPolicy,
    ToolRisk,
)
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry

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
        context=object(),
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
            {ContinuationAction.ANSWER_INTERRUPT},
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
            {ContinuationAction.CONTINUE},
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
                self.assertFalse(inspection.allowed_actions)

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

    async def test_unknown_outcome_precedes_protocol_repair(self):
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

        self.assertEqual(inspection.status, ContinuationStatus.OUTCOME_UNKNOWN)


class SessionContinuationExecutionTests(unittest.IsolatedAsyncioTestCase):
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
