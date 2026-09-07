import asyncio
import unittest
from contextlib import suppress
from types import SimpleNamespace

from langchain_core.messages import AIMessage
from langgraph.types import StateSnapshot

from langchain_agent.app.context import AgentContext
from langchain_agent.app.session_continuation import (
    ActiveExecutionError,
    ContinuationAction,
    ContinuationRequest,
    ContinuationStatus,
    SessionContinuation,
    StopOutcome,
)
from langchain_agent.harness.middleware.turn_recovery import TurnRecoveryMode
from langchain_agent.harness.permissions.registry import ToolPolicyRegistry


def snapshot(*, checkpoint_id=None, next_nodes=(), thread_id="thread-1"):
    configurable = {"thread_id": thread_id, "checkpoint_ns": ""}
    if checkpoint_id is not None:
        configurable["checkpoint_id"] = checkpoint_id
    return StateSnapshot(
        values={},
        next=tuple(next_nodes),
        config={"configurable": configurable},
        metadata=None,
        created_at=None,
        parent_config=None,
        tasks=(),
        interrupts=(),
    )


def runtime(thread_id="thread-1"):
    return SimpleNamespace(
        session=SimpleNamespace(thread_id=thread_id),
        invoke_config={"configurable": {"thread_id": thread_id}},
        context=AgentContext(
            repository_path="C:/repository",
            repository_knowledge=object(),
        ),
    )


class FakeSessionStore:
    def __init__(self):
        self.touched = []

    def touch(self, thread_id):
        self.touched.append(thread_id)


class CancellableAgent:
    def __init__(self, *, state_after_cancel=None, finish_on_cancel=False):
        self.state = snapshot()
        self.state_after_cancel = state_after_cancel or snapshot(
            checkpoint_id="checkpoint-after-cancel",
            next_nodes=("model",),
        )
        self.finish_on_cancel = finish_on_cancel
        self.started = asyncio.Event()
        self.cancellation_observed = asyncio.Event()
        self.allow_cleanup = asyncio.Event()
        self.termination_started = asyncio.Event()
        self.invocations = []

    async def aget_state(self, config):
        return self.state

    async def ainvoke(self, input_value, *, config, context):
        self.invocations.append((input_value, config, context))
        if (
            context.turn_recovery is not None
            and context.turn_recovery.mode == TurnRecoveryMode.TERMINATE
        ):
            self.termination_started.set()
            self.state = snapshot(checkpoint_id="checkpoint-ready")
            return {"messages": [AIMessage(content="turn terminated")]}

        self.started.set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.state = self.state_after_cancel
            self.cancellation_observed.set()
            await self.allow_cleanup.wait()
            if self.finish_on_cancel:
                return {"messages": [AIMessage(content="completed naturally")]}
            raise


class MultiThreadAgent:
    def __init__(self, thread_ids):
        self.states = {
            thread_id: snapshot(thread_id=thread_id) for thread_id in thread_ids
        }
        self.started = {thread_id: asyncio.Event() for thread_id in thread_ids}
        self.cancelled = {thread_id: asyncio.Event() for thread_id in thread_ids}

    async def aget_state(self, config):
        return self.states[config["configurable"]["thread_id"]]

    async def ainvoke(self, input_value, *, config, context):
        thread_id = config["configurable"]["thread_id"]
        self.started[thread_id].set()
        try:
            await asyncio.Future()
        except asyncio.CancelledError:
            self.states[thread_id] = snapshot(
                checkpoint_id=f"{thread_id}-ready",
                thread_id=thread_id,
            )
            self.cancelled[thread_id].set()
            raise


class ActiveTurnStopTests(unittest.IsolatedAsyncioTestCase):
    async def test_stop_waits_for_cancelled_invocation_before_termination(self):
        agent = CancellableAgent()
        store = FakeSessionStore()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=store,
        )
        active_runtime = runtime()

        execution = asyncio.create_task(
            continuation.execute(
                active_runtime,
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="long task",
                ),
            )
        )
        await agent.started.wait()

        stopping = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        self.assertFalse(agent.termination_started.is_set())
        with self.assertRaises(ActiveExecutionError):
            await continuation.execute(
                active_runtime,
                ContinuationRequest(
                    action=ContinuationAction.CONTINUE,
                    observed_checkpoint_id="checkpoint-after-cancel",
                ),
            )

        agent.allow_cleanup.set()
        stop_result = await stopping
        execution_result = await execution

        self.assertEqual(stop_result.outcome, StopOutcome.STOPPED)
        self.assertEqual(stop_result.inspection.status, ContinuationStatus.READY)
        self.assertEqual(execution_result.inspection, stop_result.inspection)
        self.assertEqual(execution_result.value, stop_result.value)
        self.assertTrue(agent.termination_started.is_set())
        self.assertEqual(store.touched, ["thread-1", "thread-1"])

    async def test_second_execution_for_same_thread_is_rejected(self):
        agent = CancellableAgent()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        active_runtime = runtime()
        execution = asyncio.create_task(
            continuation.execute(
                active_runtime,
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="first",
                ),
            )
        )
        await agent.started.wait()

        with self.assertRaises(ActiveExecutionError):
            await continuation.execute(
                active_runtime,
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="second",
                ),
            )

        stopping = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()
        await stopping
        await execution
        self.assertEqual(len(agent.invocations), 2)

    async def test_repeated_stop_requests_share_one_finalization(self):
        agent = CancellableAgent()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="long task",
                ),
            )
        )
        await agent.started.wait()

        first = asyncio.create_task(continuation.stop("thread-1"))
        second = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()
        first_result, second_result = await asyncio.gather(first, second)
        await execution

        self.assertEqual(first_result, second_result)
        self.assertEqual(first_result.outcome, StopOutcome.STOPPED)
        self.assertEqual(len(agent.invocations), 2)

    async def test_stop_without_active_execution_is_a_no_op(self):
        agent = CancellableAgent()
        store = FakeSessionStore()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=store,
        )

        result = await continuation.stop("thread-1")

        self.assertEqual(result.outcome, StopOutcome.NOT_RUNNING)
        self.assertIsNone(result.inspection)
        self.assertFalse(agent.invocations)
        self.assertFalse(store.touched)

    async def test_completion_during_cancellation_is_reported_truthfully(self):
        agent = CancellableAgent(
            state_after_cancel=snapshot(checkpoint_id="checkpoint-ready"),
            finish_on_cancel=True,
        )
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="almost done",
                ),
            )
        )
        await agent.started.wait()

        stopping = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()
        stop_result = await stopping
        execution_result = await execution

        self.assertEqual(
            stop_result.outcome,
            StopOutcome.COMPLETED_BEFORE_STOP,
        )
        self.assertEqual(stop_result.value, execution_result.value)
        self.assertFalse(agent.termination_started.is_set())
        self.assertEqual(len(agent.invocations), 1)

    async def test_needs_repair_after_cancellation_fails_closed(self):
        damaged = StateSnapshot(
            values={
                "messages": [
                    AIMessage(
                        content="",
                        tool_calls=[
                            {
                                "name": "write",
                                "args": {},
                                "id": "unresolved",
                                "type": "tool_call",
                            }
                        ],
                    )
                ]
            },
            next=("model",),
            config={
                "configurable": {
                    "thread_id": "thread-1",
                    "checkpoint_ns": "",
                    "checkpoint_id": "damaged",
                }
            },
            metadata=None,
            created_at=None,
            parent_config=None,
            tasks=(),
            interrupts=(),
        )
        agent = CancellableAgent(state_after_cancel=damaged)
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="damage",
                ),
            )
        )
        await agent.started.wait()

        stopping = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()
        stop_result = await stopping
        execution_result = await execution

        self.assertEqual(stop_result.outcome, StopOutcome.NEEDS_REPAIR)
        self.assertEqual(
            stop_result.inspection.status,
            ContinuationStatus.NEEDS_REPAIR,
        )
        self.assertEqual(execution_result.inspection, stop_result.inspection)
        self.assertFalse(agent.termination_started.is_set())

    async def test_cancellation_not_requested_by_stop_propagates(self):
        agent = CancellableAgent()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="cancel externally",
                ),
            )
        )
        await agent.started.wait()

        execution.cancel()
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()

        with self.assertRaises(asyncio.CancelledError):
            await execution
        self.assertEqual(
            (await continuation.stop("thread-1")).outcome,
            StopOutcome.NOT_RUNNING,
        )

    async def test_cancelling_one_stop_waiter_does_not_cancel_finalization(self):
        agent = CancellableAgent()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="long task",
                ),
            )
        )
        await agent.started.wait()

        abandoned_waiter = asyncio.create_task(continuation.stop("thread-1"))
        await agent.cancellation_observed.wait()
        abandoned_waiter.cancel()
        with self.assertRaises(asyncio.CancelledError):
            await abandoned_waiter

        replacement_waiter = asyncio.create_task(continuation.stop("thread-1"))
        agent.allow_cleanup.set()
        result = await replacement_waiter
        await execution

        self.assertEqual(result.outcome, StopOutcome.STOPPED)
        self.assertEqual(len(agent.invocations), 2)

    async def test_stopping_one_thread_does_not_cancel_another(self):
        agent = MultiThreadAgent(("thread-1", "thread-2"))
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )

        executions = {
            thread_id: asyncio.create_task(
                continuation.execute(
                    runtime(thread_id),
                    ContinuationRequest(
                        action=ContinuationAction.START_TURN,
                        observed_checkpoint_id=None,
                        message=thread_id,
                    ),
                )
            )
            for thread_id in ("thread-1", "thread-2")
        }
        await asyncio.gather(
            *(started.wait() for started in agent.started.values())
        )

        first_stop = await continuation.stop("thread-1")
        await executions["thread-1"]

        self.assertEqual(first_stop.outcome, StopOutcome.STOPPED)
        self.assertTrue(agent.cancelled["thread-1"].is_set())
        self.assertFalse(agent.cancelled["thread-2"].is_set())
        self.assertFalse(executions["thread-2"].done())

        await continuation.stop("thread-2")
        await executions["thread-2"]

    async def test_application_close_cancels_process_local_execution(self):
        agent = CancellableAgent()
        continuation = SessionContinuation(
            agent=agent,
            policy_registry=ToolPolicyRegistry(),
            session_store=FakeSessionStore(),
        )
        execution = asyncio.create_task(
            continuation.execute(
                runtime(),
                ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=None,
                    message="still running",
                ),
            )
        )
        await agent.started.wait()

        closing = asyncio.create_task(continuation.aclose())
        await agent.cancellation_observed.wait()
        agent.allow_cleanup.set()
        await closing

        with self.assertRaises(asyncio.CancelledError):
            await execution
        self.assertEqual(
            (await continuation.stop("thread-1")).outcome,
            StopOutcome.NOT_RUNNING,
        )

    async def asyncTearDown(self):
        for task in asyncio.all_tasks():
            if task is not asyncio.current_task() and not task.done():
                task.cancel()
                with suppress(asyncio.CancelledError):
                    await task


if __name__ == "__main__":
    unittest.main()
