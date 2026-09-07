from __future__ import annotations

import argparse
import asyncio
from contextlib import suppress
from dataclasses import replace

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from langchain_agent.app.bootstrap import Application, bootstrap_application
from langchain_agent.app.config import AppConfig
from langchain_agent.app.session_continuation import (
    ContinuationAction,
    ContinuationError,
    ContinuationInspection,
    ContinuationRequest,
    ContinuationStatus,
)
from langchain_agent.app.session_runtime import (
    SessionRuntime,
    build_session_runtime,
    delete_session,
)
from langchain_agent.cli.commands import (
    create_session_interactively,
    resolve_repository_path,
    select_session,
)
from langchain_agent.cli.hitl import collect_hitl_decisions
from langchain_agent.cli.rendering import (
    render_active_session,
    render_continuation,
    render_help,
    render_mcp_tools,
    render_result,
    render_sessions,
    render_stop_result,
)
from langchain_agent.harness.permissions.models import PermissionMode

KEY_BINDINGS = KeyBindings()


@KEY_BINDINGS.add("c-l")
def clear_screen(event):
    event.app.renderer.clear()


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--permission-mode",
        choices=[
            PermissionMode.DEFAULT.value,
            PermissionMode.READ_ONLY.value,
            PermissionMode.FULL_ACCESS.value,
        ],
        default=PermissionMode.DEFAULT.value,
    )

    return parser.parse_args(argv)


def _build_session_runtime(
    application: Application,
    session,
) -> SessionRuntime:
    repository_path = resolve_repository_path(
        session=session,
        session_store=application.session_store,
    )

    return build_session_runtime(
        session=session,
        repository_path=repository_path,
        session_store=application.session_store,
        repository_knowledge_provider=application.repository_knowledge,
        config=application.config,
    )


async def run_cli(application: Application) -> None:
    prompt_session = PromptSession(key_bindings=KEY_BINDINGS)
    active_runtime: SessionRuntime | None = None
    active_inspection: ContinuationInspection | None = None

    render_mcp_tools(application.mcp.tools)
    print("\nLangGraph Agent" "\nType /help for commands.")

    while True:
        try:
            prompt = (
                "\nYou> "
                if active_runtime is None
                else f"\n[{active_runtime.session.name}] You> "
            )
            user_input = (await prompt_session.prompt_async(prompt)).strip()
        except (EOFError, KeyboardInterrupt):
            print()
            break

        if not user_input:
            continue

        if user_input in {"/exit", "/quit"}:
            break

        if user_input == "/help":
            render_help()
            continue

        if user_input == "/list":
            render_sessions(
                application.session_store.list(),
                active_thread_id=(
                    active_runtime.session.thread_id
                    if active_runtime is not None
                    else None
                ),
            )
            continue

        if user_input == "/new":
            session = create_session_interactively(application.session_store)
            active_runtime = _build_session_runtime(application, session)
            render_active_session(
                active_runtime.session,
                active_runtime.context,
            )
            active_inspection = await application.continuation.inspect(active_runtime)
            render_continuation(active_inspection)
            continue

        if user_input == "/resume":
            session = select_session(
                application.session_store,
                active_thread_id=(
                    active_runtime.session.thread_id
                    if active_runtime is not None
                    else None
                ),
            )

            if session is None:
                continue

            if (
                active_runtime is not None
                and session.thread_id == active_runtime.session.thread_id
            ):
                active_inspection = await application.continuation.inspect(
                    active_runtime
                )
                render_continuation(active_inspection)
                continue

            active_runtime = _build_session_runtime(application, session)
            render_active_session(
                active_runtime.session,
                active_runtime.context,
            )
            active_inspection = await application.continuation.inspect(active_runtime)
            render_continuation(active_inspection)
            continue

        if user_input == "/terminate":
            if active_runtime is None or active_inspection is None:
                print("No active session. Use /new or /resume first.")
                continue

            if (
                ContinuationAction.TERMINATE_TURN
                not in active_inspection.allowed_actions
            ):
                render_continuation(active_inspection)
                continue

            try:
                result, active_inspection = await _execute_with_live_hitl(
                    application=application,
                    runtime=active_runtime,
                    request=ContinuationRequest(
                        action=ContinuationAction.TERMINATE_TURN,
                        observed_checkpoint_id=active_inspection.checkpoint_id,
                    ),
                    prompt_session=prompt_session,
                )
            except ContinuationError as exc:
                print(f"Turn termination rejected: {exc}")
                active_inspection = await application.continuation.inspect(
                    active_runtime
                )
                render_continuation(active_inspection)
                continue

            render_result(result)
            if active_inspection.status != ContinuationStatus.READY:
                render_continuation(active_inspection)
            continue

        if user_input == "/stop":
            if active_runtime is None:
                print("No active turn is running.")
                continue
            try:
                stop_result = await application.continuation.stop(
                    active_runtime.session.thread_id
                )
            except ContinuationError as exc:
                print(f"Active turn stop failed: {exc}")
                active_inspection = await application.continuation.inspect(
                    active_runtime
                )
                render_continuation(active_inspection)
                continue
            render_stop_result(stop_result)
            if stop_result.inspection is not None:
                active_inspection = stop_result.inspection
                if active_inspection.status != ContinuationStatus.READY:
                    render_continuation(active_inspection)
            continue

        if user_input == "/continue":
            if active_runtime is None or active_inspection is None:
                print("No active session. Use /new or /resume first.")
                continue

            if active_inspection.status == ContinuationStatus.WAITING_HUMAN:
                decisions = tuple(collect_hitl_decisions(active_inspection.interrupts))
                action = ContinuationAction.ANSWER_INTERRUPT
            elif active_inspection.status in {
                ContinuationStatus.RESUMABLE,
                ContinuationStatus.OUTCOME_UNKNOWN,
            }:
                decisions = ()
                action = ContinuationAction.CONTINUE
            else:
                render_continuation(active_inspection)
                continue

            try:
                result, active_inspection = await _execute_with_live_hitl(
                    application=application,
                    runtime=active_runtime,
                    request=ContinuationRequest(
                        action=action,
                        observed_checkpoint_id=active_inspection.checkpoint_id,
                        decisions=decisions,
                    ),
                    prompt_session=prompt_session,
                )
            except ContinuationError as exc:
                print(f"Continuation rejected: {exc}")
                active_inspection = await application.continuation.inspect(
                    active_runtime
                )
                render_continuation(active_inspection)
                continue

            render_result(result)
            if active_inspection.status != ContinuationStatus.READY:
                render_continuation(active_inspection)
            continue

        if user_input == "/rename":
            if active_runtime is None:
                print("No active session. " "Use /new or /resume first.")
                continue

            new_name = input("New session name: ").strip()

            if not new_name:
                print("Session name cannot be empty.")
                continue

            renamed_session = application.session_store.rename(
                active_runtime.session.thread_id,
                new_name,
            )

            if renamed_session is None:
                print("Failed to rename session.")
                continue

            active_runtime = replace(
                active_runtime,
                session=renamed_session,
            )
            print(f"Session renamed to " f"'{active_runtime.session.name}'.")
            continue

        if user_input == "/delete":
            target_session = select_session(
                application.session_store,
                active_thread_id=(
                    active_runtime.session.thread_id
                    if active_runtime is not None
                    else None
                ),
            )

            if target_session is None:
                continue

            answer = (
                input(f"Delete session " f"'{target_session.name}'? " "[Y/N]: ")
                .strip()
                .lower()
            )

            if answer not in {"y", "yes"}:
                print("Delete cancelled.")
                continue

            deleting_active = (
                active_runtime is not None
                and target_session.thread_id == active_runtime.session.thread_id
            )
            deleted = await delete_session(
                session=target_session,
                session_store=application.session_store,
                checkpointer=application.checkpointer,
            )

            if not deleted:
                print("Failed to delete session.")
                continue

            print(f"Deleted session " f"'{target_session.name}'.")

            if deleting_active:
                active_runtime = None
                active_inspection = None
                print("No active session. " "Use /new or /resume.")

            continue

        if user_input.startswith("/"):
            print("Unknown command. " "Use /help.")
            continue

        if active_runtime is None:
            print("No active session. " "Use /new or /resume first.")
            continue

        if active_inspection is None:
            active_inspection = await application.continuation.inspect(active_runtime)

        try:
            result, active_inspection = await _execute_with_live_hitl(
                application=application,
                runtime=active_runtime,
                request=ContinuationRequest(
                    action=ContinuationAction.START_TURN,
                    observed_checkpoint_id=active_inspection.checkpoint_id,
                    message=user_input,
                ),
                prompt_session=prompt_session,
            )
        except ContinuationError as exc:
            print(f"Message rejected: {exc}")
            active_inspection = await application.continuation.inspect(active_runtime)
            render_continuation(active_inspection)
            continue

        render_result(result)
        if active_inspection.status != ContinuationStatus.READY:
            render_continuation(active_inspection)


async def _execute_with_live_hitl(
    *,
    application: Application,
    runtime: SessionRuntime,
    request: ContinuationRequest,
    prompt_session: PromptSession | None = None,
) -> tuple[dict, ContinuationInspection]:
    execution = await _execute_with_running_prompt(
        application=application,
        runtime=runtime,
        request=request,
        prompt_session=prompt_session,
    )

    while execution.inspection.status == ContinuationStatus.WAITING_HUMAN:
        decisions = collect_hitl_decisions(execution.inspection.interrupts)
        execution = await _execute_with_running_prompt(
            application=application,
            runtime=runtime,
            request=ContinuationRequest(
                action=ContinuationAction.ANSWER_INTERRUPT,
                observed_checkpoint_id=execution.inspection.checkpoint_id,
                decisions=tuple(decisions),
            ),
            prompt_session=prompt_session,
        )

    return execution.value, execution.inspection


async def _execute_with_running_prompt(
    *,
    application: Application,
    runtime: SessionRuntime,
    request: ContinuationRequest,
    prompt_session: PromptSession | None,
):
    if prompt_session is None:
        return await application.continuation.execute(runtime, request)

    execution_task = asyncio.create_task(
        application.continuation.execute(runtime, request)
    )
    completed_normally = False
    input_task = None
    try:
        while True:
            input_task = asyncio.create_task(
                prompt_session.prompt_async(
                    "\nAgent is running. Enter /stop to cancel> "
                )
            )
            done, _ = await asyncio.wait(
                {execution_task, input_task},
                return_when=asyncio.FIRST_COMPLETED,
            )

            if execution_task in done or execution_task.done():
                if not input_task.done():
                    input_task.cancel()
                with suppress(asyncio.CancelledError):
                    await input_task
                completed_normally = True
                return await execution_task

            running_input = (await input_task).strip()
            if running_input != "/stop":
                print(
                    "Only /stop is accepted while the Agent is running; "
                    "that input was rejected and not queued."
                )
                continue

            print(
                "Stop requested; waiting for it to settle before finalizing "
                "the latest checkpoint."
            )
            stop_result = await application.continuation.stop(
                runtime.session.thread_id
            )
            render_stop_result(stop_result)
            completed_normally = True
            return await execution_task
    finally:
        if input_task is not None and not input_task.done():
            input_task.cancel()
            with suppress(asyncio.CancelledError):
                await input_task
        if not completed_normally and not execution_task.done():
            execution_task.cancel()
            with suppress(asyncio.CancelledError):
                await execution_task


async def main(argv: list[str] | None = None) -> None:
    args = parse_args(argv)
    config = AppConfig(
        permission_mode=PermissionMode(args.permission_mode),
    )

    async with bootstrap_application(config) as application:
        await run_cli(application)


def cli() -> None:
    """Run the asynchronous CLI from a synchronous console-script entry point."""
    asyncio.run(main())
