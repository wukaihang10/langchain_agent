from __future__ import annotations

import argparse
import asyncio
from dataclasses import replace

from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from langchain_agent.app.bootstrap import Application, bootstrap_application
from langchain_agent.app.config import AppConfig
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
from langchain_agent.cli.hitl import invoke_with_hitl
from langchain_agent.cli.rendering import (
    render_active_session,
    render_help,
    render_mcp_tools,
    render_result,
    render_sessions,
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
                print("That session is already active.")
                continue

            active_runtime = _build_session_runtime(application, session)
            render_active_session(
                active_runtime.session,
                active_runtime.context,
            )
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
                print("No active session. " "Use /new or /resume.")

            continue

        if user_input.startswith("/"):
            print("Unknown command. " "Use /help.")
            continue

        if active_runtime is None:
            print("No active session. " "Use /new or /resume first.")
            continue

        result = await invoke_with_hitl(
            agent=application.agent,
            input_value={
                "messages": [
                    {
                        "role": "user",
                        "content": user_input,
                    }
                ]
            },
            config=active_runtime.invoke_config,
            context=active_runtime.context,
        )
        render_result(result)


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
