from __future__ import annotations

import argparse
import asyncio
import json
import sys
from pathlib import Path
from prompt_toolkit import PromptSession
from prompt_toolkit.key_binding import KeyBindings

from langgraph.checkpoint.base import BaseCheckpointSaver
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.checkpoint.sqlite.aio import AsyncSqliteSaver
from langgraph.types import Command
from langchain.agents import create_agent
from langchain.agents.middleware import (
    SummarizationMiddleware,
    TodoListMiddleware,
)
from deepagents.backends import StateBackend
from deepagents.middleware import SubAgentMiddleware

from langchain_agent.model import create_model
from langchain_agent.context import AgentContext
from langchain_agent.git_changes import GitAuditMiddleware
from langchain_agent.tool_errors import ToolErrorMiddleware
from langchain_agent.tool_retry import (
    build_mcp_no_retry_failure_middleware,
    build_mcp_retry_middleware,
)
from langchain_agent.tools.repository import REPOSITORY_TOOLS
from langchain_agent.tools.rag import search_repository_knowledge
from langchain_agent.ragservice.repository_manager import RepositoryKnowledgeManager
from langchain_agent.permissions.types import PermissionMode
from langchain_agent.session import Session, SessionStore
from langchain_agent.mcp_config import load_mcp_config
from langchain_agent.permissions.registry import build_tool_policy_registry
from langchain_agent.permissions.middleware import (
    PermissionEnforcementMiddleware,
    build_hitl_middleware,
)
from langchain_agent.subagents.code_researcher import build_code_researcher
from langchain_agent.subagents.code_reviewer import build_code_reviewer

AGENT_DIR = Path(".agent")
CHECKPOINT_PATH = AGENT_DIR / "checkpoints.sqlite"
SESSION_PATH = AGENT_DIR / "sessions.json"
MCP_CONFIG_PATH = Path(".agent") / "mcp.json"

NATIVE_TOOLS = [*REPOSITORY_TOOLS, search_repository_knowledge]


kb = KeyBindings()


@kb.add("c-l")
def clear_screen(event):
    event.app.renderer.clear()


session = PromptSession(
    key_bindings=kb,
)


def parse_args():
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

    return parser.parse_args()


def collect_hitl_decisions(interrupts) -> list[dict]:
    decisions: list[dict] = []

    for interrupt_item in interrupts:
        value = interrupt_item.value

        action_requests = value.get(
            "action_requests",
            [],
        )

        review_configs = value.get(
            "review_configs",
            [],
        )

        for action, review_config in zip(
            action_requests,
            review_configs,
            strict=True,
        ):
            print("\n--- Tool approval required ---")

            print(
                "Tool:",
                action["name"],
            )

            print(
                "Arguments:",
                action["args"],
            )

            description = action.get("description")

            if description:
                print(
                    "Reason:",
                    description,
                )

            allowed = review_config["allowed_decisions"]

            print(
                "Allowed:",
                ", ".join(allowed),
            )

            while True:
                answer = input("Approve? [y/n]: ").strip().lower()

                if answer in {"y", "yes"} and "approve" in allowed:
                    decisions.append(
                        {
                            "type": "approve",
                        }
                    )
                    break

                if answer in {"n", "no"} and "reject" in allowed:
                    message = input("Optional rejection reason: ").strip()

                    decision = {
                        "type": "reject",
                    }

                    if message:
                        decision["message"] = message

                    decisions.append(decision)

                    break

                print("Invalid decision. " f"Allowed: {allowed}")

    return decisions


async def invoke_with_hitl(*, agent, input_value, config, context):
    current_input = input_value

    while True:
        result = await agent.ainvoke(
            current_input,
            config=config,
            context=context,
            version="v2",
        )

        if not result.interrupts:
            return result.value

        decisions = collect_hitl_decisions(result.interrupts)

        current_input = Command(
            resume={
                "decisions": decisions,
            }
        )


def print_result(result: dict) -> None:
    messages = result["messages"]

    if messages:
        print("\n--- Agent ---")
        print(messages[-1].content)

    # print("\n--- Edited files ---")

    # edited_files = result.get(
    #     "edited_file_list",
    #     [],
    # )

    # if not edited_files:
    #     print("(none)")
    # else:
    #     for file_path in edited_files:
    #         print(f"- {file_path}")

    # print("\n--- Git audit ---")

    # editions = result.get(
    #     "edition_list",
    #     [],
    # )

    # if not editions:
    #     print("(clean)")
    # else:
    #     print(
    #         json.dumps(
    #             editions,
    #             ensure_ascii=False,
    #             indent=2,
    #         )
    #     )

    # print("\n--- Todos ---")

    # todos = result.get(
    #     "todos",
    #     [],
    # )

    # if not todos:
    #     print("(none)")
    # else:
    #     for todo in todos:
    #         print(f"- [{todo['status']}] " f"{todo['content']}")


def print_sessions(
    session_store: SessionStore,
    *,
    active_thread_id: str | None = None,
) -> list[Session]:
    sessions = session_store.list()

    if not sessions:
        print("\nNo sessions.")
        return []

    print("\n--- Sessions ---")

    for index, session in enumerate(sessions, start=1):
        marker = "*" if session.thread_id == active_thread_id else " "

        print(
            f"{marker} {index}. "
            f"{session.name} "
            f"({session.repository_path}) "
            f"[{session.thread_id}]"
        )

    return sessions


def print_active_session(
    session: Session,
    context: AgentContext,
) -> None:
    print("\n--- Active session ---")
    print(f"Session: {session.name}")
    print(f"Repository: {context.repository_path}")
    print(f"Thread: {session.thread_id}")
    print("Permission mode: " f"{context.permission_mode.value}")


def create_session_interactively(
    session_store: SessionStore,
) -> Session:
    while True:
        name = input("Session name: ").strip()

        if not name:
            print("Session name cannot be empty.")
            continue

        repository_path = (
            Path(input("Repository path: ").strip()).expanduser().resolve()
        )

        if not repository_path.is_dir():
            print("Repository path does not exist.")
            continue

        return session_store.create(
            name=name,
            repository_path=str(repository_path),
        )


def select_session(
    session_store: SessionStore,
    *,
    active_thread_id: str | None = None,
) -> Session | None:
    sessions = print_sessions(
        session_store,
        active_thread_id=active_thread_id,
    )

    if not sessions:
        return None

    print("0. Cancel")

    while True:
        choice = input("\nSelect session: ").strip()

        if choice == "0":
            return None

        try:
            index = int(choice) - 1
        except ValueError:
            print("Invalid selection.")
            continue

        if 0 <= index < len(sessions):
            return sessions[index]

        print("Invalid selection.")


async def delete_session(
    *,
    session: Session,
    session_store: SessionStore,
    checkpointer: BaseCheckpointSaver,
) -> bool:
    if session_store.get(session.thread_id) is None:
        return False

    await checkpointer.adelete_thread(session.thread_id)

    return session_store.delete(session.thread_id)


def resolve_repository_path(
    session: Session,
    session_store: SessionStore,
) -> Path:
    repository_path = Path(session.repository_path).expanduser().resolve()

    if repository_path.is_dir():
        return repository_path

    print("\nRepository no longer exists:\n" f"{repository_path}")

    while True:
        raw_path = input("Enter the new repository path: ").strip()

        new_path = Path(raw_path).expanduser().resolve()

        if not new_path.is_dir():
            print("Repository path does not exist.")
            continue

        session_store.update_repository_path(
            session.thread_id,
            str(new_path),
        )

        # 同步当前内存中的 Session。
        session.repository_path = str(new_path)

        return new_path


def build_session_runtime(
    *,
    session: Session,
    session_store: SessionStore,
    permission_mode: PermissionMode,
) -> tuple[AgentContext, dict]:
    repository_path = resolve_repository_path(
        session=session,
        session_store=session_store,
    )

    rag_manager = RepositoryKnowledgeManager(
        retrieval_mode="fast",
    )

    context = AgentContext(
        repository_path=str(repository_path),
        rag_manager=rag_manager,
        permission_mode=permission_mode,
    )

    config = {
        "configurable": {
            "thread_id": session.thread_id,
        },
        # "recursion_limit": 60,
        "run_name": "langgraph-agent-v0",
        "tags": [
            "langgraph-agent",
            "v0",
            "local",
        ],
        "metadata": {
            "agent_version": "langgraph-agent-v0",
            "thread_id": session.thread_id,
            "repository": repository_path.name,
            "permission_mode": permission_mode.value,
        },
    }

    session_store.touch(session.thread_id)

    return context, config


async def main():
    args = parse_args()

    permission_mode = PermissionMode(args.permission_mode)

    SESSION_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )
    CHECKPOINT_PATH.parent.mkdir(
        parents=True,
        exist_ok=True,
    )

    session_store = SessionStore(SESSION_PATH)

    active_session: Session | None = None
    context: AgentContext | None = None
    config: dict | None = None

    #
    # Tool Config
    #

    mcp_config = load_mcp_config(MCP_CONFIG_PATH)

    if mcp_config.servers:
        mcp_client = MultiServerMCPClient(
            mcp_config.servers,
            tool_name_prefix=True,
        )

        mcp_tools = await mcp_client.get_tools()
    else:
        mcp_tools = []

    print("\n--- MCP tools ---")

    for tool in mcp_tools:
        print(f"- {tool.name}")

    tools = [*NATIVE_TOOLS, *mcp_tools]

    protected_tool_names = {tool.name for tool in tools}

    policy_registry = build_tool_policy_registry(
        local_tools=NATIVE_TOOLS,
        external_policy_overrides=mcp_config.tool_policies,
    )

    mcp_retry_middleware = build_mcp_retry_middleware(
        mcp_tools=mcp_tools,
        policy_registry=policy_registry,
    )
    mcp_no_retry_failure_middleware = build_mcp_no_retry_failure_middleware(
        mcp_tools=mcp_tools,
        policy_registry=policy_registry,
    )

    async with AsyncSqliteSaver.from_conn_string(str(CHECKPOINT_PATH)) as checkpointer:
        #
        # MiddleWare config
        #

        code_researcher = build_code_researcher(create_model())
        code_reviewer = build_code_reviewer(create_model())

        subagent_middleware = SubAgentMiddleware(
            backend=StateBackend(),
            subagents=[
                code_researcher,
                code_reviewer,
            ],
            system_prompt="""
                Use subagents proactively when their specialization matches the task.

                Delegation policy:
                - For complex, multi-step, read-only repository investigation spanning
                multiple files, symbols, or modules, delegate to code-researcher.
                - For review of an existing change set, working tree, git diff, or recently
                modified code, delegate to code-reviewer.
                - Use repository tools directly for simple, targeted lookups that normally
                require only one or two tool calls, such as reading a known file or locating
                a known symbol.
                - After a subagent returns, use direct tools only when targeted verification
                or follow-up is still needed.
                """.strip(),
        )

        hitl_middleware = build_hitl_middleware(
            tools=tools,
            registry=policy_registry,
        )

        permission_enforcement_middleware = PermissionEnforcementMiddleware(
            registry=policy_registry,
            protected_tool_names=protected_tool_names,
        )

        model = create_model()
        summary_model = create_model()

        middleware = [
            hitl_middleware,
            TodoListMiddleware(),
            subagent_middleware,
            SummarizationMiddleware(
                model=summary_model,
                trigger=("tokens", 30_000),
                keep=("tokens", 8_000),
            ),
            permission_enforcement_middleware,
            *(
                [mcp_retry_middleware]
                if mcp_retry_middleware is not None
                else []
            ),
            *(
                [mcp_no_retry_failure_middleware]
                if mcp_no_retry_failure_middleware is not None
                else []
            ),
            ToolErrorMiddleware(),
            GitAuditMiddleware(),
        ]

        #
        # Agent config
        #

        agent = create_agent(
            model=model,
            tools=tools,
            middleware=middleware,
            context_schema=AgentContext,
            system_prompt=(
                "You are a repository analysis agent. "
                "Use repository tools when source-code "
                "information is required. "
                "Do not invent file contents."
            ),
            checkpointer=checkpointer,
            name="create-agent-runtime-demo",
        )

        print("\nLangGraph Agent" "\nType /help for commands.")

        #
        # Loop
        #

        while True:
            try:
                if active_session is None:
                    prompt = "\nYou> "
                else:
                    prompt = f"\n[{active_session.name}] You> "

                user_input = await session.prompt_async(prompt)

                user_input = user_input.strip()

            except (
                EOFError,
                KeyboardInterrupt,
            ):
                print()
                break

            if not user_input:
                continue

            #
            # Exit
            #

            if user_input in {
                "/exit",
                "/quit",
            }:
                break

            #
            # Help
            #

            if user_input == "/help":
                print(
                    "\nAvailable commands:\n"
                    "  /list    List sessions\n"
                    "  /new     Create a new session\n"
                    "  /resume  Resume an existing session\n"
                    "  /rename  Rename the active session\n"
                    "  /delete  Delete a session\n"
                    "  /help    Show commands\n"
                    "  /exit    Exit"
                )
                continue

            #
            # List
            #

            if user_input == "/list":
                print_sessions(
                    session_store,
                    active_thread_id=(
                        active_session.thread_id if active_session is not None else None
                    ),
                )
                continue

            #
            # New
            #

            if user_input == "/new":
                new_session = create_session_interactively(session_store)

                new_context, new_config = build_session_runtime(
                    session=new_session,
                    session_store=session_store,
                    permission_mode=permission_mode,
                )

                active_session = new_session
                context = new_context
                config = new_config

                print_active_session(
                    active_session,
                    context,
                )
                continue

            #
            # Resume
            #

            if user_input == "/resume":
                selected_session = select_session(
                    session_store,
                    active_thread_id=(
                        active_session.thread_id if active_session is not None else None
                    ),
                )

                if selected_session is None:
                    continue

                if (
                    active_session is not None
                    and selected_session.thread_id == active_session.thread_id
                ):
                    print("That session is already active.")
                    continue

                new_context, new_config = build_session_runtime(
                    session=selected_session,
                    session_store=session_store,
                    permission_mode=permission_mode,
                )

                active_session = selected_session
                context = new_context
                config = new_config

                print_active_session(
                    active_session,
                    context,
                )
                continue

            #
            # Rename
            #

            if user_input == "/rename":
                if active_session is None:
                    print("No active session. " "Use /new or /resume first.")
                    continue

                new_name = input("New session name: ").strip()

                if not new_name:
                    print("Session name cannot be empty.")
                    continue

                renamed_session = session_store.rename(
                    active_session.thread_id,
                    new_name,
                )

                if renamed_session is None:
                    print("Failed to rename session.")
                    continue

                active_session = renamed_session

                print(f"Session renamed to " f"'{active_session.name}'.")
                continue

            #
            # Delete
            #

            if user_input == "/delete":
                target_session = select_session(
                    session_store,
                    active_thread_id=(
                        active_session.thread_id if active_session is not None else None
                    ),
                )

                if target_session is None:
                    continue

                answer = (
                    input(f"Delete session " f"'{target_session.name}'? " "[Y/N]: ")
                    .strip()
                    .lower()
                )

                if answer not in {
                    "y",
                    "yes",
                }:
                    print("Delete cancelled.")
                    continue

                deleting_active = (
                    active_session is not None
                    and target_session.thread_id == active_session.thread_id
                )

                deleted = await delete_session(
                    session=target_session,
                    session_store=session_store,
                    checkpointer=checkpointer,
                )

                if not deleted:
                    print("Failed to delete session.")
                    continue

                print(f"Deleted session " f"'{target_session.name}'.")

                if deleting_active:
                    active_session = None
                    context = None
                    config = None

                    print("No active session. " "Use /new or /resume.")

                continue

            #
            # Unknown slash command
            #

            if user_input.startswith("/"):
                print("Unknown command. " "Use /help.")
                continue

            #
            # Normal Agent query
            #

            if active_session is None or context is None or config is None:
                print("No active session. " "Use /new or /resume first.")
                continue

            result = await invoke_with_hitl(
                agent=agent,
                input_value={
                    "messages": [
                        {
                            "role": "user",
                            "content": user_input,
                        }
                    ]
                },
                config=config,
                context=context,
            )

            print_result(result)


if __name__ == "__main__":
    asyncio.run(main())
