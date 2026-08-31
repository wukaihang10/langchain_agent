from collections.abc import Sequence

from langchain_core.tools import BaseTool

from langchain_agent.app.context import AgentContext
from langchain_agent.persistence.sessions import Session


def render_mcp_tools(tools: Sequence[BaseTool]) -> None:
    print("\n--- MCP tools ---")

    for tool in tools:
        print(f"- {tool.name}")


def render_result(result: dict) -> None:
    messages = result["messages"]

    if messages:
        print("\n--- Agent ---")
        print(messages[-1].content)


def render_sessions(
    sessions: list[Session],
    *,
    active_thread_id: str | None = None,
) -> None:
    if not sessions:
        print("\nNo sessions.")
        return

    print("\n--- Sessions ---")

    for index, session in enumerate(sessions, start=1):
        marker = "*" if session.thread_id == active_thread_id else " "
        print(
            f"{marker} {index}. "
            f"{session.name} "
            f"({session.repository_path}) "
            f"[{session.thread_id}]"
        )


def render_active_session(
    session: Session,
    context: AgentContext,
) -> None:
    print("\n--- Active session ---")
    print(f"Session: {session.name}")
    print(f"Repository: {context.repository_path}")
    print(f"Thread: {session.thread_id}")
    print("Permission mode: " f"{context.permission_mode.value}")


def render_help() -> None:
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
