from collections.abc import Mapping, Sequence

from langchain_core.tools import BaseTool

from langchain_agent.app.context import AgentContext
from langchain_agent.app.session_continuation import ContinuationInspection
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


def render_continuation(inspection: ContinuationInspection) -> None:
    print("\n--- Continuation status ---")
    print(f"Status: {inspection.status.value}")
    print(inspection.reason)

    if inspection.pending_nodes:
        print("Pending nodes: " + ", ".join(inspection.pending_nodes))

    if inspection.unresolved_tool_calls:
        automatic = ", ".join(
            f"{call.name}{('[' + (call.args.get('subagent_type') or '') + ']' if call.name == 'task' else '')} ({call.id})"
            for call in inspection.unresolved_tool_calls
            if call.replay_safe
        )
        uncertain = ", ".join(
            f"{call.name}{('[' + (call.args.get('subagent_type') or '') + ']' if call.name == 'task' else '')} ({call.id})"
            for call in inspection.unresolved_tool_calls
            if call.outcome_unknown
        )
        pending = ", ".join(
            f"{call.name}{('[' + (call.args.get('subagent_type') or '') + ']' if call.name == 'task' else '')} ({call.id})"
            for call in inspection.unresolved_tool_calls
            if not call.replay_safe and not call.outcome_unknown
        )
        if automatic:
            print("Automatic retry Tools: " + automatic)
        if uncertain:
            print("Outcome-unknown Tools: " + uncertain)
        if pending:
            print("Pending Tools for approval: " + pending)

    for interrupt in inspection.interrupts:
        value = interrupt.value
        if not isinstance(value, Mapping):
            continue

        action_requests = value.get("action_requests", ())
        review_configs = value.get("review_configs", ())
        if not isinstance(action_requests, Sequence):
            continue

        for index, action in enumerate(action_requests):
            if not isinstance(action, Mapping):
                continue
            # print("Tool:", action.get("name", "unknown"))
            # print("Arguments:", action.get("args", {}))
            if description := action.get("description"):
                print("Tool Interrupt:", description)

            review_config = (
                review_configs[index]
                if isinstance(review_configs, Sequence)
                and index < len(review_configs)
                and isinstance(review_configs[index], Mapping)
                else {}
            )
            allowed = review_config.get("allowed_decisions", ())
            if isinstance(allowed, Sequence):
                print("Allowed decisions: " + ", ".join(map(str, allowed)))

    if inspection.allowed_actions:
        if any(
            action.value
            in {
                "CONTINUE",
                "ANSWER_INTERRUPT",
            }
            for action in inspection.allowed_actions
        ):
            print("Next: enter /continue.")
        else:
            print("Next: enter a message to start a turn.")
        if any(
            action.value == "TERMINATE_TURN"
            for action in inspection.allowed_actions
        ):
            print("Or enter /terminate to stop the unfinished turn.")


def render_help() -> None:
    print(
        "\nAvailable commands:\n"
        "  /list    List sessions\n"
        "  /new     Create a new session\n"
        "  /resume  Resume an existing session\n"
        "  /continue Continue pending work or answer an interrupt\n"
        "  /terminate Terminate the unfinished turn without continuing it\n"
        "  /rename  Rename the active session\n"
        "  /delete  Delete a session\n"
        "  /help    Show commands\n"
        "  /exit    Exit"
    )
