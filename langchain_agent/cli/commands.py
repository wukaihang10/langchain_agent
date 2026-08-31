from pathlib import Path

from langchain_agent.cli.rendering import render_sessions
from langchain_agent.persistence.sessions import Session, SessionStore


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
    sessions = session_store.list()
    render_sessions(
        sessions,
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
        session.repository_path = str(new_path)

        return new_path
