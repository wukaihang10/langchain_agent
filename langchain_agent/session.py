from __future__ import annotations

from dataclasses import asdict, dataclass
from datetime import datetime
import json
from pathlib import Path
from uuid import uuid4


@dataclass
class Session:
    thread_id: str
    name: str
    repository_path: str
    created_at: str
    updated_at: str


class SessionStore:
    def __init__(self, path: str | Path):
        self.path = Path(path)

    def create(
        self,
        *,
        name: str,
        repository_path: str,
    ) -> Session:
        now = datetime.now().astimezone().isoformat()

        session = Session(
            thread_id=str(uuid4()),
            name=name,
            repository_path=repository_path,
            created_at=now,
            updated_at=now,
        )

        sessions = self.list()
        sessions.append(session)
        self._save_all(sessions)

        return session

    def list(self) -> list[Session]:
        if not self.path.exists():
            return []

        content = self.path.read_text(encoding="utf-8")

        if not content:
            return []

        data = json.loads(content)

        return [Session(**item) for item in data.get("sessions", [])]

    def get(self, thread_id: str) -> Session | None:
        for session in self.list():
            if session.thread_id == thread_id:
                return session

        return None

    def touch(self, thread_id: str) -> Session | None:
        sessions = self.list()

        target = None
        now = datetime.now().astimezone().isoformat()

        for session in sessions:
            if session.thread_id == thread_id:
                session.updated_at = now
                target = session
                break

        if target is not None:
            self._save_all(sessions)

        return target

    def rename(
        self,
        thread_id: str,
        name: str,
    ) -> Session | None:
        sessions = self.list()

        for session in sessions:
            if session.thread_id == thread_id:
                session.name = name
                session.updated_at = datetime.now().astimezone().isoformat()

                self._save_all(sessions)
                return session

        return None

    def update_repository_path(
        self,
        thread_id: str,
        repository_path: str,
    ) -> Session | None:
        sessions = self.list()

        for session in sessions:
            if session.thread_id == thread_id:
                session.repository_path = repository_path
                session.updated_at = datetime.now().astimezone().isoformat()

                self._save_all(sessions)

                return session

        return None

    def delete(self, thread_id: str) -> bool:
        sessions = self.list()

        remaining_sessions = [
            session for session in sessions if session.thread_id != thread_id
        ]

        if len(remaining_sessions) == len(sessions):
            return False

        self._save_all(remaining_sessions)

        return True

    def _save_all(self, sessions: list[Session]) -> None:
        data = {"sessions": [asdict(session) for session in sessions]}

        self.path.write_text(
            json.dumps(
                data,
                ensure_ascii=False,
                indent=2,
            ),
            encoding="utf-8",
        )
