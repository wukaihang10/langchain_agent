from __future__ import annotations

import hashlib
from pathlib import Path
from typing import Any, TypeAlias

from langchain_agent.repository_knowledge._internal.source.python_loader import (
    PythonDocumentLoader,
)

RepositorySnapshot: TypeAlias = dict[
    str,
    dict[str, Any],
]

class RepositorySnapshotBuilder:
    """
    为 Python 仓库建立文件内容快照。

    Snapshot 使用相对 POSIX 路径作为键：

        agent/agent.py

    每个文件保存：

        size
        sha256
    """

    def __init__(
        self,
        loader: PythonDocumentLoader,
    ) -> None:
        self.loader = loader

    def build(
        self,
        repository_path: str | Path,
    ) -> RepositorySnapshot:
        root = Path(repository_path).resolve()

        snapshot: RepositorySnapshot = {}

        for file_path in self.loader.discover_files(root):
            source = file_path.relative_to(root).as_posix()

            size, sha256 = self._fingerprint_file(file_path)

            snapshot[source] = {
                "size": size,
                "sha256": sha256,
            }

        return snapshot

    @staticmethod
    def _fingerprint_file(
        file_path: Path,
    ) -> tuple[int, str]:
        """
        一次读取同时计算文件大小与 SHA-256。

        不使用修改时间作为正确性依据，
        因为修改时间可能被保留或被外部工具改变。
        """

        digest = hashlib.sha256()
        size = 0

        with file_path.open("rb") as file:
            while block := file.read(1024 * 1024):
                digest.update(block)
                size += len(block)

        return size, digest.hexdigest()


def describe_repository_changes(
    stored_snapshot: RepositorySnapshot,
    current_snapshot: RepositorySnapshot,
    preview_limit: int = 5,
) -> str:
    """
    描述两个仓库快照之间的区别。
    """

    stored_paths = set(stored_snapshot)
    current_paths = set(current_snapshot)

    added = sorted(current_paths - stored_paths)

    removed = sorted(stored_paths - current_paths)

    modified = sorted(
        path
        for path in stored_paths & current_paths
        if (stored_snapshot[path] != current_snapshot[path])
    )

    groups: list[str] = []

    def append_group(
        label: str,
        paths: list[str],
    ) -> None:
        if not paths:
            return

        preview = ", ".join(paths[:preview_limit])

        remaining = len(paths) - preview_limit

        if remaining > 0:
            preview += f", ... and {remaining} more"

        groups.append(f"{label} ({len(paths)}): " f"{preview}")

    append_group("added", added)
    append_group("removed", removed)
    append_group("modified", modified)

    if not groups:
        return "no source changes"

    return "; ".join(groups)
