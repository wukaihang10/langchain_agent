import subprocess
from pathlib import Path

from langchain.agents.middleware import (
    AgentMiddleware,
)
from langgraph.runtime import Runtime

from langchain_agent.context import AgentContext
from langchain_agent.state import (
    GitAuditState,
)
from langchain_agent.state import FileEdition


def _run_git(
    repository_path: str,
    *args: str,
) -> str:
    result = subprocess.run(
        [
            "git",
            "-C",
            repository_path,
            *args,
        ],
        capture_output=True,
        text=True,
        encoding="utf-8",
        errors="surrogateescape",
    )

    if result.returncode != 0:
        raise RuntimeError(result.stderr.strip() or "Git command failed.")

    return result.stdout


def collect_file_editions(
    repository_path: str,
) -> list[FileEdition]:
    output = _run_git(
        repository_path,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )

    if not output:
        return []

    entries = output.split("\0")
    editions: list[FileEdition] = []

    index = 0

    while index < len(entries):
        entry = entries[index]

        if not entry:
            index += 1
            continue

        status = entry[:2]
        file_path = entry[3:]

        edition: FileEdition = {
            "file_path": file_path,
            "change_type": _parse_change_type(status),
        }

        # In -z porcelain format, rename/copy is:
        #
        # status target_path\0source_path\0
        if "R" in status or "C" in status:
            index += 1

            if index < len(entries):
                old_path = entries[index]

                if old_path:
                    edition["old_path"] = old_path

        editions.append(edition)

        index += 1

    return editions


def _parse_change_type(
    status: str,
) -> str:
    if status == "??":
        return "untracked"

    if "U" in status or status in {"AA", "DD"}:
        return "conflicted"

    if "R" in status:
        return "renamed"

    if "C" in status:
        return "copied"

    if "D" in status:
        return "deleted"

    if "A" in status:
        return "added"

    if "M" in status:
        return "modified"

    if "T" in status:
        return "type_changed"

    return "changed"


def ensure_clean_worktree(
    repository_path: str,
) -> None:
    editions = collect_file_editions(repository_path)

    if editions:
        changed_files = ", ".join(edition["file_path"] for edition in editions)

        raise RuntimeError(
            "Repository worktree must be clean "
            f"before starting a new agent thread: "
            f"{changed_files}"
        )


class GitAuditMiddleware(
    AgentMiddleware[
        GitAuditState,
        AgentContext,
    ]
):
    state_schema = GitAuditState

    def after_agent(
        self,
        state: GitAuditState,
        runtime: Runtime[AgentContext],
    ):
        audit = collect_file_editions(runtime.context.repository_path)

        return {
            "edited_file_list": (audit.edited_file_list),
            "edition_list": (audit.edition_list),
        }
