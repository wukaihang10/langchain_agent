import subprocess
from typing import NotRequired, TypedDict


class GitIntegrationError(RuntimeError):
    """Expected failure while invoking or communicating with Git."""


class GitCommandError(GitIntegrationError):
    """Git started successfully but rejected or could not complete a command."""


class FileEdition(TypedDict):
    file_path: str
    change_type: str
    old_path: NotRequired[str]


class FileEditionAudit(TypedDict):
    edited_file_list: list[str]
    edition_list: list[FileEdition]


def _run_git(
    repository_path: str,
    *args: str,
) -> str:
    command = [
        "git",
        "-C",
        repository_path,
        *args,
    ]

    try:
        result = subprocess.run(
            command,
            capture_output=True,
            text=True,
            encoding="utf-8",
            errors="surrogateescape",
        )
    except OSError as error:
        raise GitIntegrationError(f"Could not start Git: {error}") from error

    if result.returncode != 0:
        detail = result.stderr.strip() or result.stdout.strip()
        message = f"Git command failed with exit code {result.returncode}."

        if detail:
            message = f"{message} {detail}"

        raise GitCommandError(message)

    return result.stdout


def collect_file_editions(
    repository_path: str,
) -> FileEditionAudit:
    output = _run_git(
        repository_path,
        "status",
        "--porcelain=v1",
        "-z",
        "--untracked-files=all",
    )

    if not output:
        return {
            "edited_file_list": [],
            "edition_list": [],
        }

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

    return {
        "edited_file_list": [edition["file_path"] for edition in editions],
        "edition_list": editions,
    }


def collect_git_diff(
    repository_path: str,
    file_path: str | None = None,
) -> str:
    args = [
        "diff",
        "HEAD",
        "--",
    ]

    if file_path is not None:
        args.append(file_path)

    return _run_git(
        repository_path,
        *args,
    )


def collect_untracked_files(
    repository_path: str,
) -> list[str]:
    output = _run_git(
        repository_path,
        "ls-files",
        "--others",
        "--exclude-standard",
    )

    return [line for line in output.splitlines() if line]


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
    audit = collect_file_editions(repository_path)

    if audit["edited_file_list"]:
        changed_files = ", ".join(audit["edited_file_list"])

        raise RuntimeError(
            "Repository worktree must be clean "
            f"before starting a new agent thread: "
            f"{changed_files}"
        )
