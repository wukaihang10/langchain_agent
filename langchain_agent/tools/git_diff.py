from langchain.tools import tool, ToolRuntime

from langchain_agent.context import AgentContext
from langchain_agent.git_changes import collect_git_diff, collect_untracked_files
from langchain_agent.permissions.types import ToolCategory, ToolRisk


@tool
def get_git_diff(
    runtime: ToolRuntime[AgentContext],
    file_path: str | None = None,
) -> dict:
    """Read the current repository changes for code review.

    Returns the tracked git diff relative to HEAD and, for a repository-wide
    request, the paths of untracked files that should also be reviewed.

    Args:
        file_path: Optional repository-relative file path whose diff to return.
    """

    repo_path = runtime.context.repository_path

    diff = collect_git_diff(
        repo_path,
        file_path=file_path,
    )

    untracked_files = (
        [] if file_path is not None else collect_untracked_files(repo_path)
    )

    return {
        "success": True,
        "diff": diff,
        "untracked_files": untracked_files,
    }


get_git_diff.metadata = {
    "category": ToolCategory.READ.value,
    "idempotent": True,
    "side_effect": False,
    "risk": ToolRisk.LOW.value,
}
