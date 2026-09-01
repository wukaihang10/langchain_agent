from typing import Literal, NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from langchain_agent.app.context import AgentContext
from langchain_agent.integrations.git import (
    FileEdition,
    GitIntegrationError,
    collect_file_editions,
)


class GitAuditState(AgentState):
    edited_file_list: NotRequired[list[str]]
    edition_list: NotRequired[list[FileEdition]]
    git_audit_status: NotRequired[Literal["available", "unavailable"]]
    git_audit_error: NotRequired[str | None]


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
        try:
            audit = collect_file_editions(runtime.context.repository_path)
        except GitIntegrationError as error:
            return {
                "edited_file_list": [],
                "edition_list": [],
                "git_audit_status": "unavailable",
                "git_audit_error": str(error),
            }

        return {
            "edited_file_list": audit["edited_file_list"],
            "edition_list": audit["edition_list"],
            "git_audit_status": "available",
            "git_audit_error": None,
        }
