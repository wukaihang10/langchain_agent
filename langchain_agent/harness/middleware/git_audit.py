from typing import NotRequired

from langchain.agents.middleware import AgentMiddleware, AgentState
from langgraph.runtime import Runtime

from langchain_agent.app.context import AgentContext
from langchain_agent.integrations.git import FileEdition, collect_file_editions


class GitAuditState(AgentState):
    edited_file_list: NotRequired[list[str]]
    edition_list: NotRequired[list[FileEdition]]


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
            "edited_file_list": audit.get("edited_file_list", []),
            "edition_list": audit.get("edition_list", []),
        }
