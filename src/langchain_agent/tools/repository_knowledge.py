from typing import Annotated

from langchain.tools import ToolRuntime, tool
from pydantic import Field, StringConstraints

from langchain_agent.app.context import AgentContext
from langchain_agent.repository_knowledge import (
    RepositoryKnowledgeError,
    SearchResponse,
)
from langchain_agent.tools.repository_errors import RepositoryToolError


def format_search_response_for_agent(response: SearchResponse) -> str:
    """Render structured repository evidence for the model context."""

    return response.context


@tool
def search_repository_knowledge(
    query: Annotated[str, StringConstraints(strip_whitespace=True, min_length=1)],
    runtime: ToolRuntime[AgentContext],
    top_k: Annotated[int, Field(ge=1, le=12)] = 5,
) -> str:
    """Semantically search the current repository for code relevant to a question.

    Use this tool for broad or conceptual repository exploration, especially when:
    - you do not yet know which files or symbols are relevant,
    - the question spans multiple files or modules,
    - you need to understand architecture, responsibilities, relationships,
      or implementation flows.

    Use `search_code` instead for exact identifiers or literal keywords.
    Use `read_file` instead when the relevant file is already known and exact
    source content is needed.
    """

    repository_knowledge = runtime.context.repository_knowledge

    try:
        repository_knowledge.prepare()

        response = repository_knowledge.search(
            query,
            top_k=top_k,
        )
    except RepositoryKnowledgeError as error:
        raise RepositoryToolError(
            f"Repository knowledge search failed: {error}"
        ) from error

    return format_search_response_for_agent(response)


search_repository_knowledge.metadata = {
    "category": "read",
    "idempotent": True,
    "side_effect": False,
    "risk": "low",
}
