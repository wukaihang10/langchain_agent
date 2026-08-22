from langchain.tools import ToolRuntime, tool

from langchain_agent.context import AgentContext
from langchain_agent.rag.graph import rag_graph


@tool
def search_repository_knowledge(
    query: str,
    runtime: ToolRuntime[AgentContext],
    top_k: int = 5,
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

    result = rag_graph.invoke(
        {
            "query": query,
            "top_k": top_k,
        },
        config=runtime.config,
        context=runtime.context,
    )

    return result["context"]


search_repository_knowledge.metadata = {
    "category": "read",
    "idempotent": True,
    "side_effect": False,
    "risk": "low",
}
