from langchain.agents import create_agent

from langchain_agent.app.context import AgentContext
from langchain_agent.harness.middleware.tool_errors import ToolErrorMiddleware
from langchain_agent.tools.repository import (
    list_files,
    read_file,
    search_code,
    summarize_repository,
)
from langchain_agent.tools.repository_knowledge import (
    search_repository_knowledge,
)

RESEARCH_TOOLS = [
    list_files,
    read_file,
    search_code,
    summarize_repository,
    search_repository_knowledge,
]


CODE_RESEARCHER_DESCRIPTION = """
Investigate complex questions about the current code repository.

Use this subagent proactively when a task requires multi-step, read-only
exploration across multiple files, symbols, or modules, such as architecture
analysis, dependency tracing, debugging investigation, or understanding
unfamiliar implementation flows.

Prefer delegation to this subagent over performing a long sequence of
repository lookups in the main agent.

Do not delegate simple lookups that can normally be answered with one or two
direct tool calls, such as reading a known file or locating a known symbol.
""".strip()


CODE_RESEARCHER_PROMPT = """
You are a read-only repository research agent.

Investigate the delegated question independently using authoritative
repository evidence.

Tool guidance:
- Use search_code for exact identifiers, symbols, and literal keywords.
- Use read_file when you know which file contains relevant implementation.
- Use search_repository_knowledge for broad architectural or conceptual
  investigation.
- Use summarize_repository or list_files when repository structure is useful.

Continue investigating until you have enough evidence to answer the task.

Do not modify files.

Return a concise report containing:
1. conclusion
2. relevant files and symbols
3. important execution or dependency relationships
4. supporting evidence
5. unresolved questions or uncertainty, if any

Do not dump large raw tool outputs.
""".strip()


def build_code_researcher(model):
    runnable = create_agent(
        model=model,
        tools=RESEARCH_TOOLS,
        middleware=[ToolErrorMiddleware()],
        context_schema=AgentContext,
        system_prompt=CODE_RESEARCHER_PROMPT,
        name="code-researcher",
    )

    return {
        "name": "code-researcher",
        "description": CODE_RESEARCHER_DESCRIPTION,
        "runnable": runnable,
    }
