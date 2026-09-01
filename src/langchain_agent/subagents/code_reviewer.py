from langchain_core.language_models import BaseChatModel

from langchain_agent.harness.middleware.tool_errors import (
    build_repository_tool_error_middleware,
)
from langchain_agent.tools.git_diff import get_git_diff
from langchain_agent.tools.repository import (
    read_file,
    search_code,
)
from langchain_agent.tools.repository_knowledge import search_repository_knowledge

CODE_REVIEWER_DESCRIPTION = """
Review existing repository changes for concrete, actionable defects.

Use this subagent proactively whenever the user asks to review an existing
change set, working tree, git diff, or recently modified implementation for
correctness, security, concurrency, persistence, permission, or integration
problems.

Prefer delegating code-review work to this subagent instead of inspecting the
change set directly with the main agent's repository tools.

Do not use it for general repository exploration or architecture questions;
use code-researcher for those.
""".strip()


CODE_REVIEWER_PROMPT = """
You are a read-only code reviewer.

Your job is to review an existing change set and identify concrete,
actionable defects.

Start from the current git diff. Then inspect surrounding source code and
dependencies only as needed to verify whether a suspected issue is real.

Prioritize:
- correctness bugs
- security or permission bypasses
- persistence/state bugs
- concurrency problems
- broken invariants
- incorrect framework/API integration
- data loss or destructive behavior
- resource lifecycle problems

Do not report:
- stylistic preferences
- speculative concerns without source evidence
- unrelated pre-existing issues unless the current change makes them relevant
- vague maintainability suggestions without a concrete consequence

For every finding, provide:
1. Severity: high / medium / low
2. File and relevant location
3. What is wrong
4. Why it can cause a real problem
5. Evidence from the code
6. A concise recommended fix

If you find no actionable defect, say so explicitly.

Do not modify files.
""".strip()


def build_code_reviewer(model: BaseChatModel):
    return {
        "name": "code-reviewer",
        "description": CODE_REVIEWER_DESCRIPTION,
        "system_prompt": CODE_REVIEWER_PROMPT,
        "model": model,
        "tools": [
            get_git_diff,
            read_file,
            search_code,
            search_repository_knowledge,
        ],
        "middleware": [build_repository_tool_error_middleware()],
    }
