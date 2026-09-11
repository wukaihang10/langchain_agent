from __future__ import annotations

import shutil
import subprocess
import tempfile
from collections.abc import AsyncIterator, Callable
from contextlib import asynccontextmanager
from dataclasses import dataclass, replace
from pathlib import Path

from langchain_core.language_models import BaseChatModel
from langgraph.checkpoint.memory import InMemorySaver

from evals.targets.repository_fact import RepositoryFactTarget
from langchain_agent.app.agent import NATIVE_TOOLS, build_agent
from langchain_agent.app.config import AppConfig, AppPaths
from langchain_agent.app.context import AgentContext
from langchain_agent.app.repository_knowledge import RepositoryKnowledgeProvider
from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.harness.permissions.registry import build_tool_policy_registry
from langchain_agent.repository_knowledge import (
    EmbeddingClient,
    QueryExpander,
    RepositoryKnowledgeConfig,
)

FIXTURE_VERSION = "repository_fact_v0"
FIXTURE_TEMPLATE = Path(__file__).resolve().parents[1] / "fixtures" / FIXTURE_VERSION


@dataclass(frozen=True)
class RepositoryFactModels:
    primary: BaseChatModel
    summary: BaseChatModel
    researcher: BaseChatModel
    reviewer: BaseChatModel


@dataclass(frozen=True)
class RepositoryFactEnvironment:
    target: RepositoryFactTarget
    repository_path: Path
    runtime_root: Path
    config: AppConfig


@asynccontextmanager
async def open_repository_fact_environment(
    *,
    models: RepositoryFactModels,
    embedding_client_factory: Callable[[], EmbeddingClient],
    query_expander: QueryExpander,
    config: AppConfig | None = None,
    fixture_template: Path = FIXTURE_TEMPLATE,
) -> AsyncIterator[RepositoryFactEnvironment]:
    """Build an isolated read-only production Agent around the fixture."""

    template = fixture_template.resolve()
    if not template.is_dir():
        raise NotADirectoryError(f"fixture repository does not exist: {template}")

    base_config = config or AppConfig(agent_version="langchain-agent-v0")

    with tempfile.TemporaryDirectory(prefix="langchain-agent-eval-") as directory:
        environment_root = Path(directory)
        repository_path = environment_root / "repository"
        runtime_root = environment_root / "runtime"
        shutil.copytree(template, repository_path)
        _initialize_git_repository(repository_path)

        evaluation_config = replace(
            base_config,
            permission_mode=PermissionMode.READ_ONLY,
            paths=AppPaths.under(runtime_root),
        )
        evaluation_config.paths.ensure_directories()

        repository_knowledge = RepositoryKnowledgeProvider(
            index_root=evaluation_config.paths.index_root,
            embedding_client_factory=embedding_client_factory,
            query_expander=query_expander,
            config=RepositoryKnowledgeConfig(
                retrieval_mode=evaluation_config.retrieval_mode,
                max_query_rewrites=evaluation_config.max_query_rewrites,
            ),
        )
        context = AgentContext(
            repository_path=str(repository_path.resolve()),
            repository_knowledge=repository_knowledge.get(repository_path),
            permission_mode=PermissionMode.READ_ONLY,
        )
        agent = build_agent(
            model=models.primary,
            summary_model=models.summary,
            researcher_model=models.researcher,
            reviewer_model=models.reviewer,
            native_tools=NATIVE_TOOLS,
            mcp_tools=[],
            policy_registry=build_tool_policy_registry(local_tools=NATIVE_TOOLS),
            checkpointer=InMemorySaver(),
            config=evaluation_config,
        )
        target = RepositoryFactTarget(
            agent=agent,
            context=context,
            config=evaluation_config,
            fixture_version=FIXTURE_VERSION.replace("_", "-"),
        )

        yield RepositoryFactEnvironment(
            target=target,
            repository_path=repository_path,
            runtime_root=runtime_root,
            config=evaluation_config,
        )


def _initialize_git_repository(repository_path: Path) -> None:
    _run_git(repository_path, "init", "--quiet")
    _run_git(repository_path, "add", "--all")
    _run_git(
        repository_path,
        "-c",
        "user.name=LangChain Agent Evaluation",
        "-c",
        "user.email=evaluation@example.invalid",
        "-c",
        "commit.gpgsign=false",
        "commit",
        "--quiet",
        "-m",
        "Create evaluation fixture baseline",
    )


def _run_git(repository_path: Path, *arguments: str) -> None:
    completed = subprocess.run(
        ["git", *arguments],
        cwd=repository_path,
        check=False,
        capture_output=True,
        text=True,
        encoding="utf-8",
    )
    if completed.returncode == 0:
        return

    detail = completed.stderr.strip() or completed.stdout.strip()
    raise RuntimeError(f"Git fixture setup failed: {detail}")
