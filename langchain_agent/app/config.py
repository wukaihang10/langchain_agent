from dataclasses import dataclass, field
from pathlib import Path

from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.repository_knowledge.config import RetrievalMode


@dataclass(frozen=True)
class AppPaths:
    agent_dir: Path = Path(".agent")

    @property
    def checkpoint_path(self) -> Path:
        return self.agent_dir / "checkpoints.sqlite"

    @property
    def session_path(self) -> Path:
        return self.agent_dir / "sessions.json"

    @property
    def mcp_config_path(self) -> Path:
        return self.agent_dir / "mcp.json"

    @property
    def index_root(self) -> Path:
        return self.agent_dir / "indexes"

    def ensure_directories(self) -> None:
        self.agent_dir.mkdir(parents=True, exist_ok=True)
        self.index_root.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    paths: AppPaths = field(default_factory=AppPaths)
    agent_version: str = "langgraph-agent-v0"
    agent_name: str = "create-agent-runtime-demo"
    embedding_model_name: str = "BAAI/bge-small-zh-v1.5"
    retrieval_mode: RetrievalMode = "fast"
    max_query_rewrites: int = 2
    summarization_trigger_tokens: int = 30_000
    summarization_keep_tokens: int = 8_000

    @property
    def run_tags(self) -> tuple[str, ...]:
        return (
            "langgraph-agent",
            "v0",
            "local",
        )
