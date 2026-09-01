from dataclasses import dataclass, field
from pathlib import Path

from platformdirs import PlatformDirs

from langchain_agent.harness.permissions.models import PermissionMode
from langchain_agent.repository_knowledge.config import RetrievalMode

_APP_NAME = "langchain-agent"


@dataclass(frozen=True)
class AppPaths:
    data_dir: Path
    config_dir: Path
    cache_dir: Path

    @classmethod
    def user_default(cls) -> "AppPaths":
        """Resolve per-user paths independently of the process working directory."""

        directories = PlatformDirs(_APP_NAME, appauthor=False)

        return cls(
            data_dir=directories.user_data_path,
            config_dir=directories.user_config_path,
            cache_dir=directories.user_cache_path,
        )

    @classmethod
    def under(cls, root: str | Path) -> "AppPaths":
        """Co-locate application paths under an injected root for tests or embeds."""

        root_path = Path(root)

        return cls(
            data_dir=root_path,
            config_dir=root_path,
            cache_dir=root_path,
        )

    @property
    def checkpoint_path(self) -> Path:
        return self.data_dir / "checkpoints.sqlite"

    @property
    def session_path(self) -> Path:
        return self.data_dir / "sessions.json"

    @property
    def mcp_config_path(self) -> Path:
        return self.config_dir / "mcp.json"

    @property
    def environment_path(self) -> Path:
        return self.config_dir / ".env"

    @property
    def index_root(self) -> Path:
        return self.cache_dir / "indexes"

    def ensure_directories(self) -> None:
        for directory in (
            self.data_dir,
            self.config_dir,
            self.index_root,
        ):
            directory.mkdir(parents=True, exist_ok=True)


@dataclass(frozen=True)
class AppConfig:
    permission_mode: PermissionMode = PermissionMode.DEFAULT
    paths: AppPaths = field(default_factory=AppPaths.user_default)
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
