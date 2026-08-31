from __future__ import annotations

from dataclasses import dataclass
import json
import os
import re
from pathlib import Path
from typing import Any
from dotenv import load_dotenv

from langchain_agent.harness.permissions.models import ToolPolicy

_ENV_PATTERN = re.compile(r"\$\{([A-Za-z_][A-Za-z0-9_]*)\}")

load_dotenv()


@dataclass(frozen=True)
class MCPConfig:
    servers: dict[str, dict[str, Any]]
    tool_policies: dict[str, ToolPolicy]


def _resolve_env(value: Any) -> Any:
    if isinstance(value, str):

        def replace(match: re.Match) -> str:
            env_name = match.group(1)
            env_value = os.getenv(env_name)

            if env_value is None:
                raise ValueError(f"Environment variable " f"'{env_name}' is not set.")

            return env_value

        return _ENV_PATTERN.sub(
            replace,
            value,
        )

    if isinstance(value, dict):
        return {key: _resolve_env(item) for key, item in value.items()}

    if isinstance(value, list):
        return [_resolve_env(item) for item in value]

    return value


def load_mcp_config(
    path: str | Path,
) -> MCPConfig:
    path = Path(path)

    if not path.exists():
        return MCPConfig(
            servers={},
            tool_policies={},
        )

    data = json.loads(path.read_text(encoding="utf-8"))

    data = _resolve_env(data)

    servers = data.get("servers", {})

    raw_policies = data.get(
        "tool_policies",
        {},
    )

    tool_policies = {
        tool_name: ToolPolicy(
            category=policy["category"],
            side_effect=policy["side_effect"],
            risk=policy["risk"],
            idempotent=policy["idempotent"],
        )
        for tool_name, policy in raw_policies.items()
    }

    return MCPConfig(
        servers=servers,
        tool_policies=tool_policies,
    )
