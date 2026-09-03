from langchain_agent.harness.permissions.models import ToolPolicy


class ToolPolicyRegistry:
    def __init__(
        self,
        policies: dict[str, ToolPolicy] | None = None,
    ):
        self._policies = dict(policies or {})

    def register(
        self,
        tool_name: str,
        policy: ToolPolicy,
    ) -> None:
        self._policies[tool_name] = policy

    def get(
        self,
        tool_name: str,
    ) -> ToolPolicy | None:
        return self._policies.get(tool_name)


def build_native_tool_policies(
    tools,
) -> dict[str, ToolPolicy]:
    policies: dict[str, ToolPolicy] = {}

    for tool in tools:
        metadata = tool.metadata or {}

        try:
            policy = ToolPolicy(
                category=metadata["category"],
                side_effect=metadata["side_effect"],
                risk=metadata["risk"],
                idempotent=metadata["idempotent"],
            )
        except KeyError as exc:
            raise ValueError(
                f"Local tool '{tool.name}' " "is missing permission metadata."
            ) from exc

        policies[tool.name] = policy

    return policies


def build_tool_policy_registry(
    *,
    local_tools,
    external_policy_overrides: (
        dict[
            str,
            ToolPolicy,
        ]
        | None
    ) = None,
) -> ToolPolicyRegistry:
    policies = build_native_tool_policies(local_tools)

    policies.update(external_policy_overrides or {})

    return ToolPolicyRegistry(policies)
