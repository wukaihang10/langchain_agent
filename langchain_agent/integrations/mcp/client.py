from dataclasses import dataclass

from langchain_core.tools import BaseTool
from langchain_mcp_adapters.client import MultiServerMCPClient

from langchain_agent.integrations.mcp.config import MCPConfig


@dataclass(frozen=True)
class MCPIntegration:
    client: MultiServerMCPClient | None
    tools: list[BaseTool]


async def load_mcp_integration(config: MCPConfig) -> MCPIntegration:
    if not config.servers:
        return MCPIntegration(
            client=None,
            tools=[],
        )

    client = MultiServerMCPClient(
        config.servers,
        tool_name_prefix=True,
    )
    tools = await client.get_tools()

    return MCPIntegration(
        client=client,
        tools=tools,
    )
