import asyncio
import sys
from pathlib import Path

from langchain_core.tools import BaseTool, tool
from langchain_mcp_adapters.client import MultiServerMCPClient
from langchain_core.messages import AIMessage
from langgraph.prebuilt import ToolNode
from langgraph.runtime import Runtime

SERVER_PATH = Path(__file__).with_name("server.py")


@tool
def multiply(a: int, b: int) -> int:
    """Multiply two integers."""
    return a * b


async def main():
    client = MultiServerMCPClient(
        {
            "demo": {
                "transport": "stdio",
                "command": sys.executable,
                "args": [str(SERVER_PATH)],
            }
        },
        tool_name_prefix=True,
    )

    mcp_tools = await client.get_tools()

    native_tools = [
        multiply,
    ]

    unified_tools = [
        *native_tools,
        *mcp_tools,
    ]

    print("\n--- Unified tools ---")

    for current_tool in unified_tools:
        print(f"name: {current_tool.name}")
        print(f"type: {type(current_tool).__name__}")
        print(
            "is BaseTool:",
            isinstance(current_tool, BaseTool),
        )
        print(f"description: {current_tool.description}")
        print(f"args: {current_tool.args}")
        print()

    add_tool = next(
        current_tool for current_tool in mcp_tools if current_tool.name == "demo_add"
    )

    result = await add_tool.ainvoke(
        {
            "a": 2,
            "b": 3,
        }
    )

    print("--- MCP tool result ---")
    print(result)

    tool_node = ToolNode(unified_tools)

    state = {
        "messages": [
            AIMessage(
                content="",
                tool_calls=[
                    {
                        "name": "demo_add",
                        "args": {
                            "a": 10,
                            "b": 20,
                        },
                        "id": "test-call-1",
                        "type": "tool_call",
                    }
                ],
            )
        ]
    }

    result = await tool_node.ainvoke(state, runtime=Runtime())

    print("\n--- ToolNode result ---")
    print(result["messages"][-1])


if __name__ == "__main__":
    asyncio.run(main())
