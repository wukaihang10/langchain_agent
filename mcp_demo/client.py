import asyncio
import sys
from pathlib import Path

from mcp import Client, StdioServerParameters
from mcp.client.stdio import stdio_client
from mcp.types import TextContent

SERVER_PATH = Path(__file__).with_name("server.py")


async def main():
    server = StdioServerParameters(
        command=sys.executable,
        args=[str(SERVER_PATH)],
    )

    transport = stdio_client(server)

    async with Client(transport) as client:
        print(
            "Protocol version:",
            client.protocol_version,
        )

        print(
            "Server:",
            client.server_info,
        )

        tools_result = await client.list_tools()

        print("\n--- Tools ---")

        for tool in tools_result.tools:
            print(f"name: {tool.name}")
            print(f"description: {tool.description}")
            print(f"schema: {tool.input_schema}")
            print()

        result = await client.call_tool(
            "add",
            {
                "a": 2,
                "b": 3,
            },
        )

        print("--- Tool result ---")
        print("is_error:", result.is_error)
        print(
            "structured_content:",
            result.structured_content,
        )

        for block in result.content:
            if isinstance(block, TextContent):
                print("text:", block.text)


if __name__ == "__main__":
    asyncio.run(main())
