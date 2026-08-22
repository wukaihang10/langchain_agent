from mcp.server.fastmcp import FastMCP

mcp = FastMCP("demo-server")


@mcp.tool()
def add(a: int, b: int) -> int:
    """Add two integers."""
    return a + b


@mcp.tool()
def greet(name: str) -> str:
    """Return a greeting."""
    return f"Hello, {name}!"


if __name__ == "__main__":
    mcp.run(transport="stdio")
