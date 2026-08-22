from mcp.server.fastmcp import FastMCP

mcp = FastMCP(
    "http-demo",
    host="127.0.0.1",
    port=8000,
    stateless_http=True,
    json_response=True,
)


@mcp.tool()
def echo(message: str) -> str:
    """Echo the provided message."""
    return message


@mcp.tool()
def square(number: int) -> int:
    """Return the square of an integer."""
    return number * number


if __name__ == "__main__":
    mcp.run(transport="streamable-http")
