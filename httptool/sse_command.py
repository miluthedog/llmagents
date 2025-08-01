import os
import subprocess
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
import uvicorn
import argparse


mcp = FastMCP("terminal")
DEFAULT_WORKSPACE = os.path.expanduser(".")

@mcp.tool()
async def run_command(command: str) -> str:
    """
    Execute shell or terminal commands using this tool.

    Use it when the user requests to run, test, or inspect something via the command line—such as listing files, checking versions, installing packages, or running scripts.

    Input: A single shell command as a string.

    Examples:
        - "ls -la"
        - "python3 --version"
        - "ping google.com"
        - "echo Hello, World"
    """
    try:
        result = subprocess.run(
            command,
            shell=True,
            cwd=DEFAULT_WORKSPACE,
            capture_output=True,
            text=True
        )
        return result.stdout or result.stderr
    except Exception as e:
        return str(e)

@mcp.tool()
async def add_numbers(a: float, b: float) -> float:
    """
    Use this tool when asked to add two numbers.

    Arg:
        a (float): The first number.
        b (float): The second number.

    Returns:
        float: a + b
    """
    return a + b

def create_starlette_app(mcp_server: Server, *, debug: bool = False) -> Starlette:
    sse = SseServerTransport("/messages/")

    async def handle_sse(request: Request) -> None:
        async with sse.connect_sse(
            request.scope,
            request.receive,
            request._send
        ) as (read_stream, write_stream):
            await mcp_server.run(
                read_stream,
                write_stream,
                mcp_server.create_initialization_options()
            )

    return Starlette(
        debug=debug,
        routes=[
            Route("/sse", endpoint=handle_sse),
            Mount("/messages/", app=sse.handle_post_message)
        ]
    )


if __name__ == "__main__":
    mcp_server = mcp._mcp_server

    parser = argparse.ArgumentParser(description='Run MCP server')
    parser.add_argument('--host', default='localhost', help='Host to bind to')
    parser.add_argument('--port', type=int, default=8000, help='Port to listen on')
    args = parser.parse_args()

    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port)
