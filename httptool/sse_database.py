import os
from mcp.server.fastmcp import FastMCP
from mcp.server import Server
from mcp.server.sse import SseServerTransport
import aiosqlite
from starlette.applications import Starlette
from starlette.routing import Route, Mount
from starlette.requests import Request
import uvicorn
import argparse


mcp = FastMCP("terminal")
DEFAULT_WORKSPACE = os.path.expanduser(".")


@mcp.tool()
async def vimes_lab_members(name: str) -> str:
    """
    Query Vimes Lab members by partial name (case-insensitive).
    If name is empty, return all members.

    Args:
        name (str): Partial name to search for (can be empty)

    Returns:
        str: Matching members' names
    """
    try:
        async with aiosqlite.connect("db/vimes.db") as db:
            db.row_factory = aiosqlite.Row

            cursor = await db.execute(
                "SELECT name FROM members WHERE name LIKE ? COLLATE NOCASE",(f"%{name}%",)
            )

            rows = await cursor.fetchall()
            if rows:
                return "\n".join(r["name"] for r in rows)
            else:
                return "No matching member found."
    except Exception as e:
        return f"Database error: {e}"

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
    parser.add_argument('--port', type=int, default=8001, help='Port to listen on')
    args = parser.parse_args()

    starlette_app = create_starlette_app(mcp_server, debug=True)
    uvicorn.run(starlette_app, host=args.host, port=args.port)
