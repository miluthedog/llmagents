import subprocess
import random
from mcp.server.fastmcp import FastMCP

mcp = FastMCP("find_one")
DEFAULT_WORKSPACE = "D:/"

@mcp.tool()
async def return_one(command: str) -> str:
    result = random.randint(1, 3)
    return f"result: {result}"

if __name__ == "__main__":
    mcp.run(transport='stdio')

''' host's config
{
"mcpServers": {
    "find_one": {
        "command": "D:/python/Scripts/uv.exe",
        "args": [
            "--directory",
            "D:/Pha hust/Mechatronics Engineering/random code/random Py/AItraining/agent",
            "run",
            "toolreturn1.py"]}}
}
'''