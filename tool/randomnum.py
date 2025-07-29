import random
import logging
from mcp.server.fastmcp import FastMCP


logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)

mcp = FastMCP("number_generator")
DEFAULT_WORKSPACE = "D:/"

@mcp.tool()
async def random_number(command: str) -> str:
    result = random.randint(1, 3)
    log.info(f"[TOOL] Received command: {command} → Generated number: {result}")
    return {"number": result}


if __name__ == "__main__":
    mcp.run(transport='stdio')

''' host's config
{
"mcpServers": {
    "number_generator": {
        "command": "D:/python/Scripts/uv.exe",
        "args": [
            "--directory",
            "D:/Pha hust/Mechatronics Engineering/random code/random Py/AItraining/agent/tool",
            "run",
            "randomnum.py"]}}
}
'''