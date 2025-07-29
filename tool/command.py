import subprocess
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("command_executor")
DEFAULT_WORKSPACE = "D:/"

@mcp.tool()
async def run_command(command: str) -> str:
    try: 
        result = subprocess.run(command, shell=True, check=True, capture_output=True, text=True)
        return f"STDOUT: {result.stdout}\nSTDERR: {result.stderr}"
    except Exception as e:
        return f"An error occurred: {e}"


if __name__ == "__main__":
    mcp.run(transport='stdio')

''' host's config
{
"mcpServers": {
    "command_executor": {
        "command": "D:/python/Scripts/uv.exe",
        "args": [
            "--directory",
            "D:/Pha hust/Mechatronics Engineering/random code/random Py/AItraining/agent/tool",
            "run",
            "command.py"]}}
}
'''