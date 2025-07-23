import subprocess
from mcp.server.fastmcp import FastMCP


mcp = FastMCP("terminal")
DEFAULT_WORKSPACE = "D:/"

# Allow host to run "command" in shell (cwd)
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
    "terminal": {
        "command": "D:/python/Scripts/uv.exe",
        "args": [
            "--directory",
            "D:/Pha hust/Mechatronics Engineering/random code/random Py/AItraining/agent",
            "run",
            "toolTerminal.py"]}}
}
'''