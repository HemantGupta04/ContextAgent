"""
MCP Tool Server — Local Filesystem

Exposes a single tool: read_file
Lets the pipeline read local documents without importing filesystem logic
directly. Swap to S3 / GCS by editing only this file.
"""

import asyncio
import os

from mcp.server import Server
from mcp.server.stdio import stdio_server

app = Server("filesystem-server")

# Restrict access to an allowed base directory (safety guard)
BASE_DIR = os.getenv("FS_BASE_DIR", ".")


@app.list_tools()
async def list_tools():
    return [
        {
            "name": "read_file",
            "description": (
                "Read the full text content of a local file by path. "
                "Returns the raw text string."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "path": {
                        "type": "string",
                        "description": "Absolute or relative path to the file",
                    }
                },
                "required": ["path"],
            },
        }
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "read_file":
        raise ValueError(f"Unknown tool: {name}")

    path = arguments["path"]

    # Resolve and safety-check the path
    resolved = os.path.realpath(path)
    base = os.path.realpath(BASE_DIR)
    if not resolved.startswith(base):
        raise PermissionError(
            f"Access denied: {path!r} is outside the allowed base directory."
        )

    if not os.path.isfile(resolved):
        raise FileNotFoundError(f"File not found: {path!r}")

    with open(resolved, "r", encoding="utf-8", errors="replace") as f:
        content = f.read()

    return [{"type": "text", "text": content}]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
