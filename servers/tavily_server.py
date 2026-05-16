"""
MCP Tool Server — Tavily Web Search

Exposes a single tool: web_search
Used as fallback when ChromaDB does not have relevant chunks.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server
from tavily import TavilyClient

load_dotenv()

tavily = TavilyClient(api_key=os.getenv("TAVILY_API_KEY"))

app = Server("tavily-server")


@app.list_tools()
async def list_tools():
    return [
        {
            "name": "web_search",
            "description": "Search the web for up-to-date information on a query.",
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "max_results": {
                        "type": "integer",
                        "description": "Number of results to return",
                        "default": 5,
                    },
                },
                "required": ["query"],
            },
        }
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "web_search":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]
    max_results = arguments.get("max_results", 5)

    results = tavily.search(
        query=query,
        max_results=max_results,
        search_depth="advanced",
    )

    return [{"type": "text", "text": json.dumps(results["results"])}]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
