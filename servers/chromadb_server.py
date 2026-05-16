"""
MCP Tool Server — ChromaDB

Exposes a single tool: query_vector_db
The LangGraph pipeline calls this by name and never imports ChromaDB directly.
Swapping to Pinecone means editing only this file — zero graph changes.
"""

import asyncio
import json
import os

import chromadb
from dotenv import load_dotenv
from mcp.server import Server
from mcp.server.stdio import stdio_server

load_dotenv()

# ── ChromaDB setup ──────────────────────────────────────────────────────────
PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "context_agent_docs")

client = chromadb.PersistentClient(path=PERSIST_DIR)
collection = client.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

# ── MCP Server ──────────────────────────────────────────────────────────────
app = Server("chromadb-server")


@app.list_tools()
async def list_tools():
    return [
        {
            "name": "query_vector_db",
            "description": (
                "Search the ChromaDB vector database for the most relevant "
                "document chunks given a query string."
            ),
            "inputSchema": {
                "type": "object",
                "properties": {
                    "query": {
                        "type": "string",
                        "description": "The search query",
                    },
                    "k": {
                        "type": "integer",
                        "description": "Number of chunks to return",
                        "default": 10,
                    },
                },
                "required": ["query"],
            },
        }
    ]


@app.call_tool()
async def call_tool(name: str, arguments: dict):
    if name != "query_vector_db":
        raise ValueError(f"Unknown tool: {name}")

    query = arguments["query"]
    k = arguments.get("k", 10)

    results = collection.query(query_texts=[query], n_results=k)

    chunks = []
    for i, doc in enumerate(results["documents"][0]):
        chunks.append(
            {
                "text": doc,
                "metadata": results["metadatas"][0][i],
                "distance": results["distances"][0][i],
            }
        )

    return [{"type": "text", "text": json.dumps(chunks)}]


async def main():
    async with stdio_server() as streams:
        await app.run(*streams, app.create_initialization_options())


if __name__ == "__main__":
    asyncio.run(main())
