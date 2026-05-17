import json
import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

import chromadb
from dotenv import load_dotenv
from mcp.server.fastmcp import FastMCP

from ingestion.embedder import embed_texts

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "context_agent_docs")

_chroma = chromadb.PersistentClient(path=PERSIST_DIR)
_collection = _chroma.get_or_create_collection(
    name=COLLECTION_NAME,
    metadata={"hnsw:space": "cosine"},
)

mcp = FastMCP("chromadb-server")


@mcp.tool()
def query_vector_db(query: str, k: int = 10) -> str:
    """Search the ChromaDB vector database for the most relevant document chunks."""
    query_embedding = embed_texts([query])[0]
    results = _collection.query(query_embeddings=[query_embedding], n_results=k)

    chunks = [
        {
            "text": doc,
            "metadata": results["metadatas"][0][i],
            "distance": results["distances"][0][i],
        }
        for i, doc in enumerate(results["documents"][0])
    ]
    return json.dumps(chunks)


if __name__ == "__main__":
    mcp.run(transport="stdio")
