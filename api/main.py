"""
ContextAgent — FastAPI entry point.

Endpoints:
  POST /query   — run a query through the LangGraph RAG pipeline
  POST /ingest  — chunk and embed documents into ChromaDB
  GET  /health  — liveness check
"""

import os
import sys
import time

from fastapi import FastAPI, HTTPException
from fastapi.middleware.cors import CORSMiddleware

# Ensure project root is on sys.path when running from any directory
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from api.models import (
    IngestRequest,
    IngestResponse,
    QueryRequest,
    QueryResponse,
    SourceChunk,
)
from ingestion.ingest import ingest_files
from pipeline.graph import app as langgraph_app

# ── App setup ─────────────────────────────────────────────────────────────────

fastapi_app = FastAPI(
    title="ContextAgent",
    description=(
        "Self-correcting RAG pipeline built with LangGraph + MCP. "
        "Retrieves, grades, generates, and hallucination-checks answers "
        "over your own documents."
    ),
    version="1.0.0",
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)


# ── Routes ────────────────────────────────────────────────────────────────────

@fastapi_app.get("/health")
async def health():
    return {"status": "ok", "timestamp": time.time()}


@fastapi_app.post("/query", response_model=QueryResponse)
async def query(request: QueryRequest):
    """
    Run a user query through the full ContextAgent pipeline:
      retriever → grader → generator → hallucination_checker (→ loop if needed)
    """
    try:
        initial_state = {
            "query": request.query,
            "retrieved_chunks": [],
            "graded_chunks": [],
            "answer": "",
            "hallucination_flag": False,
            "refined_query": None,
            "iterations": 0,
        }

        final_state = await langgraph_app.ainvoke(initial_state)

        sources = [
            SourceChunk(
                text=chunk.get("text", ""),
                metadata=chunk.get("metadata", {}),
                distance=chunk.get("distance", 0.0),
            )
            for chunk in final_state.get("graded_chunks", [])
        ]

        return QueryResponse(
            query=request.query,
            answer=final_state.get("answer", ""),
            sources=sources,
            iterations=final_state.get("iterations", 0),
            hallucination_detected=final_state.get("hallucination_flag", False),
        )

    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


@fastapi_app.post("/ingest", response_model=IngestResponse)
async def ingest(request: IngestRequest):
    """
    Chunk and embed local files, then upsert them into ChromaDB.
    """
    try:
        result = await ingest_files(
            file_paths=request.file_paths,
            chunk_size=request.chunk_size,
            chunk_overlap=request.chunk_overlap,
        )
        return IngestResponse(
            status="success",
            files_processed=result["files_processed"],
            chunks_added=result["chunks_added"],
        )
    except Exception as exc:
        raise HTTPException(status_code=500, detail=str(exc)) from exc


# ── Dev runner ────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("api.main:fastapi_app", host="0.0.0.0", port=8000, reload=True)
