"""
Pydantic models for the ContextAgent FastAPI layer.
"""

from typing import List, Optional

from pydantic import BaseModel, Field


# ── Request models ─────────────────────────────────────────────────────────────

class QueryRequest(BaseModel):
    query: str = Field(..., description="The user's natural-language question")
    k: int = Field(10, description="Number of chunks to retrieve from the vector DB")


class IngestRequest(BaseModel):
    file_paths: List[str] = Field(
        ..., description="Absolute paths to local files to ingest into ChromaDB"
    )
    chunk_size: int = Field(500, description="Token/character size of each chunk")
    chunk_overlap: int = Field(50, description="Overlap between consecutive chunks")


# ── Response models ────────────────────────────────────────────────────────────

class SourceChunk(BaseModel):
    text: str
    metadata: dict
    distance: float


class QueryResponse(BaseModel):
    query: str
    answer: str
    sources: List[SourceChunk]
    iterations: int
    hallucination_detected: bool


class IngestResponse(BaseModel):
    status: str
    files_processed: int
    chunks_added: int
    message: Optional[str] = None
