"""
Ingestion pipeline — reads files, chunks them, embeds, and upserts to ChromaDB.

Called by the FastAPI /ingest endpoint or directly via CLI:
    python -m ingestion.ingest --files doc1.txt doc2.pdf
"""

import asyncio
import hashlib
import os
import sys
from typing import List

import chromadb
from dotenv import load_dotenv

from ingestion.chunker import chunk_text
from ingestion.embedder import embed_texts

load_dotenv()

PERSIST_DIR = os.getenv("CHROMA_PERSIST_DIR", "./chroma_db")
COLLECTION_NAME = os.getenv("CHROMA_COLLECTION_NAME", "context_agent_docs")


def _read_file(path: str) -> str:
    """Read a text file. Extend here for PDF/DOCX support."""
    _, ext = os.path.splitext(path.lower())

    if ext == ".pdf":
        try:
            import pdfplumber  # optional dependency

            with pdfplumber.open(path) as pdf:
                return "\n".join(page.extract_text() or "" for page in pdf.pages)
        except ImportError:
            raise RuntimeError(
                "pdfplumber is required for PDF ingestion: pip install pdfplumber"
            )

    # Default: plain text
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def _stable_id(source: str, chunk_index: int) -> str:
    """Deterministic chunk ID so re-ingestion is idempotent."""
    raw = f"{source}::{chunk_index}"
    return hashlib.md5(raw.encode()).hexdigest()


async def ingest_files(
    file_paths: List[str],
    chunk_size: int = 500,
    chunk_overlap: int = 50,
) -> dict:
    """
    Main ingestion coroutine.

    1. Read each file
    2. Chunk with RecursiveCharacterTextSplitter
    3. Embed with Google text-embedding-004
    4. Upsert into ChromaDB (idempotent via stable MD5 IDs)

    Returns:
        {"files_processed": int, "chunks_added": int}
    """
    client = chromadb.PersistentClient(path=PERSIST_DIR)
    collection = client.get_or_create_collection(
        name=COLLECTION_NAME,
        metadata={"hnsw:space": "cosine"},
    )

    total_chunks = 0

    for path in file_paths:
        if not os.path.isfile(path):
            print(f"[INGEST] Skipping {path!r} — file not found")
            continue

        print(f"[INGEST] Processing {path!r} …")
        text = _read_file(path)
        source_meta = {"source": os.path.basename(path), "path": path}

        chunks = chunk_text(
            text,
            chunk_size=chunk_size,
            chunk_overlap=chunk_overlap,
            source_metadata=source_meta,
        )

        texts = [c["text"] for c in chunks]
        metadatas = [c["metadata"] for c in chunks]
        ids = [_stable_id(path, c["metadata"]["chunk_index"]) for c in chunks]

        # Embed in one batched call
        embeddings = embed_texts(texts)

        collection.upsert(
            documents=texts,
            embeddings=embeddings,
            metadatas=metadatas,
            ids=ids,
        )

        total_chunks += len(chunks)
        print(f"[INGEST] ✓ {len(chunks)} chunks upserted from {os.path.basename(path)}")

    return {"files_processed": len(file_paths), "chunks_added": total_chunks}


# ── CLI entry point ────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Ingest documents into ChromaDB")
    parser.add_argument("--files", nargs="+", required=True, help="Paths to files")
    parser.add_argument("--chunk-size", type=int, default=500)
    parser.add_argument("--chunk-overlap", type=int, default=50)
    args = parser.parse_args()

    result = asyncio.run(
        ingest_files(
            file_paths=args.files,
            chunk_size=args.chunk_size,
            chunk_overlap=args.chunk_overlap,
        )
    )
    print(f"\n✅ Done — {result['chunks_added']} chunks from {result['files_processed']} files")
