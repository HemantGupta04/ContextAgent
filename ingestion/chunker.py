"""
Text chunking utilities for the ingestion pipeline.

Uses LangChain's RecursiveCharacterTextSplitter so chunks respect
sentence / paragraph boundaries instead of splitting mid-word.
"""

from typing import List

from langchain.text_splitter import RecursiveCharacterTextSplitter


def chunk_text(
    text: str,
    chunk_size: int = 500,
    chunk_overlap: int = 50,
    source_metadata: dict | None = None,
) -> List[dict]:
    """
    Split `text` into overlapping chunks.

    Returns:
        List of dicts: {"text": str, "metadata": dict}
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )

    chunks = splitter.split_text(text)
    meta = source_metadata or {}

    return [
        {"text": chunk, "metadata": {**meta, "chunk_index": i}}
        for i, chunk in enumerate(chunks)
    ]
