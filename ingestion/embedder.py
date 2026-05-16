"""
Embedding utilities for the ingestion pipeline.

Uses Google's text-embedding-004 model via langchain-google-genai.
Swap to OpenAI / HuggingFace by editing only this file.
"""

import os
from typing import List

from dotenv import load_dotenv
from langchain_google_genai import GoogleGenerativeAIEmbeddings

load_dotenv()

_embedder = GoogleGenerativeAIEmbeddings(
    model="models/text-embedding-004",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    """
    Return a list of embedding vectors for the given texts.
    Batches automatically via the LangChain wrapper.
    """
    return _embedder.embed_documents(texts)
