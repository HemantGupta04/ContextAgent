import os
from typing import List

import requests
from dotenv import load_dotenv

load_dotenv()

_API_KEY = os.getenv("GOOGLE_API_KEY")
_BASE_URL = (
    "https://generativelanguage.googleapis.com"
    "/v1beta/models/gemini-embedding-2:embedContent"
)


def embed_texts(texts: List[str]) -> List[List[float]]:
    session = requests.Session()
    embeddings = []
    for text in texts:
        payload = {
            "model": "models/gemini-embedding-2",
            "content": {"parts": [{"text": text}]},
            "taskType": "RETRIEVAL_DOCUMENT",
        }
        resp = session.post(_BASE_URL, json=payload, params={"key": _API_KEY})
        resp.raise_for_status()
        embeddings.append(resp.json()["embedding"]["values"])
    return embeddings
