# ContextAgent

**Self-correcting RAG pipeline built with LangGraph + MCP**

ContextAgent solves the *irrelevant chunk problem* in vanilla RAG — where an LLM generates a confident but hallucinated answer because the retrieved documents weren't actually relevant. It wraps every retrieval-generation cycle in a grade → verify loop that won't return an answer until it can stand behind it.

---

## How it works

```
User Query
    │
    ▼
┌──────────────┐
│  Retriever   │  ← calls ChromaDB MCP server via stdio
└──────┬───────┘
       │
       ▼
┌──────────────┐
│    Grader    │  ← Gemini Flash, parallel relevance checks per chunk
└──────┬───────┘
       │  (only relevant chunks pass through)
       ▼
┌──────────────┐
│  Generator   │  ← Gemini Pro, grounded answer from graded chunks
└──────┬───────┘
       │
       ▼
┌──────────────────────┐
│ Hallucination Checker │  ← Gemini Flash, verifies claims against sources
└──────┬───────────────┘
       │
  hallucination?
  ┌────┴─────────────────┐
  │ YES (< 3 attempts)   │ NO → return answer ✓
  │ refine query         │
  └─────────────────────►│
        loop back to Retriever
```

Each tool call (ChromaDB, Tavily, filesystem) goes through an **MCP server** over stdio — the pipeline never imports database or search libraries directly. Swapping ChromaDB for Pinecone means editing one file.

---

## Features

- **Corrective RAG loop** — re-queries with a refined question when hallucination is detected (up to 3 iterations)
- **Relevance grading** — filters out irrelevant chunks before generation, reducing noise
- **MCP-based tool layer** — ChromaDB, Tavily web search, and local filesystem as pluggable MCP servers
- **LangGraph state machine** — explicit nodes, edges, and conditional routing; fully inspectable graph
- **FastAPI interface** — `/query` and `/ingest` REST endpoints
- **Idempotent ingestion** — MD5-based chunk IDs; re-running ingestion on the same file is safe

---

## Project structure

```
ContextAgent/
├── pipeline/
│   ├── state.py             # AgentState TypedDict
│   ├── nodes.py             # retriever, grader, generator, hallucination_checker
│   ├── edges.py             # routing logic + MAX_ITERATIONS guard
│   └── graph.py             # StateGraph assembly + compile
├── servers/
│   ├── chromadb_server.py   # MCP server — query_vector_db tool
│   ├── tavily_server.py     # MCP server — web_search tool
│   └── filesystem_server.py # MCP server — read_file tool
├── api/
│   ├── main.py              # FastAPI app (/query, /ingest, /health)
│   └── models.py            # Pydantic request/response models
├── ingestion/
│   ├── chunker.py           # RecursiveCharacterTextSplitter wrapper
│   ├── embedder.py          # Google text-embedding-004
│   └── ingest.py            # Full ingestion pipeline + CLI
├── evaluation/
│   ├── eval_set.json        # 5 test queries with expected keywords
│   └── evaluate.py          # Keyword recall + latency metrics
├── .env.example
├── requirements.txt
└── README.md
```

---

## Setup

### 1. Clone and install

```bash
git clone https://github.com/HemantGupta04/ContextAgent.git
cd ContextAgent
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Configure environment

```bash
cp .env.example .env
# Edit .env and add your keys:
#   GOOGLE_API_KEY=...
#   TAVILY_API_KEY=...
```

### 3. Ingest your documents

```bash
python -m ingestion.ingest --files path/to/doc1.txt path/to/doc2.pdf
```

### 4. Start the API

```bash
uvicorn api.main:fastapi_app --host 0.0.0.0 --port 8000 --reload
```

### 5. Query

```bash
curl -X POST http://localhost:8000/query \
  -H "Content-Type: application/json" \
  -d '{"query": "What is the main architecture of ContextAgent?"}'
```

Response:
```json
{
  "query": "What is the main architecture of ContextAgent?",
  "answer": "ContextAgent uses a LangGraph StateGraph with four nodes...",
  "sources": [...],
  "iterations": 1,
  "hallucination_detected": false
}
```

---

## Evaluation

```bash
python -m evaluation.evaluate
```

Output:
```
============================================================
EVALUATION SUMMARY  (5 cases)
============================================================
  Answered          : 5/5
  Avg keyword recall: 0.84
  Avg iterations    : 1.2
============================================================
```

---

## Key design decisions

**Why LangGraph instead of plain LangChain?**
The corrective loop requires state that persists across nodes — query, chunks, answer, hallucination flag, iteration count. LangGraph's `StateGraph` + `TypedDict` state makes this explicit and inspectable. Adding a new node (e.g., query expansion) is one `add_node` call.

**Why MCP servers over direct imports?**
The pipeline calls tools by name over a standard JSON-RPC 2.0 interface. This means:
- Swap ChromaDB → Pinecone by editing `servers/chromadb_server.py` only
- The graph nodes have zero knowledge of which vector DB is running
- Each server can be tested and mocked independently

**Why parallel grading?**
`grader_node` calls Gemini Flash once per chunk via `asyncio.gather()`. With 10 chunks, this runs ~10 LLM calls concurrently instead of sequentially — grading time stays flat regardless of chunk count.

**Why Gemini Flash for grading/checking, Pro for generation?**
Flash is ~10× cheaper and fast enough for binary yes/no decisions. Pro's higher quality only matters for the final answer. This keeps latency and cost low without sacrificing answer quality.

---

## Inspiration

- [LangChain CRAG notebook](https://github.com/langchain-ai/langgraph/blob/main/examples/rag/langgraph_crag.ipynb) — Corrective RAG reference implementation
- [Self-RAG paper](https://arxiv.org/abs/2310.11511) — retrieve, critique, generate, repeat
- [Adaptive RAG](https://arxiv.org/abs/2403.14403) — routing based on query complexity

---

## Tech stack

| Component | Library |
|---|---|
| Agent framework | LangGraph 0.2 |
| LLM | Google Gemini 1.5 Pro / Flash |
| Embeddings | Google text-embedding-004 |
| Vector DB | ChromaDB |
| Tool protocol | MCP (Model Context Protocol) |
| Web search | Tavily |
| API | FastAPI + uvicorn |
| Text splitting | LangChain RecursiveCharacterTextSplitter |

---

Built by [Hemant Gupta](https://github.com/HemantGupta04) · Final year CS, NSUT Delhi
