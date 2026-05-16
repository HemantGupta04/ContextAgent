"""
LangGraph nodes for the ContextAgent pipeline.

Each node is a plain async function: (AgentState) -> dict
The returned dict is merged into the state by LangGraph.

Tool calls go through MCP servers over stdio — the nodes never import
chromadb, tavily, or filesystem libraries directly.
"""

import asyncio
import json
import os

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from mcp import ClientSession, StdioServerParameters
from mcp.client.stdio import stdio_client

from pipeline.state import AgentState

load_dotenv()

# ── Model initialisation ───────────────────────────────────────────────────────

_flash = ChatGoogleGenerativeAI(
    model="gemini-1.5-flash",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0,
)

_pro = ChatGoogleGenerativeAI(
    model="gemini-1.5-pro",
    google_api_key=os.getenv("GOOGLE_API_KEY"),
    temperature=0.2,
)

# ── MCP helper ─────────────────────────────────────────────────────────────────

async def _call_mcp_tool(server_script: str, tool_name: str, args: dict) -> str:
    """
    Spawn an MCP stdio server, call one tool, return its text result, shut down.
    """
    params = StdioServerParameters(command="python", args=[server_script])
    async with stdio_client(params) as (read, write):
        async with ClientSession(read, write) as session:
            await session.initialize()
            result = await session.call_tool(tool_name, args)
            return result.content[0].text


# ── Node implementations ───────────────────────────────────────────────────────

async def retriever_node(state: AgentState) -> dict:
    """
    Calls the ChromaDB MCP server to retrieve the top-k relevant chunks.
    On a re-route (after hallucination), uses the refined_query if present.
    Falls back to Tavily web search when ChromaDB returns nothing.
    """
    query = state.get("refined_query") or state["query"]
    print(f"\n[RETRIEVER] Query: {query!r}")

    server = os.path.join(os.path.dirname(__file__), "..", "servers", "chromadb_server.py")
    raw = await _call_mcp_tool(server, "query_vector_db", {"query": query, "k": 10})
    chunks = json.loads(raw)

    # Fallback: if vector DB is empty or returns nothing, hit the web
    if not chunks:
        print("[RETRIEVER] ChromaDB returned 0 results — falling back to Tavily web search")
        tavily_server = os.path.join(
            os.path.dirname(__file__), "..", "servers", "tavily_server.py"
        )
        raw_web = await _call_mcp_tool(
            tavily_server, "web_search", {"query": query, "max_results": 5}
        )
        web_results = json.loads(raw_web)
        chunks = [
            {
                "text": r.get("content", r.get("snippet", "")),
                "metadata": {"source": r.get("url", "web"), "title": r.get("title", "")},
                "distance": 0.0,
            }
            for r in web_results
        ]

    print(f"[RETRIEVER] Retrieved {len(chunks)} chunks")
    return {"retrieved_chunks": chunks, "refined_query": None}


async def _grade_single_chunk(chunk: dict, query: str) -> bool:
    """Binary relevance check for one chunk using Gemini Flash."""
    prompt = (
        f"Is the following document chunk relevant to the question?\n\n"
        f"Question: {query}\n\n"
        f"Chunk: {chunk['text'][:800]}\n\n"
        f"Answer with only 'yes' or 'no'."
    )
    response = await _flash.ainvoke(prompt)
    return response.content.strip().lower().startswith("yes")


async def grader_node(state: AgentState) -> dict:
    """
    Runs relevance checks on all retrieved chunks in parallel (asyncio.gather).
    Only chunks that pass are forwarded to the generator.
    """
    query = state["query"]
    chunks = state["retrieved_chunks"]
    print(f"\n[GRADER] Grading {len(chunks)} chunks …")

    results = await asyncio.gather(
        *[_grade_single_chunk(chunk, query) for chunk in chunks]
    )

    graded = [chunk for chunk, passed in zip(chunks, results) if passed]
    print(f"[GRADER] {len(graded)}/{len(chunks)} chunks passed relevance check")
    return {"graded_chunks": graded}


async def generator_node(state: AgentState) -> dict:
    """
    Generates the final answer using Gemini Pro, grounded in the graded chunks.
    """
    query = state["query"]
    chunks = state["graded_chunks"]
    print(f"\n[GENERATOR] Generating answer from {len(chunks)} graded chunks …")

    if not chunks:
        fallback = (
            "I could not find relevant information in the available documents "
            "to answer your question confidently."
        )
        return {"answer": fallback}

    context = "\n\n---\n\n".join(
        f"[Source {i+1}]: {c['text']}" for i, c in enumerate(chunks)
    )
    prompt = (
        f"You are a helpful assistant. Answer the following question using ONLY "
        f"the provided context. Do not add information not present in the context.\n\n"
        f"Context:\n{context}\n\n"
        f"Question: {query}\n\n"
        f"Answer:"
    )

    response = await _pro.ainvoke(prompt)
    answer = response.content.strip()
    print(f"[GENERATOR] Answer generated ({len(answer)} chars)")
    return {"answer": answer}


async def hallucination_checker_node(state: AgentState) -> dict:
    """
    Verifies the generated answer against the source chunks using Gemini Flash.
    If hallucination is detected, generates a refined query for the next loop.
    Increments the iteration counter each time it runs.
    """
    answer = state["answer"]
    chunks = state["graded_chunks"]
    query = state["query"]
    iterations = state.get("iterations", 0) + 1

    print(f"\n[HALLUCINATION CHECKER] Iteration {iterations} — verifying answer …")

    if not chunks:
        # No sources to check against — can't verify, pass through
        return {"hallucination_flag": False, "iterations": iterations}

    context = "\n\n---\n\n".join(f"[Source {i+1}]: {c['text']}" for i, c in enumerate(chunks))
    prompt = (
        f"Does the following answer contain any claims NOT supported by the provided sources?\n\n"
        f"Sources:\n{context}\n\n"
        f"Answer: {answer}\n\n"
        f"Respond with 'yes' if there are unsupported claims, 'no' if everything is grounded."
    )

    response = await _flash.ainvoke(prompt)
    hallucination = response.content.strip().lower().startswith("yes")

    refined_query = None
    if hallucination:
        # Ask the model to rewrite the query to be more specific
        refine_prompt = (
            f"The answer to the question '{query}' contained hallucinations. "
            f"Rewrite the question to be more specific and targeted so a vector "
            f"search is more likely to find grounding evidence. "
            f"Return only the rewritten question, nothing else."
        )
        refine_resp = await _flash.ainvoke(refine_prompt)
        refined_query = refine_resp.content.strip()
        print(f"[HALLUCINATION CHECKER] Hallucination detected. Refined query: {refined_query!r}")
    else:
        print("[HALLUCINATION CHECKER] Answer is grounded ✓")

    return {
        "hallucination_flag": hallucination,
        "refined_query": refined_query,
        "iterations": iterations,
    }
