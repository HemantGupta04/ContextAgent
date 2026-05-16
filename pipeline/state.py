"""
AgentState — the single source of truth passed between every node in the graph.

Using TypedDict (not a dataclass or Pydantic model) because LangGraph's
StateGraph requires a TypedDict-compatible schema for its state channels.
"""

from typing import List, Optional, TypedDict


class AgentState(TypedDict):
    # The original user query (never mutated)
    query: str

    # Raw chunks returned by the retriever node
    retrieved_chunks: List[dict]

    # Subset of retrieved_chunks that passed the relevance grader
    graded_chunks: List[dict]

    # Final answer produced by the generator
    answer: str

    # Set to True by the hallucination checker if claims can't be grounded
    hallucination_flag: bool

    # Rewritten query produced by the hallucination checker for the next loop
    refined_query: Optional[str]

    # How many times the retriever → generator loop has executed
    iterations: int
