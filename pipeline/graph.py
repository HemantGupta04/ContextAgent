from langgraph.graph import END, StateGraph

from pipeline.edges import route_after_hallucination_check
from pipeline.nodes import (
    grader_node,
    generator_node,
    hallucination_checker_node,
    retriever_node,
)
from pipeline.state import AgentState


def build_graph():
    """
    Assembles and compiles the ContextAgent LangGraph pipeline.

    Graph structure:
        retriever → grader → generator → hallucination_checker
                                               ↓              ↓
                                        (hallucination)   (clean)
                                               ↓              ↓
                                           retriever        END
    """
    graph = StateGraph(AgentState)

    # ── Add nodes ──────────────────────────────────────────────────────────
    graph.add_node("retriever", retriever_node)
    graph.add_node("grader", grader_node)
    graph.add_node("generator", generator_node)
    graph.add_node("hallucination_checker", hallucination_checker_node)

    # ── Linear edges ───────────────────────────────────────────────────────
    graph.add_edge("retriever", "grader")
    graph.add_edge("grader", "generator")
    graph.add_edge("generator", "hallucination_checker")

    # ── Conditional edge (the loop) ────────────────────────────────────────
    graph.add_conditional_edges(
        source="hallucination_checker",
        path=route_after_hallucination_check,
        path_map={
            "retriever": "retriever",
            END: END,
        },
    )

    # ── Entry point ────────────────────────────────────────────────────────
    graph.set_entry_point("retriever")

    return graph.compile()


# Compiled app — imported by the FastAPI server
app = build_graph()
