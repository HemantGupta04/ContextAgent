from langgraph.graph import END

from pipeline.state import AgentState

MAX_ITERATIONS = 3


def route_after_hallucination_check(state: AgentState) -> str:
    """
    Routing function attached to the conditional edge leaving the
    hallucination checker node.

    Returns:
        "retriever"  — if hallucination found AND we haven't hit the limit
        END          — if answer is clean OR max iterations reached
    """
    hallucination = state.get("hallucination_flag", False)
    iterations = state.get("iterations", 0)

    if hallucination and iterations < MAX_ITERATIONS:
        print(
            f"\n[ROUTER] Hallucination detected (attempt {iterations}/{MAX_ITERATIONS})"
            f" → re-routing to retriever with refined query"
        )
        return "retriever"

    if hallucination and iterations >= MAX_ITERATIONS:
        print(f"\n[ROUTER] Max iterations ({MAX_ITERATIONS}) reached → ending pipeline")
    else:
        print(f"\n[ROUTER] Answer is clean → ending pipeline ✓")

    return END
