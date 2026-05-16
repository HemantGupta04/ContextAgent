"""
Lightweight evaluation script for ContextAgent.

Runs each query in eval_set.json through the pipeline and checks:
  - Keyword recall  : fraction of expected keywords present in the answer
  - Answered        : whether a non-empty answer was returned
  - Iteration count : how many self-correction loops were needed
  - Latency         : wall-clock time per query

Usage:
    python -m evaluation.evaluate
    python -m evaluation.evaluate --eval-file evaluation/eval_set.json --output results.json
"""

import asyncio
import json
import os
import sys
import time
from typing import List

sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from pipeline.graph import app as langgraph_app

DEFAULT_EVAL_FILE = os.path.join(os.path.dirname(__file__), "eval_set.json")


async def run_eval(eval_cases: List[dict]) -> List[dict]:
    results = []

    for case in eval_cases:
        qid = case["id"]
        query = case["query"]
        expected_kw = [kw.lower() for kw in case.get("expected_keywords", [])]

        print(f"\n[EVAL] Running {qid}: {query!r}")
        t0 = time.time()

        state = {
            "query": query,
            "retrieved_chunks": [],
            "graded_chunks": [],
            "answer": "",
            "hallucination_flag": False,
            "refined_query": None,
            "iterations": 0,
        }

        try:
            final = await langgraph_app.ainvoke(state)
            answer = final.get("answer", "")
            iterations = final.get("iterations", 0)
            error = None
        except Exception as exc:
            answer = ""
            iterations = -1
            error = str(exc)

        latency = round(time.time() - t0, 2)
        answer_lower = answer.lower()
        hits = sum(1 for kw in expected_kw if kw in answer_lower)
        keyword_recall = round(hits / len(expected_kw), 2) if expected_kw else 1.0

        result = {
            "id": qid,
            "query": query,
            "answered": bool(answer),
            "keyword_recall": keyword_recall,
            "iterations": iterations,
            "latency_s": latency,
            "answer_snippet": answer[:200] if answer else "",
            "error": error,
        }
        results.append(result)
        print(
            f"  ✓ keyword_recall={keyword_recall} | iterations={iterations} | {latency}s"
        )

    return results


def print_summary(results: List[dict]):
    total = len(results)
    answered = sum(1 for r in results if r["answered"])
    avg_recall = round(sum(r["keyword_recall"] for r in results) / total, 3)
    valid_iters = [r["iterations"] for r in results if r["iterations"] >= 0]
    avg_iters = round(sum(valid_iters) / len(valid_iters), 2) if valid_iters else 0

    print("\n" + "=" * 60)
    print(f"EVALUATION SUMMARY  ({total} cases)")
    print("=" * 60)
    print(f"  Answered          : {answered}/{total}")
    print(f"  Avg keyword recall: {avg_recall}")
    print(f"  Avg iterations    : {avg_iters}")
    print("=" * 60)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser()
    parser.add_argument("--eval-file", default=DEFAULT_EVAL_FILE)
    parser.add_argument("--output", default=None, help="Save results to JSON file")
    args = parser.parse_args()

    with open(args.eval_file) as f:
        cases = json.load(f)

    results = asyncio.run(run_eval(cases))
    print_summary(results)

    if args.output:
        with open(args.output, "w") as f:
            json.dump(results, f, indent=2)
        print(f"\nResults saved to {args.output}")
