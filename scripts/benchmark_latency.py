"""Measure real interactive latency of the app's chat logic (no artificial throttling).

    python scripts/benchmark_latency.py

Asks 8 questions (including two follow-ups) through the same ChatEngine the app
uses, pausing between questions so Groq's per-minute token limit is never hit;
the pauses are NOT counted. Writes data/eval/app_latency.json.
"""

from __future__ import annotations

import json
import statistics
import sys
import time
from dataclasses import replace

from docuask.chat import ChatEngine
from docuask.config import PROJECT_ROOT, get_settings
from docuask.llm import get_llm
from docuask.rag import RAGPipeline
from docuask.retriever import build_retriever

# (question, is_follow_up_of_previous)
QUESTIONS = [
    ("How long does it take to port my number?", False),
    ("And if I'm switching from a landline to a cell phone?", True),
    ("What should I do about charges on my bill I never ordered?", False),
    ("Who do I complain to if the company won't fix it?", True),
    ("Should I call back a missed international number that rang once?", False),
    ("Can I text 911?", False),
    ("Is an eSIM safer than a normal SIM card?", False),
    ("What is the capital of Australia?", False),
]
PAUSE_S = 12


def main() -> int:
    s = get_settings()
    s = replace(s, llm=replace(s.llm, requests_per_minute=60))   # don't let the client limiter add waiting
    engine = ChatEngine(RAGPipeline(build_retriever(s), get_llm(s), top_k=s.retrieval.top_k))
    engine.pipeline.retriever.retrieve("warm-up", 1)
    rows, history = [], []
    for i, (q, follow) in enumerate(QUESTIONS, 1):
        if not follow:
            history = []
        turn = engine.ask(q, history)
        r = turn.response
        rows.append({"question": q, "follow_up": follow, "refused": r.refused,
                     "total_s": round(turn.total_ms / 1000, 3), "retrieval_ms": round(r.retrieval_ms, 1),
                     "generation_s": round(r.generation_ms / 1000, 3), "rewrite_s": round(turn.rewrite_ms / 1000, 3)})
        print(f"[{i}/{len(QUESTIONS)}] total {turn.total_ms / 1000:.2f}s (retrieval {r.retrieval_ms:.0f} ms, "
              f"generation {r.generation_ms / 1000:.2f}s, rewrite {turn.rewrite_ms / 1000:.2f}s) "
              f"{'REFUSED ' if r.refused else ''}{q}")
        if follow or turn.standalone_question != q:
            print(f"      rewritten as: {turn.standalone_question}")
        history = history + [{"role": "user", "content": q}, {"role": "assistant", "content": r.answer}]
        if i < len(QUESTIONS):
            time.sleep(PAUSE_S)

    tot = [r["total_s"] for r in rows]
    summary = {
        "n": len(rows),
        "median_total_s": round(statistics.median(tot), 2),
        "p90_total_s": round(sorted(tot)[int(0.9 * (len(tot) - 1))], 2),
        "max_total_s": round(max(tot), 2),
        "median_retrieval_ms": round(statistics.median(r["retrieval_ms"] for r in rows), 1),
        "median_generation_s": round(statistics.median(r["generation_s"] for r in rows), 2),
        "retriever": engine.pipeline.retriever.name, "llm": f"{s.llm.provider}/{s.llm.model}",
    }
    out = PROJECT_ROOT / "data" / "eval" / "app_latency.json"
    out.write_text(json.dumps({"summary": summary, "rows": rows}, indent=2), encoding="utf-8")
    print("\n===== LATENCY =====\n" + json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
