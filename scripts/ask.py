"""Ask DocuAsk one or more questions from the command line.

    python scripts/ask.py "How do I keep my number when I switch providers?"
    python scripts/ask.py --demo            # runs the built-in demo questions
    python scripts/ask.py --demo --show-passages
"""

from __future__ import annotations

import argparse
import json
import sys

from docuask.config import get_settings
from docuask.llm import get_llm
from docuask.prompts import PROMPT_VERSION
from docuask.rag import RAGPipeline
from docuask.retriever import build_retriever

# (question, kind). Kinds: answerable / partial / unanswerable. Expected behaviour is
# described in the README; the real scored test set comes in Phase 5.
DEMO = [
    ("How do I keep my phone number when I switch to a new provider?", "answerable"),
    ("What is cramming and how can I protect myself from it?", "answerable"),
    ("I got a call from an unknown international number that rang once. Should I call back?", "answerable"),
    ("Can I text 911, and what happens if text-to-911 is not available?", "answerable"),
    ("How do I file a slamming complaint, and what is the phone number of the Texas public utility commission?", "partial"),
    ("What is Vodafone Egypt's roaming price per megabyte?", "unanswerable"),
    ("Who won the 2022 FIFA World Cup?", "unanswerable"),
]


def print_response(r, show_passages: bool) -> None:
    print(f"\nQ: {r.question}")
    print(f"A: {r.answer}")
    flag = "REFUSED" if r.refused else f"{len(r.citations)} citation(s)"
    print(f"   [{flag}; invalid citations: {r.invalid_citations or 'none'}; "
          f"retrieval {r.retrieval_ms:.0f} ms, generation {r.generation_ms:.0f} ms, total {r.total_ms:.0f} ms]")
    if show_passages:
        for i, h in enumerate(r.hits, 1):
            used = "*" if h.chunk.citation in r.citations else " "
            print(f"   {used}{i}. {h.score:.3f} {h.chunk.citation} {h.chunk.text[:90].replace(chr(10), ' ')}...")


def main() -> int:
    s = get_settings()
    p = argparse.ArgumentParser()
    p.add_argument("question", nargs="?")
    p.add_argument("--demo", action="store_true")
    p.add_argument("--show-passages", action="store_true")
    p.add_argument("--k", type=int, default=s.retrieval.top_k)
    args = p.parse_args()
    if not args.question and not args.demo:
        p.error("give a question or --demo")

    pipeline = RAGPipeline(build_retriever(s), get_llm(s), top_k=args.k, min_score=s.retrieval.min_score)
    print(f"LLM: {s.llm.provider}/{s.llm.model}  retriever: {pipeline.retriever.name}  "
          f"k={args.k}  prompt: {PROMPT_VERSION}")

    items = DEMO if args.demo else [(args.question, "manual")]
    rows = []
    for q, kind in items:
        r = pipeline.answer(q)
        print_response(r, args.show_passages)
        rows.append({"kind": kind, "refused": r.refused, "citations": len(r.citations),
                     "invalid": len(r.invalid_citations), "total_ms": round(r.total_ms)})

    if args.demo:
        print("\n===== DEMO SUMMARY =====")
        print(f"{'kind':13s} {'refused':8s} {'cites':6s} {'invalid':8s} {'ms':>6s}  question")
        for (q, _), row in zip(items, rows):
            print(f"{row['kind']:13s} {str(row['refused']):8s} {row['citations']:<6d} {row['invalid']:<8d} "
                  f"{row['total_ms']:>6d}  {q[:60]}")
        answered = [r for r in rows if not r["refused"]]
        print(f"\nAnswered {len(answered)}, refused {len(rows) - len(answered)}; "
              f"answers with 0 citations: {sum(r['citations'] == 0 for r in answered)}; "
              f"invalid citations: {sum(r['invalid'] for r in rows)}")
        out = s.paths.processed_dir / f"phase4_demo_{PROMPT_VERSION}.json"
        out.write_text(json.dumps(rows, indent=2), encoding="utf-8")
    return 0


if __name__ == "__main__":
    sys.exit(main())
