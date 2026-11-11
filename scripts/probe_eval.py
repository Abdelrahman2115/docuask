"""Check the RAGAS judge setup before spending quota on a full run.

    python scripts/probe_eval.py

Runs RAGAS Faithfulness on a toy example with each candidate judge and output
mode until one works, then a quick AnswerRelevancy check. The toy answer is
half right on purpose (one supported claim, one unsupported), so a working
judge should score about 0.5. The first judge that scores between 0.25 and
0.75 is saved to data/eval/judge.json and used by ragas_eval.py.
"""

from __future__ import annotations

import asyncio
import importlib.metadata as md
import json
import time

from docuask.judge import JUDGE_CHOICE_FILE, make_judge, make_judge_embeddings  # applies the ragas patch

CANDIDATE_JUDGES = ["openai/gpt-oss-120b", "qwen/qwen3.8-27b"]
MODES = ["JSON", "MD_JSON", "TOOLS"]

TOY = dict(
    user_input="How long does a simple phone number port take?",
    response="Simple ports must be processed in one business day. Porting is always free of charge.",
    retrieved_contexts=["FCC rules require simple ports to be processed in one business day. "
                        "Companies may charge you to port your number."],
)


async def try_judge(model, mode):
    from ragas.metrics.collections import Faithfulness

    return (await Faithfulness(llm=make_judge(model, mode)).ascore(**TOY)).value


async def try_relevancy(model, mode):
    from ragas.metrics.collections import AnswerRelevancy

    scorer = AnswerRelevancy(llm=make_judge(model, mode), embeddings=make_judge_embeddings(), strictness=1)
    return (await scorer.ascore(user_input=TOY["user_input"], response=TOY["response"])).value


def main() -> None:
    print("===== VERSIONS =====")
    for pkg in ["ragas", "instructor", "openai", "langchain-community"]:
        try:
            print(f"  {pkg:20s} {md.version(pkg)}")
        except md.PackageNotFoundError:
            print(f"  {pkg:20s} NOT INSTALLED")

    print("\n===== FAITHFULNESS SMOKE TEST (expect ~0.5) =====")
    chosen = None
    for model in CANDIDATE_JUDGES:
        for mode in MODES:
            t0 = time.perf_counter()
            try:
                v = asyncio.run(try_judge(model, mode))
                ok = v is not None and 0.25 <= float(v) <= 0.75
                print(f"  {'OK  ' if ok else 'ODD '} judge={model} mode={mode} faithfulness={v} "
                      f"({time.perf_counter() - t0:.1f}s)")
                if ok:
                    chosen = chosen or (model, mode)
                    break
            except Exception as e:  # noqa: BLE001
                print(f"  FAIL judge={model} mode={mode}: {type(e).__name__}: {str(e)[:300]}")
        if chosen:
            break

    if not chosen:
        print("\nNo judge worked. Paste this log so we can debug.")
        return
    JUDGE_CHOICE_FILE.write_text(json.dumps({"model": chosen[0], "mode": chosen[1]}), encoding="utf-8")
    print(f"\nChosen judge: {chosen} -> saved to {JUDGE_CHOICE_FILE}")

    print("\n===== ANSWER RELEVANCY SMOKE TEST (expect high, ~0.7-1.0) =====")
    try:
        t0 = time.perf_counter()
        v = asyncio.run(try_relevancy(*chosen))
        print(f"  OK   answer_relevancy={v} ({time.perf_counter() - t0:.1f}s)")
    except Exception as e:  # noqa: BLE001
        print(f"  FAIL answer_relevancy: {type(e).__name__}: {str(e)[:400]}")


if __name__ == "__main__":
    main()
