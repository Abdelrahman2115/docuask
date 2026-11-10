"""Verify the Phase 1 setup: Python, packages, config, API key, and one LLM call.

Run from the project root:
    python scripts/check_setup.py
"""

from __future__ import annotations

import importlib
import sys
import time

PACKAGES = [
    "dotenv", "yaml", "langchain_core", "langchain_text_splitters",
    "langchain_groq", "langchain_ollama", "pypdf", "bs4",
    "sentence_transformers", "chromadb", "rank_bm25", "gradio",
]


def main() -> int:
    ok = True
    print(f"Python {sys.version.split()[0]}")

    print("\n[1/3] Packages")
    for name in PACKAGES:
        try:
            mod = importlib.import_module(name)
            print(f"  OK   {name:26s} {getattr(mod, '__version__', '')}")
        except Exception as e:  # noqa: BLE001
            ok = False
            print(f"  FAIL {name:26s} {type(e).__name__}: {e}")

    print("\n[2/3] Config")
    from docuask.config import get_settings

    s = get_settings()
    key = s.llm.groq_api_key
    masked = f"{key[:4]}...{key[-4:]}" if key else "(not set)"
    print(f"  provider={s.llm.provider} model={s.llm.model}")
    print(f"  GROQ_API_KEY={masked}")
    print(f"  index_dir={s.paths.index_dir}")

    print("\n[3/3] LLM test call")
    try:
        from docuask.llm import get_llm

        start = time.perf_counter()
        reply = get_llm(s).invoke("Reply with exactly the word: OK")
        print(f"  reply={reply.content.strip()!r} in {time.perf_counter() - start:.2f}s")
    except Exception as e:  # noqa: BLE001
        ok = False
        print(f"  FAIL {type(e).__name__}: {e}")

    print("\nAll checks passed." if ok else "\nSome checks failed. Paste the full output to debug.")
    return 0 if ok else 1


if __name__ == "__main__":
    sys.exit(main())
