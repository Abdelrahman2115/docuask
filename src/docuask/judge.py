"""LLM-as-judge setup for RAGAS 0.4 (collections API).

Two compatibility fixes, both found by reading the installed ragas 0.4.3 source:

1. ragas/llms/base.py imports ChatVertexAI / VertexAI from langchain_community,
   which langchain-community 0.4 removed. ragas only uses them in an isinstance
   list, so we register empty placeholder classes before ragas is imported.
2. ragas's provider="groq" path calls client.messages.create (Anthropic's API
   shape), which Groq's client doesn't have. Instead we use Groq's
   OpenAI-compatible endpoint with the openai client and build ragas's
   InstructorLLM ourselves, so we control the output mode and reasoning effort.

The judge is a different model from the generator (gpt-oss-20b), so the
system isn't grading its own answers.
"""

from __future__ import annotations

import importlib
import json
import os
import re
import sys
import types

from docuask.config import PROJECT_ROOT
from docuask.rag import CITATION_RE

GROQ_OPENAI_BASE_URL = "https://api.groq.com/openai/v1"
DEFAULT_JUDGE = "openai/gpt-oss-120b"
DEFAULT_MODE = "JSON"
JUDGE_CHOICE_FILE = PROJECT_ROOT / "data" / "eval" / "judge.json"   # written by probe_eval.py


def _patch_langchain_community() -> None:
    try:
        importlib.import_module("langchain_community.chat_models.vertexai")
    except Exception:  # noqa: BLE001
        mod = types.ModuleType("langchain_community.chat_models.vertexai")
        mod.ChatVertexAI = type("ChatVertexAI", (), {})
        sys.modules["langchain_community.chat_models.vertexai"] = mod
    try:
        from langchain_community.llms import VertexAI  # noqa: F401
    except Exception:  # noqa: BLE001
        import langchain_community.llms as llms

        llms.VertexAI = type("VertexAI", (), {})


_patch_langchain_community()   # must run before anything imports ragas


def default_judge() -> tuple[str, str]:
    """(model, mode) chosen by probe_eval.py, or the defaults."""
    if JUDGE_CHOICE_FILE.exists():
        d = json.loads(JUDGE_CHOICE_FILE.read_text("utf-8"))
        return d["model"], d["mode"]
    return DEFAULT_JUDGE, DEFAULT_MODE


def _groq_key() -> str:
    key = os.getenv("GROQ_API_KEY")
    if not key:
        from docuask.config import get_settings  # loads .env

        key = get_settings().llm.groq_api_key
    if not key:
        raise RuntimeError("GROQ_API_KEY missing")
    return key


def make_judge(model: str = DEFAULT_JUDGE, mode: str = DEFAULT_MODE, max_tokens: int = 4000):
    import instructor
    from openai import AsyncOpenAI
    from ragas.llms.base import InstructorLLM, InstructorModelArgs

    client = AsyncOpenAI(api_key=_groq_key(), base_url=GROQ_OPENAI_BASE_URL, max_retries=8)
    patched = instructor.from_openai(client, mode=getattr(instructor.Mode, mode))
    extra = {"reasoning_effort": "low"} if model.startswith("openai/gpt-oss") else {}
    return InstructorLLM(
        client=patched,
        model=model,
        provider="openai",
        model_args=InstructorModelArgs(temperature=0.0, top_p=1.0, max_tokens=max_tokens),
        **extra,
    )


def make_judge_embeddings(model: str = "sentence-transformers/all-MiniLM-L6-v2"):
    from ragas.embeddings import HuggingFaceEmbeddings

    return HuggingFaceEmbeddings(model=model)


def strip_citations(answer: str) -> str:
    """Remove '[file, p. N]' tags so the judge sees only the claims themselves."""
    return re.sub(r"\s{2,}", " ", CITATION_RE.sub("", answer)).strip()
