"""Chat logic behind the app, kept separate from Gradio so it can be unit-tested.

Chat history: the answer step never sees the conversation (that keeps answers
grounded only in retrieved passages). History is used for one thing only:
rewriting a follow-up such as "what about for wireless?" into a standalone
question before retrieval ("question condensing").
"""

from __future__ import annotations

import time
from dataclasses import dataclass

from docuask.prompts import build_condense_messages
from docuask.rag import RAGPipeline, RAGResponse


def content_text(content) -> str:
    """Gradio 6 may hand back message content as a string, a dict or a list of blocks."""
    if isinstance(content, str):
        return content
    if isinstance(content, dict):
        return str(content.get("text", ""))
    if isinstance(content, list):
        return " ".join(content_text(c) for c in content)
    return ""


def history_pairs(messages: list[dict], max_turns: int = 3) -> list[tuple[str, str]]:
    """[{'role','content'}, ...] -> last `max_turns` (user, assistant) pairs."""
    pairs, pending_user = [], None
    for m in messages or []:
        role, text = m.get("role"), content_text(m.get("content", ""))
        if role == "user":
            pending_user = text
        elif role == "assistant" and pending_user is not None:
            pairs.append((pending_user, text))
            pending_user = None
    return pairs[-max_turns:]


@dataclass
class ChatTurn:
    question: str
    standalone_question: str
    response: RAGResponse
    rewrite_ms: float
    total_ms: float


class ChatEngine:
    def __init__(self, pipeline: RAGPipeline, rewrite_followups: bool = True, max_history_turns: int = 3):
        self.pipeline = pipeline
        self.rewrite_followups = rewrite_followups
        self.max_history_turns = max_history_turns

    def condense(self, history: list[tuple[str, str]], question: str) -> str:
        reply = self.pipeline.llm.invoke(build_condense_messages(history, question))
        text = content_text(getattr(reply, "content", reply)).strip().strip('"').strip()
        # Fall back to the original if the model returns something odd (empty or an essay)
        return text if 0 < len(text) <= 3 * len(question) + 200 else question

    def ask(self, question: str, messages: list[dict] | None = None) -> ChatTurn:
        t0 = time.perf_counter()
        history = history_pairs(messages or [], self.max_history_turns)
        standalone, rewrite_ms = question, 0.0
        if self.rewrite_followups and history:
            standalone = self.condense(history, question)
            rewrite_ms = 1000 * (time.perf_counter() - t0)
        response = self.pipeline.answer(standalone)
        return ChatTurn(question, standalone, response, rewrite_ms, 1000 * (time.perf_counter() - t0))


# --------------------------------------------------------------------------- #
# Presentation helpers (Markdown for the side panel)
# --------------------------------------------------------------------------- #
def format_sources(turn: ChatTurn) -> str:
    r = turn.response
    lines = []
    if turn.standalone_question != turn.question:
        lines.append(f"*Searched for:* “{turn.standalone_question}”\n")
    if r.refused:
        lines.append("**No answer found in the documents.** Closest passages retrieved:\n")
    else:
        lines.append(f"**{len(r.citations)} cited source(s)**, marked ✅. Other retrieved passages are shown below them.\n")
    cited = set(r.citations)
    ordered = sorted(r.hits, key=lambda h: h.chunk.citation not in cited)   # cited first, stable
    for h in ordered:
        c = h.chunk
        mark = "✅" if c.citation in cited else "▫️"
        snippet = c.text.strip().replace("\n", " ")
        if len(snippet) > 600:
            snippet = snippet[:600].rsplit(" ", 1)[0] + " …"
        lines.append(f"{mark} **{c.citation}** · {c.title} · score {h.score:.2f}  \n"
                     f"> {snippet}  \n[Open original document]({c.url})\n")
    if r.invalid_citations:
        lines.append(f"⚠️ The answer cited passages that were not retrieved: {', '.join(r.invalid_citations)}")
    return "\n".join(lines)


def format_timing(turn: ChatTurn) -> str:
    r = turn.response
    parts = [f"**Total {turn.total_ms / 1000:.2f} s**",
             f"retrieval {r.retrieval_ms:.0f} ms",
             f"generation {r.generation_ms / 1000:.2f} s"]
    if turn.rewrite_ms:
        parts.append(f"follow-up rewrite {turn.rewrite_ms / 1000:.2f} s")
    return " · ".join(parts)
