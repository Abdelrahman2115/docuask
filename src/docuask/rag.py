"""The RAG pipeline: retrieve -> prompt -> generate -> check citations."""

from __future__ import annotations

import re
import time
from dataclasses import dataclass, field

from docuask.index import SearchHit
from docuask.prompts import REFUSAL, build_messages

# [porting.pdf, p. 1]   [telephone_bill.html, § Cramming]
CITATION_RE = re.compile(r"\[([^\[\]\n,]+?\.(?:pdf|html)),\s*((?:p\.\s*\d+)|(?:§\s*[^\]\n]+?))\s*\]")


@dataclass
class RAGResponse:
    question: str
    answer: str
    hits: list[SearchHit]
    refused: bool
    citations: list[str] = field(default_factory=list)          # citations found in the answer
    invalid_citations: list[str] = field(default_factory=list)  # cited but NOT among retrieved passages
    retrieval_ms: float = 0.0
    generation_ms: float = 0.0
    total_ms: float = 0.0
    skipped_llm: bool = False        # True when the score threshold refused before calling the LLM

    @property
    def cited_hits(self) -> list[SearchHit]:
        cited = set(self.citations)
        return [h for h in self.hits if h.chunk.citation in cited]


def normalize_citation(source_file: str, where: str) -> str:
    where = re.sub(r"\s+", " ", where.strip())
    where = re.sub(r"^p\.\s*", "p. ", where)
    where = re.sub(r"^§\s*", "§ ", where)
    return f"[{source_file.strip()}, {where}]"


def extract_citations(answer: str) -> list[str]:
    seen, out = set(), []
    for f, w in CITATION_RE.findall(answer):
        c = normalize_citation(f, w)
        if c not in seen:
            seen.add(c)
            out.append(c)
    return out


def is_refusal(answer: str) -> bool:
    a = answer.strip().replace("’", "'").lower()
    return a.startswith("i don't know") or a.startswith("i do not know")


class RAGPipeline:
    def __init__(self, retriever, llm, top_k: int = 4, min_score: float = 0.0):
        self.retriever = retriever
        self.llm = llm
        self.top_k = top_k
        self.min_score = min_score

    def answer(self, question: str) -> RAGResponse:
        t0 = time.perf_counter()
        hits = self.retriever.retrieve(question, self.top_k)
        t1 = time.perf_counter()

        if not hits or (self.min_score > 0 and hits[0].score < self.min_score):
            return RAGResponse(question, REFUSAL, hits, refused=True, skipped_llm=True,
                               retrieval_ms=1000 * (t1 - t0), total_ms=1000 * (t1 - t0))

        reply = self.llm.invoke(build_messages(question, hits))
        text = (reply.content if hasattr(reply, "content") else str(reply)).strip()
        t2 = time.perf_counter()

        refused = is_refusal(text)
        citations = [] if refused else extract_citations(text)
        retrieved = {h.chunk.citation for h in hits}
        return RAGResponse(
            question=question,
            answer=text,
            hits=hits,
            refused=refused,
            citations=citations,
            invalid_citations=[c for c in citations if c not in retrieved],
            retrieval_ms=1000 * (t1 - t0),
            generation_ms=1000 * (t2 - t1),
            total_ms=1000 * (t2 - t0),
        )
