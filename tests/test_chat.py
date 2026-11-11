"""Chat logic tests: history parsing, follow-up rewriting, side-panel formatting."""

from dataclasses import dataclass

from docuask.chat import ChatEngine, content_text, format_sources, format_timing, history_pairs
from docuask.index import SearchHit
from docuask.ingest import Chunk
from docuask.prompts import REFUSAL
from docuask.rag import RAGPipeline


@dataclass
class Reply:
    content: str


class ScriptedLLM:
    """Returns queued replies in order and records every call."""

    def __init__(self, *replies):
        self.replies, self.calls = list(replies), []

    def invoke(self, messages):
        self.calls.append(messages)
        return Reply(self.replies.pop(0))


class FakeRetriever:
    name = "fake"

    def __init__(self):
        self.queries = []

    def retrieve(self, query, k):
        self.queries.append(query)
        c = Chunk("porting:p1:c0", "porting", "Porting", "https://fcc.gov/p.pdf", "porting.pdf", 1, None, 0,
                  "Simple ports must be processed in one business day.")
        return [SearchHit(c, 0.83)]


def test_content_text_handles_gradio_formats():
    assert content_text("hi") == "hi"
    assert content_text([{"type": "text", "text": "hi"}]) == "hi"
    assert content_text({"type": "text", "text": "hi"}) == "hi"


def test_history_pairs_keeps_last_turns():
    msgs = []
    for i in range(5):
        msgs += [{"role": "user", "content": f"q{i}"}, {"role": "assistant", "content": f"a{i}"}]
    assert history_pairs(msgs, max_turns=2) == [("q3", "a3"), ("q4", "a4")]
    assert history_pairs([]) == []


def test_first_question_is_not_rewritten():
    llm = ScriptedLLM("Ports take one business day [porting.pdf, p. 1].")
    retr = FakeRetriever()
    turn = ChatEngine(RAGPipeline(retr, llm)).ask("How long does porting take?", [])
    assert turn.standalone_question == "How long does porting take?"
    assert len(llm.calls) == 1 and turn.rewrite_ms == 0


def test_follow_up_is_rewritten_before_retrieval():
    llm = ScriptedLLM("How long does porting a wireless number take?", "Within hours [porting.pdf, p. 1].")
    retr = FakeRetriever()
    history = [{"role": "user", "content": "How long does porting take?"},
               {"role": "assistant", "content": "One business day."}]
    turn = ChatEngine(RAGPipeline(retr, llm)).ask("and for wireless?", history)
    assert retr.queries == ["How long does porting a wireless number take?"]
    assert turn.standalone_question != turn.question
    assert "and for wireless?" in llm.calls[0][1][1]          # rewrite prompt saw the follow-up
    assert "Searched for" in format_sources(turn)


def test_bad_rewrite_falls_back_to_original():
    llm = ScriptedLLM("", REFUSAL)
    history = [{"role": "user", "content": "q"}, {"role": "assistant", "content": "a"}]
    turn = ChatEngine(RAGPipeline(FakeRetriever(), llm)).ask("and then?", history)
    assert turn.standalone_question == "and then?"


def test_panels_show_citations_and_timing():
    llm = ScriptedLLM("Ports take one business day [porting.pdf, p. 1].")
    turn = ChatEngine(RAGPipeline(FakeRetriever(), llm)).ask("How long does porting take?")
    md = format_sources(turn)
    assert "✅ **[porting.pdf, p. 1]**" in md and "Open original document" in md
    assert format_timing(turn).startswith("**Total")


def test_refusal_panel():
    turn = ChatEngine(RAGPipeline(FakeRetriever(), ScriptedLLM(REFUSAL))).ask("World Cup winner?")
    assert "No answer found in the documents" in format_sources(turn)
