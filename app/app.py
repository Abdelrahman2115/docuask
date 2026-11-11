"""DocuAsk chat UI (Gradio 6).

Run locally from the project root:
    python app/app.py            -> http://127.0.0.1:7860

On Hugging Face Spaces / Docker the same file is the entry point; set
GROQ_API_KEY as a secret and GRADIO_SERVER_NAME=0.0.0.0.
"""

from __future__ import annotations

import os
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))          # works without `pip install -e .` (e.g. in Docker)

import gradio as gr  # noqa: E402

from docuask.chat import ChatEngine, format_sources, format_timing  # noqa: E402
from docuask.config import get_settings  # noqa: E402
from docuask.llm import get_llm  # noqa: E402
from docuask.rag import RAGPipeline  # noqa: E402
from docuask.retriever import build_retriever  # noqa: E402

EXAMPLES = [
    "How long does it take to port my number to a new carrier?",
    "My phone bill has charges for services I never ordered. What can I do?",
    "I got a missed call from a strange international number. Should I call back?",
    "Can I text 911 if I can't talk?",
    "What download speed do I need for 4K streaming?",
]

INTRO = """# 📞 DocuAsk
Ask questions about U.S. telecom consumer rights. Answers come **only** from 40 FCC consumer
guides and cite the exact page or section; if the guides don't cover it, DocuAsk says so.
*Not legal advice. U.S. rules only.*"""


def build_engine() -> tuple[ChatEngine, str]:
    s = get_settings()
    retriever = build_retriever(s)
    pipeline = RAGPipeline(retriever, get_llm(s), top_k=s.retrieval.top_k, min_score=s.retrieval.min_score)
    info = f"Retriever: `{retriever.name}` · LLM: `{s.llm.provider}/{s.llm.model}` · top-k {s.retrieval.top_k}"
    return ChatEngine(pipeline), info


ENGINE, CONFIG_INFO = build_engine()


def respond(message: str, history: list[dict]):
    message = (message or "").strip()
    if not message:
        return history, gr.update(), gr.update(), ""
    history = list(history or [])
    try:
        turn = ENGINE.ask(message, history)
        answer, sources, timing = turn.response.answer, format_sources(turn), format_timing(turn)
    except Exception as e:  # noqa: BLE001 - show the problem instead of crashing the UI
        answer = f"⚠️ Something went wrong: {type(e).__name__}: {e}"
        sources, timing = "", ""
    history += [{"role": "user", "content": message}, {"role": "assistant", "content": answer}]
    return history, sources, timing, ""


def clear():
    return [], "*Sources for the latest answer appear here.*", "", ""


with gr.Blocks(title="DocuAsk") as demo:
    gr.Markdown(INTRO)
    with gr.Row():
        with gr.Column(scale=3):
            chatbot = gr.Chatbot(label="Conversation", height=480, buttons=["copy"])
            with gr.Row():
                box = gr.Textbox(placeholder="Ask about phone bills, porting, scam calls, 911, broadband…",
                                 show_label=False, scale=5, autofocus=True)
                send = gr.Button("Ask", variant="primary", scale=1)
            with gr.Row():
                clear_btn = gr.Button("New conversation")
            gr.Examples(EXAMPLES, inputs=box, label="Try one")
        with gr.Column(scale=2):
            timing = gr.Markdown("")
            gr.Markdown("### Sources")
            sources = gr.Markdown("*Sources for the latest answer appear here.*")
    gr.Markdown(f"<small>{CONFIG_INFO}</small>")

    outputs = [chatbot, sources, timing, box]
    box.submit(respond, [box, chatbot], outputs)
    send.click(respond, [box, chatbot], outputs)
    clear_btn.click(clear, None, outputs)


if __name__ == "__main__":
    demo.queue(default_concurrency_limit=2).launch(
        server_name=os.getenv("GRADIO_SERVER_NAME", "127.0.0.1"),
        server_port=int(os.getenv("PORT", os.getenv("GRADIO_SERVER_PORT", "7860"))),
        theme=gr.themes.Soft(),
        footer_links=["gradio", "settings"],
    )
