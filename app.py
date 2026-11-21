"""Gradio web UI for the Enterprise RAG Knowledge Assistant.

Usage:
    python app.py                 # http://127.0.0.1:7860
    python app.py --port 8000 --share
"""

from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

os.environ.setdefault("GRADIO_ANALYTICS_ENABLED", "False")  # no usage telemetry from company machines

import gradio as gr

from rag import RAGAssistant, get_settings
from rag.evaluation import evaluate, load_dataset, save_report
from rag.loaders import SUPPORTED_EXTENSIONS

logging.basicConfig(level=logging.INFO, format="%(levelname)s %(name)s: %(message)s")
logger = logging.getLogger("app")

settings = get_settings()
assistant = RAGAssistant(settings)

EXAMPLES = [
    "How many days of annual leave do I get?",
    "Who approves an expense of INR 80,000?",
    "What is the hotel limit per night in Bengaluru?",
    "How soon must I report a security incident?",
    "Can I work from home?",
    "When is salary credited?",
]


# -- helpers -------------------------------------------------------------------------------
def mode_banner() -> str:
    if assistant.fallback_reason:
        return f"**Mode:** Offline (fallback) — {assistant.fallback_reason}"
    if assistant.settings.use_openai:
        return (
            f"**Mode:** OpenAI — chat model `{settings.chat_model}`, embeddings `{settings.embedding_model}`"
        )
    return (
        "**Mode:** Offline demo — local TF-IDF retrieval with extractive answers. "
        "Add `OPENAI_API_KEY` to `.env` for generated answers."
    )


def kb_status() -> str:
    meta = assistant.prepare()
    return (
        f"**{meta['num_documents']} documents · {meta['num_chunks']} chunks** · "
        f"indexed {meta['built_at']} · backend `{meta['backend']}`"
    )


def documents_table() -> list[list]:
    meta = assistant.prepare()
    rows = []
    for name, chunks in meta["sources"].items():
        path = settings.docs_dir / name
        size_kb = round(path.stat().st_size / 1024, 1) if path.exists() else "-"
        rows.append([name, chunks, size_kb])
    return rows


def format_sources(answer) -> str:
    if not answer.sources:
        return "_No matching passages._"
    parts = []
    for s in answer.sources:
        where = f" — {s.location}" if s.location else ""
        snippet = s.content if len(s.content) <= 700 else s.content[:700] + "…"
        parts.append(f"**[{s.index}] {s.source}**{where}  \n_relevance {s.score:.2f}_\n\n> " + snippet.replace("\n", "\n> "))
    return "\n\n".join(parts)


# -- event handlers --------------------------------------------------------------------------
def respond(message: str, history: list[dict]):
    history = history or []
    message = (message or "").strip()
    if not message:
        return "", history, gr.update()
    try:
        answer = assistant.ask(message, history)
    except Exception as exc:  # surface errors in the chat instead of crashing the UI
        logger.exception("Question failed")
        reply = f"⚠️ Something went wrong: {exc}"
        return "", history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}], gr.update()

    reply = answer.answer
    if answer.sources:
        refs = ", ".join(dict.fromkeys(s.source for s in answer.sources))
        reply += f"\n\n**Sources:** {refs}"
    if answer.note:
        reply += f"\n\n_{answer.note}_"
    history = history + [{"role": "user", "content": message}, {"role": "assistant", "content": reply}]
    return "", history, format_sources(answer)


def upload_files(files):
    if not files:
        return "Select one or more files first.", documents_table(), kb_status()
    paths = [f if isinstance(f, str) else f.name for f in files]
    try:
        added, meta = assistant.kb.add_files(paths)
    except Exception as exc:
        logger.exception("Upload failed")
        return f"❌ {exc}", documents_table(), kb_status()
    return f"✅ Added {', '.join(added)} and re-indexed ({meta['num_chunks']} chunks).", documents_table(), kb_status()


def rebuild_index():
    try:
        meta = assistant.kb.build()
    except Exception as exc:
        return f"❌ {exc}", documents_table(), kb_status()
    return f"✅ Rebuilt index: {meta['num_chunks']} chunks.", documents_table(), kb_status()


def delete_document(name: str):
    name = (name or "").strip()
    if not name:
        return "Type the exact file name to delete.", documents_table(), kb_status()
    try:
        assistant.kb.remove_document(name)
    except Exception as exc:
        return f"❌ {exc}", documents_table(), kb_status()
    return f"🗑️ Removed {name} and re-indexed.", documents_table(), kb_status()


def run_evaluation():
    try:
        report = evaluate(assistant, load_dataset(settings.eval_file))
    except Exception as exc:
        logger.exception("Evaluation failed")
        return f"❌ {exc}", []
    save_report(report, Path("reports/eval_report.json"))
    s = report["summary"]

    def pct(v):
        return "n/a" if v is None else f"{v * 100:.1f}%"

    summary = (
        f"### Overall score: {pct(s['overall_score'])}\n"
        f"| Metric | Value |\n|---|---|\n"
        f"| Questions | {s['num_questions']} |\n"
        f"| Retrieval hit rate | {pct(s['retrieval_hit_rate'])} |\n"
        f"| MRR | {pct(s['mrr'])} |\n"
        f"| Keyword recall | {pct(s['keyword_recall'])} |\n"
        f"| Refusal accuracy (out-of-scope) | {pct(s['refusal_accuracy'])} |\n"
        f"| Mode | {s['mode']} |\n| Time | {s['seconds']}s |"
    )
    rows = [
        [r["question"], "yes" if r["answerable"] else "no", r["retrieval_hit"], round(r["keyword_recall"], 2), r["declined"], round(r["score"], 2)]
        for r in report["records"]
    ]
    return summary, rows


# -- layout ------------------------------------------------------------------------------------
def build_ui() -> gr.Blocks:
    with gr.Blocks(title="Enterprise RAG Knowledge Assistant") as demo:
        gr.Markdown(
            "# 🏢 Enterprise RAG Knowledge Assistant\n"
            "Ask questions about company HR policies, finance & expense rules, IT security and FAQs. "
            "Answers are grounded in the indexed documents and cite their sources."
        )
        gr.Markdown(mode_banner())

        with gr.Tab("💬 Ask"):
            with gr.Row():
                with gr.Column(scale=3):
                    chatbot = gr.Chatbot(height=480, label="Assistant", buttons=["copy"])
                    with gr.Row():
                        question = gr.Textbox(
                            placeholder="e.g. How many days of sick leave do I get?",
                            show_label=False,
                            scale=5,
                            autofocus=True,
                        )
                        send = gr.Button("Ask", variant="primary", scale=1)
                    gr.Examples(EXAMPLES, inputs=question, label="Try these")
                    clear = gr.Button("Clear conversation")
                with gr.Column(scale=2):
                    gr.Markdown("### 📄 Retrieved passages")
                    sources = gr.Markdown("_Ask a question to see the supporting passages._")

            send.click(respond, [question, chatbot], [question, chatbot, sources])
            question.submit(respond, [question, chatbot], [question, chatbot, sources])
            clear.click(lambda: ([], "_Ask a question to see the supporting passages._"), None, [chatbot, sources])

        with gr.Tab("📚 Knowledge base"):
            status = gr.Markdown(kb_status)
            docs = gr.Dataframe(
                value=documents_table,
                headers=["Document", "Chunks", "Size (KB)"],
                interactive=False,
            )
            with gr.Row():
                uploader = gr.File(
                    label=f"Upload documents ({', '.join(sorted(SUPPORTED_EXTENSIONS))})",
                    file_count="multiple",
                    file_types=sorted(SUPPORTED_EXTENSIONS),
                )
            with gr.Row():
                upload_btn = gr.Button("Add to knowledge base", variant="primary")
                rebuild_btn = gr.Button("Rebuild index")
            with gr.Row():
                delete_name = gr.Textbox(label="Remove a document (exact file name)", scale=4)
                delete_btn = gr.Button("Remove", variant="stop", scale=1)
            kb_message = gr.Markdown()

            upload_btn.click(upload_files, uploader, [kb_message, docs, status])
            rebuild_btn.click(rebuild_index, None, [kb_message, docs, status])
            delete_btn.click(delete_document, delete_name, [kb_message, docs, status])

        with gr.Tab("📊 Evaluation"):
            gr.Markdown(
                f"Runs the labelled question set in `{settings.eval_file.name}` and measures retrieval "
                "accuracy, answer keyword recall and whether out-of-scope questions are declined."
            )
            eval_btn = gr.Button("Run evaluation", variant="primary")
            eval_summary = gr.Markdown()
            eval_table = gr.Dataframe(
                headers=["Question", "Answerable", "Retrieval hit", "Keyword recall", "Declined", "Score"],
                interactive=False,
                wrap=True,
            )
            eval_btn.click(run_evaluation, None, [eval_summary, eval_table])

    return demo


def main() -> None:
    parser = argparse.ArgumentParser(description="Run the Gradio web app")
    parser.add_argument("--host", default="127.0.0.1")
    parser.add_argument("--port", type=int, default=7860)
    parser.add_argument("--share", action="store_true", help="create a public Gradio link")
    args = parser.parse_args()

    assistant.prepare()  # build the index on first launch
    build_ui().launch(server_name=args.host, server_port=args.port, share=args.share, theme=gr.themes.Soft())


if __name__ == "__main__":
    main()
