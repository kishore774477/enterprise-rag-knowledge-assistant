# Enterprise RAG Knowledge Assistant

A Retrieval-Augmented Generation (RAG) assistant that answers employee questions from
company documents such as HR policies, finance and expense manuals, IT security policies
and FAQs. Every answer is grounded in the indexed documents and cites its sources.
Questions the documents don't cover are declined instead of answered with made-up facts.

**Stack:** Python · LangChain · OpenAI · Gradio

## Features

- **Chat UI (Gradio)** with inline citations `[1]`, a panel showing the retrieved passages and their relevance, and follow-up questions that use the conversation history.
- **Multi-format ingestion:** PDF, DOCX, Markdown and TXT. Markdown is split by heading, so every chunk knows its section (for example *HR Policy Handbook > Leave Policy > Sick Leave*).
- **Knowledge base management:** upload, remove and re-index documents from the browser.
- **Two modes:**
  - **OpenAI mode:** OpenAI embeddings plus a chat model that writes the answer from the retrieved context (LangChain LCEL chain).
  - **Offline mode:** used when no API key is set. Local TF-IDF hashing embeddings plus an extractive answerer, so the project runs on any laptop at no cost. If OpenAI is unreachable or the key is invalid, the app switches to offline mode by itself and tells you.
- **Built-in evaluation:** retrieval hit rate, MRR, answer keyword recall and refusal accuracy on a labelled question set, available from the CLI or the UI.
- **Tests:** pytest suite covering ingestion, retrieval, answers, uploads, evaluation and the OpenAI fallback.

## Architecture

```mermaid
flowchart LR
    A[PDF / DOCX / MD / TXT] --> B[Loaders]
    B --> C[Heading-aware chunking]
    C --> D[Embeddings<br/>OpenAI or local TF-IDF]
    D --> E[(Vector index<br/>storage/index)]
    Q[Employee question] --> R[Retriever top-k]
    E --> R
    R --> G{Mode}
    G -->|OpenAI| L[ChatOpenAI + grounded prompt]
    G -->|Offline| X[Extractive answerer]
    L --> O[Answer + citations]
    X --> O
```

## Quick start (Windows PowerShell)

```powershell
git clone https://github.com/kishore774477/enterprise-rag-knowledge-assistant.git
cd enterprise-rag-knowledge-assistant
python -m venv .venv
.\.venv\Scripts\Activate.ps1
pip install -r requirements.txt
python app.py
```

Then open http://127.0.0.1:7860. The index is built automatically on first launch.

On macOS or Linux, activate the environment with `source .venv/bin/activate` instead.

### Use OpenAI (optional)

```powershell
Copy-Item .env.example .env
```

Put your key in `.env` (`OPENAI_API_KEY=sk-...`) and restart the app. The chat and embedding
models can be changed there too (`OPENAI_CHAT_MODEL`, `OPENAI_EMBEDDING_MODEL`). Each mode
keeps its own index under `storage/index/<mode>/`.

## Command-line tools

| Command | What it does |
|---|---|
| `python app.py [--port 7860] [--share]` | Start the web app |
| `python ingest.py [--docs FOLDER]` | Rebuild the index from a documents folder |
| `python evaluate.py [-v]` | Run the evaluation and save `reports/eval_report.json` |
| `pytest -q` | Run the tests |

## Evaluation

`data/eval/qa_dataset.json` holds 28 labelled questions: 24 answerable ones across the four
sample documents and 4 out-of-scope ones that the assistant should decline.

| Metric | Offline mode |
|---|---|
| Retrieval hit rate (correct document in top 4) | 100.0% |
| MRR | 100.0% |
| Answer keyword recall | 95.8% |
| Refusal accuracy (out-of-scope questions) | 100.0% |
| **Overall score** | **98.2%** |

The overall score averages, per question, `0.5 × retrieval hit + 0.5 × keyword recall` for
answerable questions and `1 if correctly declined` for out-of-scope ones. The one miss in
offline mode is *"What is the minimum password length?"*: the policy says "at least 12
characters long", and keyword matching can't link "minimum length" to that wording. Semantic
OpenAI embeddings are built for that kind of paraphrase. Run `python evaluate.py` with a key
set to measure OpenAI mode on your own account.

## Project structure

```
├── app.py                  # Gradio web UI
├── ingest.py               # CLI: build the index
├── evaluate.py             # CLI: run the evaluation
├── rag/
│   ├── config.py           # settings from .env / environment
│   ├── loaders.py          # PDF/DOCX/MD/TXT loading + chunking
│   ├── embeddings.py       # OpenAI + offline TF-IDF hashing embeddings
│   ├── knowledge_base.py   # build / persist / load / search the vector index
│   ├── assistant.py        # retrieval + grounded answer generation
│   ├── evaluation.py       # metrics
│   └── text.py             # tokenizer, stemmer, sentence splitter
├── data/
│   ├── documents/          # sample company documents (replace with your own)
│   └── eval/qa_dataset.json
└── tests/
```

## Using your own documents

Drop files into `data/documents/` and run `python ingest.py`, or upload them from the
**Knowledge base** tab. Update `data/eval/qa_dataset.json` with questions about your
documents to measure quality.

The sample documents describe a fictional company, *Orion Dynamics Ltd.*, and are for
demonstration only.
