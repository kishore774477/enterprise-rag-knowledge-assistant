"""End-to-end tests in offline mode (no API key or network needed)."""

from __future__ import annotations

import shutil
from pathlib import Path

import pytest

from rag import NOT_FOUND, KnowledgeBase, RAGAssistant, get_settings
from rag.config import ROOT_DIR
from rag.evaluation import evaluate, load_dataset
from rag.loaders import load_file, split_documents
from rag.text import stem, tokenize

SAMPLE_DOCS = ROOT_DIR / "data" / "documents"


@pytest.fixture()
def settings(tmp_path: Path):
    docs = tmp_path / "docs"
    shutil.copytree(SAMPLE_DOCS, docs)
    return get_settings(requested_mode="offline", docs_dir=docs, index_dir=tmp_path / "index")


@pytest.fixture()
def assistant(settings):
    return RAGAssistant(settings)


def test_stemming_and_tokenizing():
    assert stem("policies") == stem("policy")
    assert stem("approved") == stem("approve")
    assert "the" not in tokenize("What is the leave policy?")


def test_markdown_chunks_keep_section_titles():
    chunks = split_documents(load_file(SAMPLE_DOCS / "hr_policy_handbook.md"))
    sick = [c for c in chunks if "Sick Leave" in c.metadata.get("section", "")]
    assert sick and sick[0].page_content.startswith("HR Policy Handbook > Leave Policy > Sick Leave")


def test_build_save_and_reload(settings):
    meta = KnowledgeBase(settings).build()
    assert meta["num_documents"] == 4 and meta["num_chunks"] > 10

    reloaded = KnowledgeBase(settings)
    assert reloaded.load()
    assert reloaded.search("sick leave medical certificate")[0].source == "hr_policy_handbook.md"


def test_answer_is_grounded_and_cited(assistant):
    answer = assistant.ask("How many days of annual leave do employees get?")
    assert answer.found
    assert "24 days" in answer.answer
    assert "[1]" in answer.answer
    assert answer.sources[0].source == "hr_policy_handbook.md"


def test_out_of_scope_question_is_declined(assistant):
    answer = assistant.ask("Who won the cricket world cup in 2011?")
    assert answer.answer == NOT_FOUND
    assert answer.sources == []


def test_follow_up_question_uses_history(assistant):
    history = [
        {"role": "user", "content": "What is the hotel limit per night in metro cities?"},
        {"role": "assistant", "content": "INR 6,000"},
    ]
    answer = assistant.ask("and for international travel?", history)
    assert "USD 180" in answer.answer


def test_uploaded_document_becomes_searchable(assistant, tmp_path):
    new_doc = tmp_path / "canteen_policy.txt"
    new_doc.write_text("The canteen serves free masala dosa every Wednesday morning.", encoding="utf-8")
    added, meta = assistant.kb.add_files([new_doc])
    assert added == ["canteen_policy.txt"] and "canteen_policy.txt" in meta["sources"]
    assert "masala dosa" in assistant.ask("When is free masala dosa served in the canteen?").answer


def test_unsupported_upload_is_rejected(assistant, tmp_path):
    bad = tmp_path / "virus.exe"
    bad.write_bytes(b"MZ")
    with pytest.raises(ValueError):
        assistant.kb.add_files([bad])


def test_evaluation_meets_target(assistant, settings):
    report = evaluate(assistant, load_dataset(ROOT_DIR / "data" / "eval" / "qa_dataset.json"))
    summary = report["summary"]
    assert summary["retrieval_hit_rate"] >= 0.9
    assert summary["refusal_accuracy"] == 1.0
    assert summary["overall_score"] >= 0.85


def test_openai_mode_requires_key():
    with pytest.raises(ValueError):
        get_settings(requested_mode="openai", openai_api_key=None)


def test_falls_back_to_offline_when_openai_unavailable(settings, monkeypatch):
    original_build = KnowledgeBase.build

    def flaky_build(self):
        if self.settings.use_openai:
            raise ConnectionError("simulated OpenAI outage")
        return original_build(self)

    monkeypatch.setattr(KnowledgeBase, "build", flaky_build)
    openai_settings = settings.with_overrides(requested_mode="auto", openai_api_key="sk-test-not-real")
    assistant = RAGAssistant(openai_settings)
    assert assistant.mode == "openai"

    answer = assistant.ask("How many days of sick leave do employees get?")
    assert assistant.mode == "offline"
    assert "12 days" in answer.answer
    assert "offline mode" in answer.note
