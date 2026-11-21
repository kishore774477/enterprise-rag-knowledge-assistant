"""Evaluate the assistant on a labelled question set.

Each item in the dataset looks like::

    {"question": "...", "expected_sources": ["hr_policy_handbook.md"], "expected_keywords": ["24", "days"]}

Items with an empty ``expected_sources`` list are *unanswerable* questions: the assistant
should say it could not find the answer instead of making something up.

Metrics
-------
* retrieval hit rate - an expected source is among the top-k retrieved chunks
* MRR               - mean reciprocal rank of the first correct source
* keyword recall    - share of expected keywords (facts) present in the answer
* refusal accuracy  - unanswerable questions correctly declined
* overall score     - answerable: 0.5 * hit + 0.5 * keyword recall; unanswerable: 1 if declined
"""

from __future__ import annotations

import json
import time
from dataclasses import asdict, dataclass
from pathlib import Path

from .assistant import RAGAssistant


@dataclass
class EvalRecord:
    question: str
    answerable: bool
    retrieval_hit: float
    reciprocal_rank: float
    keyword_recall: float
    declined: bool
    score: float
    retrieved: list[str]
    answer: str


def load_dataset(path: str | Path) -> list[dict]:
    data = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all("question" in d for d in data):
        raise ValueError(f"{path} must be a JSON list of objects with a 'question' field")
    return data


def evaluate(assistant: RAGAssistant, dataset: list[dict]) -> dict:
    records: list[EvalRecord] = []
    started = time.perf_counter()
    assistant.prepare()

    for item in dataset:
        question = item["question"]
        expected_sources = set(item.get("expected_sources", []))
        keywords = [k.lower() for k in item.get("expected_keywords", [])]
        answerable = bool(expected_sources)

        answer = assistant.ask(question)  # may switch to offline mode, so search afterwards
        retrieved = [r.source for r in assistant.kb.search(question)]
        declined = not answer.found

        rr = 0.0
        for rank, source in enumerate(retrieved, start=1):
            if source in expected_sources:
                rr = 1.0 / rank
                break
        hit = 1.0 if rr > 0 else 0.0
        text = answer.answer.lower()
        recall = sum(k in text for k in keywords) / len(keywords) if keywords else 1.0

        if answerable:
            score = 0.5 * hit + 0.5 * (0.0 if declined else recall)
        else:
            score = 1.0 if declined else 0.0

        records.append(
            EvalRecord(question, answerable, hit, rr, recall if not declined else 0.0, declined, score, retrieved, answer.answer)
        )

    answerable = [r for r in records if r.answerable]
    unanswerable = [r for r in records if not r.answerable]

    def mean(values):
        values = list(values)
        return round(sum(values) / len(values), 4) if values else None

    summary = {
        "mode": assistant.mode,
        "num_questions": len(records),
        "retrieval_hit_rate": mean(r.retrieval_hit for r in answerable),
        "mrr": mean(r.reciprocal_rank for r in answerable),
        "keyword_recall": mean(r.keyword_recall for r in answerable),
        "refusal_accuracy": mean(1.0 if r.declined else 0.0 for r in unanswerable),
        "overall_score": mean(r.score for r in records),
        "seconds": round(time.perf_counter() - started, 2),
    }
    return {"summary": summary, "records": [asdict(r) for r in records]}


def save_report(report: dict, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(report, indent=2), encoding="utf-8")
    return path
