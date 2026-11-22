"""Evaluate retrieval and answer quality on the labelled question set.

Usage:
    python evaluate.py
    python evaluate.py --dataset data/eval/qa_dataset.json --out reports/eval_report.json
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import envcheck  # noqa: F401  (must come before third-party imports)

from rag import RAGAssistant, get_settings
from rag.evaluation import evaluate, load_dataset, save_report


def _pct(value) -> str:
    return "n/a" if value is None else f"{value * 100:.1f}%"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--dataset", type=Path, help="JSON question set")
    parser.add_argument("--out", type=Path, default=Path("reports/eval_report.json"))
    parser.add_argument("--verbose", "-v", action="store_true", help="print every question")
    args = parser.parse_args()

    settings = get_settings()
    assistant = RAGAssistant(settings)
    dataset = load_dataset(args.dataset or settings.eval_file)
    report = evaluate(assistant, dataset)
    s = report["summary"]

    if args.verbose:
        for r in report["records"]:
            status = "OK " if r["score"] >= 0.75 else "LOW"
            print(f"[{status}] {r['score']:.2f}  {r['question']}")

    print(f"\nMode:               {s['mode']}")
    print(f"Questions:          {s['num_questions']}")
    print(f"Retrieval hit rate: {_pct(s['retrieval_hit_rate'])}")
    print(f"MRR:                {_pct(s['mrr'])}")
    print(f"Keyword recall:     {_pct(s['keyword_recall'])}")
    print(f"Refusal accuracy:   {_pct(s['refusal_accuracy'])}")
    print(f"OVERALL SCORE:      {_pct(s['overall_score'])}   ({s['seconds']}s)")
    print(f"Report saved to {save_report(report, args.out)}")
    sys.exit(0)


if __name__ == "__main__":
    main()
