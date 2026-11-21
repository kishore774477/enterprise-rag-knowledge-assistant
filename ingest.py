"""Build (or rebuild) the vector index from the documents folder.

Usage:
    python ingest.py                 # index data/documents
    python ingest.py --docs my_docs  # index another folder
"""

from __future__ import annotations

import argparse
import logging
from pathlib import Path

from rag import KnowledgeBase, get_settings


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--docs", type=Path, help="folder containing PDF/DOCX/MD/TXT files")
    parser.add_argument("--chunk-size", type=int, help="characters per chunk")
    parser.add_argument("--chunk-overlap", type=int, help="overlap between chunks")
    args = parser.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s %(message)s")

    overrides = {}
    if args.docs:
        overrides["docs_dir"] = args.docs.resolve()
    if args.chunk_size:
        overrides["chunk_size"] = args.chunk_size
    if args.chunk_overlap is not None:
        overrides["chunk_overlap"] = args.chunk_overlap
    settings = get_settings(**overrides)

    print(f"Mode: {settings.mode}  |  documents: {settings.docs_dir}")
    meta = KnowledgeBase(settings).build()
    print(f"Indexed {meta['num_chunks']} chunks from {meta['num_documents']} documents -> {settings.index_dir}")
    for source, count in meta["sources"].items():
        print(f"  - {source}: {count} chunks")


if __name__ == "__main__":
    main()
