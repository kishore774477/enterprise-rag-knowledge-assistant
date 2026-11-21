"""The knowledge base: build, persist, reload and search the vector index."""

from __future__ import annotations

import json
import logging
import shutil
from collections import Counter
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from langchain_core.documents import Document
from langchain_core.vectorstores import InMemoryVectorStore

from .config import Settings
from .embeddings import HashingTfidfEmbeddings, backend_id, make_embeddings
from .loaders import SUPPORTED_EXTENSIONS, load_directory, split_documents

logger = logging.getLogger(__name__)

STORE_FILE = "vector_store.json"
META_FILE = "index_meta.json"
IDF_FILE = "offline_idf.json"


@dataclass
class SearchResult:
    document: Document
    score: float

    @property
    def source(self) -> str:
        return self.document.metadata.get("source", "unknown")

    @property
    def location(self) -> str:
        meta = self.document.metadata
        if meta.get("section"):
            return meta["section"]
        if meta.get("page"):
            return f"page {meta['page']}"
        return ""


class KnowledgeBase:
    def __init__(self, settings: Settings):
        self.settings = settings
        self.embeddings = make_embeddings(settings)
        self.store: InMemoryVectorStore | None = None
        self.meta: dict = {}

    # -- paths ---------------------------------------------------------------------
    @property
    def index_dir(self) -> Path:
        # One sub-folder per backend so switching modes never overwrites the other index.
        return self.settings.index_dir / self.settings.mode

    def _is_offline(self) -> bool:
        return isinstance(self.embeddings, HashingTfidfEmbeddings)

    # -- build / load ------------------------------------------------------------------
    def build(self) -> dict:
        """(Re)build the index from every supported file in the documents folder."""
        docs = load_directory(self.settings.docs_dir)
        chunks = split_documents(docs, self.settings.chunk_size, self.settings.chunk_overlap)
        if not chunks:
            raise ValueError(
                f"No readable documents found in {self.settings.docs_dir}. "
                f"Add files with extensions {sorted(SUPPORTED_EXTENSIONS)}."
            )

        if self._is_offline():
            self.embeddings.fit([c.page_content for c in chunks])

        self.store = InMemoryVectorStore.from_documents(chunks, self.embeddings)
        per_source = Counter(c.metadata["source"] for c in chunks)
        self.meta = {
            "backend": backend_id(self.settings),
            "built_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
            "num_documents": len(per_source),
            "num_chunks": len(chunks),
            "chunk_size": self.settings.chunk_size,
            "chunk_overlap": self.settings.chunk_overlap,
            "sources": dict(sorted(per_source.items())),
        }
        self._save()
        logger.info("Indexed %d chunks from %d documents", len(chunks), len(per_source))
        return self.meta

    def _save(self) -> None:
        self.index_dir.mkdir(parents=True, exist_ok=True)
        self.store.dump(str(self.index_dir / STORE_FILE))
        (self.index_dir / META_FILE).write_text(json.dumps(self.meta, indent=2), encoding="utf-8")
        if self._is_offline():
            self.embeddings.save(self.index_dir / IDF_FILE)

    def load(self) -> bool:
        """Load a saved index. Returns False if missing or built with another embedding backend."""
        meta_path = self.index_dir / META_FILE
        store_path = self.index_dir / STORE_FILE
        if not (meta_path.exists() and store_path.exists()):
            return False
        meta = json.loads(meta_path.read_text(encoding="utf-8"))
        if meta.get("backend") != backend_id(self.settings):
            logger.info("Index was built with %s; current backend differs, rebuild needed", meta.get("backend"))
            return False
        if self._is_offline():
            idf_path = self.index_dir / IDF_FILE
            if not idf_path.exists():
                return False
            self.embeddings = HashingTfidfEmbeddings.load(idf_path)
        self.store = InMemoryVectorStore.load(str(store_path), self.embeddings)
        self.meta = meta
        return True

    def ensure_ready(self) -> dict:
        """Load the saved index, or build it if there is none / it is stale."""
        if self.store is None and not self.load():
            self.build()
        return self.meta

    # -- documents -----------------------------------------------------------------------
    def add_files(self, paths: list[str | Path]) -> tuple[list[str], dict]:
        """Copy uploaded files into the documents folder and rebuild the index."""
        self.settings.docs_dir.mkdir(parents=True, exist_ok=True)
        added = []
        for p in paths:
            p = Path(p)
            if p.suffix.lower() not in SUPPORTED_EXTENSIONS:
                raise ValueError(f"Unsupported file type: {p.name}")
            shutil.copy2(p, self.settings.docs_dir / p.name)
            added.append(p.name)
        return added, self.build()

    def remove_document(self, name: str) -> dict:
        target = (self.settings.docs_dir / name).resolve()
        if target.parent != self.settings.docs_dir.resolve() or not target.exists():
            raise FileNotFoundError(f"No such document: {name}")
        target.unlink()
        return self.build()

    # -- search ---------------------------------------------------------------------------
    def search(self, query: str, k: int | None = None) -> list[SearchResult]:
        self.ensure_ready()
        hits = self.store.similarity_search_with_score(query, k=k or self.settings.top_k)
        return [SearchResult(doc, float(score)) for doc, score in hits]
