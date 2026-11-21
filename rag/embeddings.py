"""Embedding backends.

* OpenAI mode uses ``OpenAIEmbeddings`` (e.g. ``text-embedding-3-small``).
* Offline mode uses ``HashingTfidfEmbeddings``: a dependency-free, deterministic
  TF-IDF embedding (unigrams + bigrams hashed into a fixed-size vector). It needs no
  API key or model download, so the whole app runs on any laptop.
"""

from __future__ import annotations

import json
import math
import zlib
from collections import Counter
from pathlib import Path

import numpy as np
from langchain_core.embeddings import Embeddings

from .config import Settings
from .text import tokenize


class HashingTfidfEmbeddings(Embeddings):
    def __init__(self, n_features: int = 4096, idf: list[float] | None = None):
        self.n_features = n_features
        self.idf = np.asarray(idf, dtype=np.float32) if idf is not None else None

    # -- features -------------------------------------------------------------
    def _features(self, text: str) -> list[str]:
        tokens = tokenize(text)
        bigrams = [f"{a}_{b}" for a, b in zip(tokens, tokens[1:])]
        return tokens + bigrams

    def _bucket(self, feature: str) -> int:
        return zlib.crc32(feature.encode("utf-8")) % self.n_features

    def _vector(self, text: str) -> np.ndarray:
        vec = np.zeros(self.n_features, dtype=np.float32)
        for feature, count in Counter(self._features(text)).items():
            vec[self._bucket(feature)] += 1.0 + math.log(count)
        if self.idf is not None:
            vec *= self.idf
        norm = float(np.linalg.norm(vec))
        return vec / norm if norm else vec

    # -- fitting / persistence ---------------------------------------------------
    def fit(self, texts: list[str]) -> "HashingTfidfEmbeddings":
        """Learn inverse-document-frequency weights from the corpus."""
        df = np.zeros(self.n_features, dtype=np.float32)
        for text in texts:
            for bucket in {self._bucket(f) for f in self._features(text)}:
                df[bucket] += 1
        n = max(len(texts), 1)
        self.idf = (np.log((1 + n) / (1 + df)) + 1.0).astype(np.float32)
        return self

    def save(self, path: Path) -> None:
        payload = {"n_features": self.n_features, "idf": self.idf.tolist() if self.idf is not None else None}
        path.write_text(json.dumps(payload), encoding="utf-8")

    @classmethod
    def load(cls, path: Path) -> "HashingTfidfEmbeddings":
        payload = json.loads(path.read_text(encoding="utf-8"))
        return cls(n_features=payload["n_features"], idf=payload["idf"])

    def term_weight(self, token: str) -> float:
        """IDF weight of a single (already stemmed) token; used by the extractive answerer."""
        if self.idf is None:
            return 1.0
        return float(self.idf[self._bucket(token)])

    # -- LangChain Embeddings API ------------------------------------------------
    def embed_documents(self, texts: list[str]) -> list[list[float]]:
        return [self._vector(t).tolist() for t in texts]

    def embed_query(self, text: str) -> list[float]:
        return self._vector(text).tolist()


def backend_id(settings: Settings) -> str:
    """Identifies which embedding space an index was built in."""
    return f"openai:{settings.embedding_model}" if settings.use_openai else "offline:hashing-tfidf-v1"


def make_embeddings(settings: Settings) -> Embeddings:
    if settings.use_openai:
        from langchain_openai import OpenAIEmbeddings

        return OpenAIEmbeddings(
            model=settings.embedding_model,
            api_key=settings.openai_api_key,
            base_url=settings.openai_base_url,
        )
    return HashingTfidfEmbeddings()
