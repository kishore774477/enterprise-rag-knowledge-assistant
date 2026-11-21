"""Central configuration, read from environment variables / a `.env` file."""

from __future__ import annotations

import os
from dataclasses import dataclass, field, replace
from pathlib import Path

from dotenv import load_dotenv

ROOT_DIR = Path(__file__).resolve().parent.parent
load_dotenv(ROOT_DIR / ".env")

VALID_MODES = ("auto", "openai", "offline")


def _path(env_name: str, default: str) -> Path:
    value = Path(os.getenv(env_name, default))
    return value if value.is_absolute() else ROOT_DIR / value


def _int(env_name: str, default: int) -> int:
    try:
        return int(os.getenv(env_name, default))
    except ValueError:
        return default


def _float(env_name: str, default: float) -> float:
    try:
        return float(os.getenv(env_name, default))
    except ValueError:
        return default


@dataclass(frozen=True)
class Settings:
    docs_dir: Path = field(default_factory=lambda: _path("RAG_DOCS_DIR", "data/documents"))
    index_dir: Path = field(default_factory=lambda: _path("RAG_INDEX_DIR", "storage/index"))
    eval_file: Path = field(default_factory=lambda: _path("RAG_EVAL_FILE", "data/eval/qa_dataset.json"))
    requested_mode: str = field(default_factory=lambda: os.getenv("RAG_MODE", "auto").strip().lower())
    openai_api_key: str | None = field(default_factory=lambda: os.getenv("OPENAI_API_KEY") or None)
    openai_base_url: str | None = field(default_factory=lambda: os.getenv("OPENAI_BASE_URL") or None)
    chat_model: str = field(default_factory=lambda: os.getenv("OPENAI_CHAT_MODEL", "gpt-4o-mini"))
    embedding_model: str = field(
        default_factory=lambda: os.getenv("OPENAI_EMBEDDING_MODEL", "text-embedding-3-small")
    )
    temperature: float = field(default_factory=lambda: _float("RAG_TEMPERATURE", 0.0))
    chunk_size: int = field(default_factory=lambda: _int("RAG_CHUNK_SIZE", 800))
    chunk_overlap: int = field(default_factory=lambda: _int("RAG_CHUNK_OVERLAP", 120))
    top_k: int = field(default_factory=lambda: _int("RAG_TOP_K", 4))
    # Offline mode only: below this cosine similarity the question is treated as unanswerable.
    min_relevance: float = field(default_factory=lambda: _float("RAG_MIN_RELEVANCE", 0.12))

    def __post_init__(self) -> None:
        if self.requested_mode not in VALID_MODES:
            raise ValueError(f"RAG_MODE must be one of {VALID_MODES}, got {self.requested_mode!r}")
        if self.requested_mode == "openai" and not self.openai_api_key:
            raise ValueError("RAG_MODE=openai requires OPENAI_API_KEY to be set (see .env.example)")

    @property
    def mode(self) -> str:
        """The mode actually in use: 'openai' or 'offline'."""
        if self.requested_mode == "offline":
            return "offline"
        return "openai" if self.openai_api_key else "offline"

    @property
    def use_openai(self) -> bool:
        return self.mode == "openai"

    def with_overrides(self, **kwargs) -> "Settings":
        return replace(self, **kwargs)


def get_settings(**overrides) -> Settings:
    settings = Settings()
    return settings.with_overrides(**overrides) if overrides else settings
