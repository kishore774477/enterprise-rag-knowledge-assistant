"""Load enterprise documents (PDF, DOCX, Markdown, TXT) and split them into chunks."""

from __future__ import annotations

import logging
from pathlib import Path

from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter

logger = logging.getLogger(__name__)

SUPPORTED_EXTENSIONS = {".pdf", ".docx", ".md", ".markdown", ".txt"}

_MD_HEADERS = [("#", "h1"), ("##", "h2"), ("###", "h3")]


def _read_text(path: Path) -> str:
    for encoding in ("utf-8", "utf-8-sig", "cp1252"):
        try:
            return path.read_text(encoding=encoding)
        except UnicodeDecodeError:
            continue
    return path.read_text(encoding="utf-8", errors="replace")


def load_file(path: str | Path) -> list[Document]:
    """Load one file into LangChain Documents (one per PDF page, one per other file)."""
    path = Path(path)
    ext = path.suffix.lower()
    base_meta = {"source": path.name, "file_type": ext.lstrip(".")}

    if ext == ".pdf":
        from pypdf import PdfReader

        reader = PdfReader(str(path))
        docs = []
        for page_no, page in enumerate(reader.pages, start=1):
            text = (page.extract_text() or "").strip()
            if text:
                docs.append(Document(page_content=text, metadata={**base_meta, "page": page_no}))
        if not docs:
            logger.warning("No extractable text in %s (scanned PDF?)", path.name)
        return docs

    if ext == ".docx":
        import docx2txt

        text = (docx2txt.process(str(path)) or "").strip()
        return [Document(page_content=text, metadata=base_meta)] if text else []

    if ext in {".md", ".markdown", ".txt"}:
        text = _read_text(path).strip()
        return [Document(page_content=text, metadata=base_meta)] if text else []

    raise ValueError(f"Unsupported file type: {path.name} (supported: {sorted(SUPPORTED_EXTENSIONS)})")


def load_directory(directory: str | Path) -> list[Document]:
    directory = Path(directory)
    if not directory.exists():
        raise FileNotFoundError(f"Documents folder not found: {directory}")
    docs: list[Document] = []
    for path in sorted(directory.rglob("*")):
        if path.is_file() and path.suffix.lower() in SUPPORTED_EXTENSIONS:
            try:
                docs.extend(load_file(path))
            except Exception as exc:  # one bad file must not break the whole index
                logger.error("Skipping %s: %s", path.name, exc)
    return docs


def _section_title(metadata: dict) -> str:
    return " > ".join(metadata[k] for k in ("h1", "h2", "h3") if metadata.get(k))


def split_documents(docs: list[Document], chunk_size: int = 800, chunk_overlap: int = 120) -> list[Document]:
    """Split documents into overlapping chunks.

    Markdown files are first split on headings so every chunk knows which section it came
    from; the section path is prepended to the chunk text, which helps retrieval a lot.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=chunk_size,
        chunk_overlap=chunk_overlap,
        separators=["\n\n", "\n", ". ", " ", ""],
    )
    md_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=_MD_HEADERS, strip_headers=True)

    chunks: list[Document] = []
    for doc in docs:
        if doc.metadata.get("file_type") in {"md", "markdown"}:
            sections = md_splitter.split_text(doc.page_content)
            for section in sections:
                title = _section_title(section.metadata)
                meta = {**doc.metadata, "section": title}
                for piece in splitter.split_text(section.page_content):
                    text = f"{title}\n{piece}" if title else piece
                    chunks.append(Document(page_content=text, metadata=dict(meta)))
        else:
            for piece in splitter.split_text(doc.page_content):
                chunks.append(Document(page_content=piece, metadata=dict(doc.metadata)))

    for i, chunk in enumerate(chunks):
        chunk.metadata["chunk_id"] = i
    return chunks
