"""Group extracted segments into retrieval passages."""

from __future__ import annotations

from examrag.config import CHUNK_MAX, CHUNK_TARGET
from examrag.extract import Document


def chunk_document(document: Document) -> list[dict]:
    chunks: list[dict] = []
    section = "Overview"
    buf: list[str] = []
    buf_len = 0
    page_start = 1
    page_end = 1

    def flush() -> None:
        nonlocal buf, buf_len
        text = "\n\n".join(part.strip() for part in buf if part.strip()).strip()
        buf = []
        buf_len = 0
        if len(text) < 40:
            return
        if (
            len(text) < 120
            and chunks
            and chunks[-1]["source"] == document.source
            and len(chunks[-1]["text"]) + len(text) < CHUNK_MAX + 400
        ):
            extra = text if chunks[-1]["section"] == section else f"{section}\n{text}"
            chunks[-1]["text"] = f"{chunks[-1]['text']}\n\n{extra}"
            chunks[-1]["page_end"] = page_end
            return
        chunks.append(
            {
                "week": document.week,
                "source": document.source,
                "title": document.title,
                "section": section,
                "page_start": page_start,
                "page_end": page_end,
                "text": text,
            }
        )

    for segment in document.segments:
        if segment.kind == "heading":
            if buf_len >= 500:
                flush()
            section = segment.text
            page_start = segment.page
            page_end = segment.page
            continue

        piece = segment.text.strip()
        if not piece:
            continue
        if buf and buf_len + len(piece) > CHUNK_MAX and buf_len >= 400:
            flush()
            page_start = segment.page
        if not buf:
            page_start = segment.page
        buf.append(piece)
        buf_len += len(piece) + 2
        page_end = segment.page
        if buf_len >= CHUNK_TARGET and segment.kind != "code":
            flush()

    flush()
    for index, chunk in enumerate(chunks):
        chunk["embed_text"] = _embed_text(chunk)
    return chunks


def chunk_documents(documents: list[Document]) -> list[dict]:
    chunks: list[dict] = []
    for document in documents:
        chunks.extend(chunk_document(document))
    for index, chunk in enumerate(chunks):
        chunk["id"] = f"c{index:04d}"
    return chunks


def _embed_text(chunk: dict) -> str:
    header = f"{chunk['title']}\n{chunk['section']}"
    return f"{header}\n{chunk['text']}"
