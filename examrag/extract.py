"""Turn course PDFs into ordered text, headings, and code blocks."""

from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

import pymupdf

from examrag.config import PROJECT_DIR, SKIP_DIRS

_BOILER = re.compile(
    r"^(genspark|page \d+ of \d+|https?://\S+)$",
    re.IGNORECASE,
)
_LANG = {
    "python",
    "bash",
    "sh",
    "shell",
    "json",
    "yaml",
    "yml",
    "sql",
    "text",
    "dockerfile",
    "html",
    "css",
    "javascript",
    "js",
}


@dataclass
class Segment:
    kind: str  # heading, text, code
    text: str
    page: int


@dataclass
class Document:
    week: str
    source: str
    title: str
    path: Path
    segments: list[Segment] = field(default_factory=list)

    @property
    def pages(self) -> int:
        if not self.segments:
            return 0
        return max(segment.page for segment in self.segments)


def discover_pdfs(root: Path | None = None) -> list[Path]:
    root = root or PROJECT_DIR
    files: list[Path] = []
    for path in root.rglob("*.pdf"):
        parts = set(path.relative_to(root).parts)
        if parts & SKIP_DIRS:
            continue
        files.append(path)
    return sorted(files)


def extract_pdf(path: Path, root: Path | None = None) -> Document:
    root = root or PROJECT_DIR
    source = path.relative_to(root).as_posix()
    week = path.parent.name if path.parent != root else "Notes"
    doc = pymupdf.open(path)
    try:
        page_blocks = [_page_blocks(page, number) for number, page in enumerate(doc, start=1)]
    finally:
        doc.close()

    title, title_keys = _title_from(page_blocks, fallback=_filename_title(path))
    segments = _segments_from(page_blocks, title_keys)
    return Document(week=week, source=source, title=title, path=path, segments=segments)


def _filename_title(path: Path) -> str:
    return re.sub(r"\s+", " ", path.stem.replace("_", " ")).strip()


def _page_blocks(page: pymupdf.Page, number: int) -> list[dict]:
    height = float(page.rect.height)
    blocks: list[dict] = []
    for block in page.get_text("dict")["blocks"]:
        if block.get("type") != 0:
            continue
        y0 = float(block["bbox"][1])
        y1 = float(block["bbox"][3])
        if y0 < 32 or y0 > height - 58:
            continue
        text, size, mono_chars, total = _block_text(block)
        if not text or _BOILER.match(text.strip()):
            continue
        kind = _classify(text, size, mono_chars, total)
        if kind == "drop":
            continue
        blocks.append(
            {
                "y": y0,
                "y1": y1,
                "kind": kind,
                "text": text,
                "size": size,
                "page": number,
            }
        )
    blocks.sort(key=lambda item: item["y"])
    return blocks


def _block_text(block: dict) -> tuple[str, float, int, int]:
    lines: list[str] = []
    max_size = 0.0
    mono_chars = 0
    total = 0
    for line in block["lines"]:
        pieces: list[str] = []
        for span in line["spans"]:
            value = span["text"]
            if not value:
                continue
            pieces.append(value)
            total += len(value)
            max_size = max(max_size, float(span["size"]))
            if span["flags"] & 8:
                mono_chars += len(value)
        line_text = "".join(pieces).rstrip()
        if line_text.strip():
            lines.append(line_text)
    return "\n".join(lines).strip("\n"), max_size, mono_chars, total


def _classify(text: str, size: float, mono_chars: int, total: int) -> str:
    stripped = text.strip()
    if stripped.lower() in _LANG and len(stripped) < 20:
        return "lang"
    if total and mono_chars / total > 0.8:
        return "code"
    if size >= 14.5 and len(stripped) < 140 and "\n" not in stripped:
        return "heading"
    return "text"


def _title_from(page_blocks: list[list[dict]], fallback: str) -> tuple[str, set[str]]:
    first = page_blocks[0] if page_blocks else []
    if not first:
        return fallback, set()
    top = max(block["size"] for block in first)
    if top < 18:
        return fallback, set()
    lines = [
        re.sub(r"\s+", " ", block["text"]).strip()
        for block in first
        if block["size"] >= top - 0.6 and block["kind"] != "code"
    ]
    title = re.sub(r"\s+", " ", " ".join(lines)).strip(" -")
    if not title:
        return fallback, set()
    return title, {_key(line) for line in lines if line}


def _segments_from(page_blocks: list[list[dict]], title_keys: set[str]) -> list[Segment]:
    segments: list[Segment] = []
    for blocks in page_blocks:
        pending_lang: str | None = None
        prose: list[str] = []
        prose_page = 1
        prose_end = 0.0

        def flush_prose() -> None:
            nonlocal prose_end
            if not prose:
                return
            text = _clean_prose("".join(prose))
            prose.clear()
            prose_end = 0.0
            if text and _key(text) not in title_keys:
                segments.append(Segment("text", text, prose_page))

        for block in blocks:
            kind = block["kind"]
            if kind == "lang":
                flush_prose()
                pending_lang = block["text"].strip().lower()
                continue
            if kind == "heading":
                flush_prose()
                pending_lang = None
                heading = re.sub(r"\s+", " ", block["text"]).strip()
                if heading and _key(heading) not in title_keys and not _BOILER.match(heading):
                    segments.append(Segment("heading", heading, block["page"]))
                continue
            if kind == "code":
                flush_prose()
                lang = pending_lang or ""
                pending_lang = None
                code = block["text"].rstrip()
                if not code:
                    continue
                fenced = f"```{lang}\n{code}\n```"
                if segments and segments[-1].kind == "code" and segments[-1].page == block["page"]:
                    segments[-1].text = _merge_code(segments[-1].text, lang, code)
                else:
                    segments.append(Segment("code", fenced, block["page"]))
                continue

            pending_lang = None
            piece = re.sub(r"\s+", " ", block["text"]).strip()
            if not piece:
                continue
            if not prose:
                prose_page = block["page"]
                prose.append(piece)
            else:
                gap = block["y"] - prose_end
                joiner = "\n" if gap > 16 else " "
                if prose[-1].endswith("-") and piece[:1].islower():
                    prose[-1] = prose[-1][:-1]
                    joiner = ""
                prose.append(joiner + piece)
            prose_end = block["y1"]
        flush_prose()
    return segments


def _merge_code(previous: str, lang: str, code: str) -> str:
    first, _, rest = previous.partition("\n")
    body = rest.rsplit("\n", 1)[0]
    kept_lang = lang or first.removeprefix("```")
    return f"```{kept_lang}\n{body}\n{code}\n```"


def _clean_prose(text: str) -> str:
    text = text.replace("\u00a0", " ")
    text = re.sub(r"[ \t]+\n", "\n", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = re.sub(r"[ \t]{2,}", " ", text)
    return text.strip()


def _key(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", "", text.lower())
