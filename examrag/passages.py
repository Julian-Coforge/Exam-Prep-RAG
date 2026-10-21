"""Citation labels shared by retrieval and the tutor."""

from __future__ import annotations


def cite_label(chunk: dict) -> str:
    start = chunk["page_start"]
    end = chunk["page_end"]
    pages = f"p.{start}" if start == end else f"p.{start}–{end}"
    section = chunk.get("section") or ""
    if section and section.lower() != "overview":
        return f"{chunk['week']} · {chunk['title']} · {pages} · {section}"
    return f"{chunk['week']} · {chunk['title']} · {pages}"


def format_passages(chunks: list[dict], numbers: dict[str, int] | None = None) -> str:
    blocks = []
    for chunk in chunks:
        number = numbers[chunk["id"]] if numbers else chunk.get("n", 0)
        body = chunk["text"].strip()
        if len(body) > 1800:
            body = body[:1800].rstrip() + "…"
        blocks.append(f"[{number}] id={chunk['id']} {cite_label(chunk)}\n{body}")
    return "\n\n".join(blocks)
