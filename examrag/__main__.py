"""Command line for indexing and asking the course notes."""

from __future__ import annotations

import argparse
import os
import sys

from examrag.config import CHAT_MODEL, PROJECT_DIR


def main(argv: list[str] | None = None) -> None:
    parser = argparse.ArgumentParser(prog="examrag", description="Query the exam-prep course notes.")
    sub = parser.add_subparsers(dest="command", required=True)

    ingest = sub.add_parser("ingest", help="Extract, chunk, and index every PDF")
    ingest.add_argument("--force", action="store_true", help="Rebuild even if the PDFs have not changed")

    search = sub.add_parser("search", help="Show the passages a question would retrieve")
    search.add_argument("question", nargs="+")
    search.add_argument("--week")
    search.add_argument("--k", type=int, default=5)

    ask = sub.add_parser("ask", help="Ask the tutor")
    ask.add_argument("question", nargs="+")
    ask.add_argument("--week")
    ask.add_argument("--mode", choices=["ask", "quiz", "outline"], default="ask")

    serve = sub.add_parser("serve", help="Open the local study desk")
    serve.add_argument("--host", default="127.0.0.1")
    serve.add_argument("--port", type=int, default=8765)
    serve.add_argument("--rebuild", action="store_true")

    args = parser.parse_args(argv)
    if args.command == "ingest":
        _ingest(args.force)
    elif args.command == "search":
        _search(" ".join(args.question), args.week, args.k)
    elif args.command == "ask":
        _ask(" ".join(args.question), args.week, args.mode)
    elif args.command == "serve":
        _serve(args.host, args.port, args.rebuild)


def _ingest(force: bool) -> None:
    from examrag.store import get_index

    get_index().build(force=force)


def _search(question: str, week: str | None, k: int) -> None:
    from examrag.passages import cite_label
    from examrag.store import get_index

    index = get_index()
    if not index.load():
        raise SystemExit("No index yet. Run: python -m examrag ingest")
    hits = index.search(question, k=k, week=week)
    if not hits:
        print("No matching passages.")
        return
    for number, hit in enumerate(hits, start=1):
        print(f"\n[{number}] {cite_label(hit)}  ({hit['id']}, score {hit['score']})")
        preview = hit["text"].strip().replace("\n", " ")
        print(preview[:420] + ("…" if len(preview) > 420 else ""))


def _ask(question: str, week: str | None, mode: str) -> None:
    from examrag.agent import answer
    from examrag.store import get_index

    index = get_index()
    if not index.load():
        raise SystemExit("No index yet. Run: python -m examrag ingest")
    print(f"Using {CHAT_MODEL}. The first reply loads the model and can take a minute.\n")
    for event in answer(question, mode=mode, week=week, index=index):
        kind = event["type"]
        if kind == "status":
            print(f"… {event['text']}", file=sys.stderr)
        elif kind == "token":
            print(event["text"], end="", flush=True)
        elif kind == "error":
            print(f"\n{event['text']}", file=sys.stderr)
        elif kind == "done":
            sources = event.get("sources")
            print("\n")
        elif kind == "sources" and event.get("sources"):
            _pending_sources = event["sources"]
    if "_pending_sources" in locals() and _pending_sources:
        print("Sources")
        for source in _pending_sources:
            print(f"  [{source['n']}] {source['label']}")


def _serve(host: str, port: int, rebuild: bool) -> None:
    import uvicorn

    from examrag.store import get_index

    os.chdir(PROJECT_DIR)
    index = get_index()
    if rebuild or not index.load():
        index.build(force=rebuild)
    else:
        print(f"Index ready: {len(index.chunks)} passages from {index.manifest.get('documents', 0)} documents.")
    print(f"Study desk at http://{host}:{port}")
    uvicorn.run("examrag.server:app", host=host, port=port, reload=False)


if __name__ == "__main__":
    main()
