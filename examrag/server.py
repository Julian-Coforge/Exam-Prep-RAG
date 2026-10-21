"""Local study desk for the course notes."""

from __future__ import annotations

import json
from pathlib import Path

from fastapi import FastAPI
from fastapi.responses import FileResponse, StreamingResponse
from fastapi.staticfiles import StaticFiles
from pydantic import BaseModel, Field

from examrag.agent import answer
from examrag.config import CHAT_MODEL, PACKAGE_DIR
from examrag.store import get_index

app = FastAPI(title="Exam prep")
_STATIC = PACKAGE_DIR / "static"


class ChatRequest(BaseModel):
    message: str = Field(min_length=1, max_length=8000)
    history: list[dict] = Field(default_factory=list)
    mode: str = "ask"
    week: str | None = None
    source: str | None = None


class RebuildRequest(BaseModel):
    force: bool = True


@app.get("/")
def home():
    return FileResponse(_STATIC / "index.html")


@app.get("/api/status")
def status():
    index = get_index()
    stats = index.stats()
    stats["model"] = CHAT_MODEL
    return stats


@app.get("/api/library")
def library():
    return {"weeks": get_index().library()}


@app.post("/api/rebuild")
def rebuild(body: RebuildRequest):
    stats = get_index().build(force=body.force)
    stats["model"] = CHAT_MODEL
    return stats


@app.post("/api/chat")
def chat(body: ChatRequest):
    def events():
        for event in answer(
            body.message,
            history=body.history,
            mode=body.mode,
            week=body.week,
            source=body.source,
        ):
            yield f"data: {json.dumps(event)}\n\n"

    return StreamingResponse(events(), media_type="text/event-stream")


def mount_static() -> None:
    app.mount("/static", StaticFiles(directory=_STATIC), name="static")


mount_static()
