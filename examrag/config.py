"""Paths and model settings for the exam-prep retrieval pipeline."""

from __future__ import annotations

import os
from pathlib import Path

PACKAGE_DIR = Path(__file__).resolve().parent
PROJECT_DIR = PACKAGE_DIR.parent
INDEX_DIR = PROJECT_DIR / ".index"

OLLAMA_HOST = os.environ.get("OLLAMA_HOST", "http://127.0.0.1:11434").rstrip("/")
CHAT_MODEL = os.environ.get("EXAMRAG_MODEL", "gemma4:26b")
EMBED_MODEL = "BAAI/bge-small-en-v1.5"

CHUNK_TARGET = 1400
CHUNK_MAX = 1800
RETRIEVAL_K = 6
SKIP_DIRS = {".venv", ".index", "examrag", "__pycache__", ".git"}
