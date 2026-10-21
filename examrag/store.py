"""Build and load the hybrid search index."""

from __future__ import annotations

import hashlib
import json
import re
import threading
from pathlib import Path

import numpy as np
from rank_bm25 import BM25Okapi

from examrag.chunk import chunk_documents
from examrag.config import EMBED_MODEL, INDEX_DIR, PROJECT_DIR
from examrag.extract import discover_pdfs, extract_pdf

_TOKEN = re.compile(r"[a-z0-9_]+", re.IGNORECASE)
_LOCK = threading.Lock()
_INDEX: "Index | None" = None


def tokenize(text: str) -> list[str]:
    return _TOKEN.findall(text.lower())


class Index:
    def __init__(self, index_dir: Path | None = None, docs_root: Path | None = None):
        self.index_dir = index_dir or INDEX_DIR
        self.docs_root = docs_root or PROJECT_DIR
        self.chunks: list[dict] = []
        self.vectors: np.ndarray | None = None
        self._bm25: BM25Okapi | None = None
        self._tokens: list[list[str]] = []
        self._model = None
        self.manifest: dict = {}

    @property
    def ready(self) -> bool:
        return self.vectors is not None and bool(self.chunks)

    def load(self) -> bool:
        chunks_path = self.index_dir / "chunks.json"
        vectors_path = self.index_dir / "vectors.npy"
        manifest_path = self.index_dir / "manifest.json"
        if not (chunks_path.exists() and vectors_path.exists() and manifest_path.exists()):
            return False
        self.chunks = json.loads(chunks_path.read_text())
        self.vectors = np.load(vectors_path)
        self.manifest = json.loads(manifest_path.read_text())
        self._fit_bm25()
        return self.ready

    def build(self, force: bool = False, progress=print) -> dict:
        files = discover_pdfs(self.docs_root)
        fingerprint = _fingerprint(files)
        if (
            not force
            and self.load()
            and self.manifest.get("fingerprint") == fingerprint
            and len(self.vectors) == len(self.chunks)
        ):
            progress(f"Index is current: {len(self.chunks)} passages, {self.manifest.get('documents', 0)} documents.")
            return self.stats()

        progress(f"Reading {len(files)} PDFs...")
        documents = []
        for path in files:
            document = extract_pdf(path, self.docs_root)
            documents.append(document)
            progress(f"  {document.source}: {document.title} ({len(document.segments)} blocks)")

        chunks = chunk_documents(documents)
        if not chunks:
            raise RuntimeError("No text could be extracted from the PDFs.")

        progress(f"Embedding {len(chunks)} passages with {EMBED_MODEL}...")
        model = self._embedding_model()
        matrix = np.vstack(list(model.embed(chunk["embed_text"] for chunk in chunks))).astype(np.float32)
        matrix = _normalize(matrix)

        self.index_dir.mkdir(parents=True, exist_ok=True)
        stored = [{key: value for key, value in chunk.items() if key != "embed_text"} for chunk in chunks]
        for chunk, stored_chunk in zip(chunks, stored):
            stored_chunk["embed_text"] = chunk["embed_text"]
        (self.index_dir / "chunks.json").write_text(json.dumps(stored))
        np.save(self.index_dir / "vectors.npy", matrix)
        manifest = {
            "fingerprint": fingerprint,
            "documents": len(documents),
            "chunks": len(stored),
            "embed_model": EMBED_MODEL,
            "titles": [document.title for document in documents],
        }
        (self.index_dir / "manifest.json").write_text(json.dumps(manifest, indent=2))

        self.chunks = stored
        self.vectors = matrix
        self.manifest = manifest
        self._fit_bm25()
        progress(f"Indexed {len(stored)} passages from {len(documents)} documents.")
        return self.stats()

    def stats(self) -> dict:
        return {
            "ready": self.ready,
            "chunks": len(self.chunks),
            "documents": self.manifest.get("documents", 0),
            "embed_model": self.manifest.get("embed_model", EMBED_MODEL),
        }

    def library(self) -> list[dict]:
        grouped: dict[str, list[dict]] = {}
        for chunk in self.chunks:
            docs = grouped.setdefault(chunk["week"], {})
            item = docs.setdefault(
                chunk["source"],
                {
                    "week": chunk["week"],
                    "source": chunk["source"],
                    "title": chunk["title"],
                    "chunks": 0,
                    "page_end": 0,
                },
            )
            item["chunks"] += 1
            item["page_end"] = max(item["page_end"], chunk["page_end"])
        weeks = []
        for week in sorted(grouped):
            docs = sorted(grouped[week].values(), key=lambda item: item["title"].lower())
            weeks.append({"week": week, "documents": docs})
        return weeks

    def search(
        self,
        query: str,
        k: int = 6,
        week: str | None = None,
        source: str | None = None,
    ) -> list[dict]:
        if not self.ready:
            raise RuntimeError("The index is empty. Build it before searching.")
        query = query.strip()
        if not query:
            return []

        allowed = [
            index
            for index, chunk in enumerate(self.chunks)
            if (not week or chunk["week"] == week) and (not source or chunk["source"] == source)
        ]
        if not allowed:
            return []

        model = self._embedding_model()
        query_vector = np.asarray(list(model.query_embed([query]))[0], dtype=np.float32)
        norm = np.linalg.norm(query_vector)
        if norm:
            query_vector = query_vector / norm
        dense_scores = self.vectors @ query_vector

        query_tokens = tokenize(query)
        if query_tokens and self._bm25 is not None:
            lexical_scores = self._bm25.get_scores(query_tokens)
        else:
            lexical_scores = np.zeros(len(self.chunks), dtype=np.float32)

        dense_rank = _rank([index for index in allowed], dense_scores)
        lexical_rank = _rank([index for index in allowed if lexical_scores[index] > 0], lexical_scores)
        fused = _reciprocal_rank_fusion([dense_rank, lexical_rank])
        if not fused and dense_rank:
            fused = {index: 1.0 / (index_rank + 1) for index_rank, index in enumerate(dense_rank)}

        ordered = sorted(fused, key=fused.get, reverse=True)[:k]
        results = []
        for index in ordered:
            chunk = dict(self.chunks[index])
            chunk.pop("embed_text", None)
            chunk["score"] = round(float(fused[index]), 6)
            results.append(chunk)
        return results

    def neighbors(self, chunk_id: str) -> list[dict]:
        ids = [chunk["id"] for chunk in self.chunks]
        if chunk_id not in ids:
            return []
        center = ids.index(chunk_id)
        source = self.chunks[center]["source"]
        picked = []
        for index in range(max(0, center - 1), min(len(self.chunks), center + 2)):
            if self.chunks[index]["source"] == source:
                chunk = dict(self.chunks[index])
                chunk.pop("embed_text", None)
                picked.append(chunk)
        return picked

    def _fit_bm25(self) -> None:
        self._tokens = [tokenize(chunk.get("embed_text") or chunk["text"]) for chunk in self.chunks]
        usable = [tokens if tokens else ["empty"] for tokens in self._tokens]
        self._bm25 = BM25Okapi(usable)

    def _embedding_model(self):
        if self._model is None:
            from fastembed import TextEmbedding

            self._model = TextEmbedding(EMBED_MODEL)
        return self._model


def get_index() -> Index:
    global _INDEX
    if _INDEX is None:
        _INDEX = Index()
    return _INDEX


def _fingerprint(files: list[Path]) -> str:
    digest = hashlib.sha256()
    for path in files:
        stat = path.stat()
        digest.update(f"{path}|{stat.st_mtime_ns}|{stat.st_size}\n".encode())
    return digest.hexdigest()


def _normalize(matrix: np.ndarray) -> np.ndarray:
    norms = np.linalg.norm(matrix, axis=1, keepdims=True)
    return matrix / np.clip(norms, 1e-9, None)


def _rank(indexes: list[int], scores: np.ndarray) -> list[int]:
    return sorted(indexes, key=lambda index: float(scores[index]), reverse=True)


def _reciprocal_rank_fusion(rankings: list[list[int]], k: int = 60) -> dict[int, float]:
    fused: dict[int, float] = {}
    for ranking in rankings:
        for place, index in enumerate(ranking):
            fused[index] = fused.get(index, 0.0) + 1.0 / (k + place + 1)
    return fused
