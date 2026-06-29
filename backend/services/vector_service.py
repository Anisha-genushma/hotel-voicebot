"""
VectorService – FAISS index management.

Strategy:
  • On first run → build index, save to disk, never rebuild unless JSON changes.
  • On every subsequent start → load from disk in ~milliseconds (no API calls).
  • Change detection → MD5 hash of the JSON file stored alongside the index.
"""

import hashlib
import json
import logging
import os
from typing import List, Tuple, Dict, Any, Optional

import faiss
import numpy as np

from config import settings

logger = logging.getLogger(__name__)


def _md5_file(path: str) -> str:
    h = hashlib.md5()
    with open(path, "rb") as f:
        for chunk in iter(lambda: f.read(8192), b""):
            h.update(chunk)
    return h.hexdigest()


class VectorService:
    """Manages the FAISS index lifecycle for the hotel knowledge base."""

    def __init__(self):
        self._index: Optional[faiss.IndexFlatIP] = None   # inner-product (cosine after L2-norm)
        self._metadata: List[Dict[str, Any]] = []
        self._hash_path = settings.faiss_index_path + ".hash"

    # ── Public ────────────────────────────────────────────────────────────────

    def load_or_build_index(self):
        """
        Load existing FAISS index from disk, or build it from scratch
        if it doesn't exist or the knowledge base has changed.
        """
        kb_hash = _md5_file(settings.knowledge_base_path)

        if self._index_exists() and self._hash_matches(kb_hash):
            logger.info("FAISS index found and up-to-date. Loading from disk.")
            self._load_from_disk()
        else:
            if self._index_exists():
                logger.info("Knowledge base changed. Rebuilding FAISS index.")
            else:
                logger.info("No FAISS index found. Building for the first time.")
            self._build_and_save(kb_hash)

    def search(self, query: str, top_k: int = None) -> List[Dict[str, Any]]:
        """
        Embed the query, search FAISS, and return top_k metadata records
        whose similarity exceeds the configured threshold.
        """
        if self._index is None:
            raise RuntimeError("VectorService not initialised. Call load_or_build_index() first.")

        from backend.services.gemini_client import get_gemini_client
        k = top_k or settings.top_k
        client = get_gemini_client()

        query_vec = np.array([client.embed_query(query)], dtype="float32")
        faiss.normalize_L2(query_vec)

        scores, indices = self._index.search(query_vec, k)

        results = []
        for score, idx in zip(scores[0], indices[0]):
            if idx == -1:
                continue
            if float(score) < settings.similarity_threshold:
                logger.debug("Skipping chunk idx=%d score=%.3f (below threshold)", idx, score)
                continue
            chunk = dict(self._metadata[idx])
            chunk["_score"] = float(score)
            results.append(chunk)

        logger.info("Query: %r → %d chunks above threshold", query[:60], len(results))
        return results

    # ── Private ───────────────────────────────────────────────────────────────

    def _index_exists(self) -> bool:
        return (
            os.path.isfile(settings.faiss_index_path)
            and os.path.isfile(settings.metadata_path)
        )

    def _hash_matches(self, current_hash: str) -> bool:
        if not os.path.isfile(self._hash_path):
            return False
        with open(self._hash_path) as f:
            return f.read().strip() == current_hash

    def _load_from_disk(self):
        self._index = faiss.read_index(settings.faiss_index_path)
        with open(settings.metadata_path) as f:
            self._metadata = json.load(f)
        logger.info("Loaded FAISS index (%d vectors).", self._index.ntotal)

    def _build_and_save(self, kb_hash: str):
        from backend.services.gemini_client import get_gemini_client
        client = get_gemini_client()

        with open(settings.knowledge_base_path) as f:
            docs = json.load(f)

        vectors: List[List[float]] = []
        metadata: List[Dict[str, Any]] = []

        logger.info("Embedding %d knowledge-base documents…", len(docs))
        for i, doc in enumerate(docs):
            text = " | ".join([
                doc.get("category", ""),
                doc.get("title", ""),
                doc.get("content", ""),
                " ".join(doc.get("keywords", [])),
            ])
            vec = client.embed_text(text)
            vectors.append(vec)
            metadata.append({
                "id": doc.get("id"),
                "category": doc.get("category"),
                "title": doc.get("title"),
                "content": doc.get("content"),
                "keywords": doc.get("keywords", []),
            })
            if (i + 1) % 10 == 0:
                logger.info("  Embedded %d / %d", i + 1, len(docs))

        arr = np.array(vectors, dtype="float32")
        faiss.normalize_L2(arr)  # cosine similarity via inner product

        dim = arr.shape[1]
        index = faiss.IndexFlatIP(dim)
        index.add(arr)

        # Persist
        os.makedirs(os.path.dirname(settings.faiss_index_path), exist_ok=True)
        faiss.write_index(index, settings.faiss_index_path)
        with open(settings.metadata_path, "w") as f:
            json.dump(metadata, f, indent=2)
        with open(self._hash_path, "w") as f:
            f.write(kb_hash)

        self._index = index
        self._metadata = metadata
        logger.info("FAISS index built and saved (%d vectors, dim=%d).", len(vectors), dim)
