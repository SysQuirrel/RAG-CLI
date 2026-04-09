"""
embedder.py
-----------
Stage 5 — Embedding
Stage 6 — Vector Storage

Embeds chunks using a sentence-transformers model and stores
them in ChromaDB for persistent local retrieval.

Model choice — configured by Config.EMBEDDING_MODEL (default: BAAI/bge-base-en-v1.5):
    - Better semantic recall than very small embedding models
    - Works on CPU and GPU (GPU optional)
    - Requires re-ingestion when changed, because vector spaces are model-specific

ChromaDB choice:
  - Embedded (no server process), stores to disk
  - Supports metadata filtering at query time
  - Handles collections of up to ~500k chunks comfortably on 8GB RAM
  - Simple Python-native API
"""

from __future__ import annotations

import logging
from typing import Optional

import chromadb
from chromadb.api import ClientAPI
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

from config import Config

logger = logging.getLogger(__name__)


# ── Singleton helpers (avoid reloading model / reopening DB) ──────────────────

_model: Optional[SentenceTransformer] = None
_chroma_client: Optional[ClientAPI] = None


def _get_model() -> SentenceTransformer:
    global _model
    if _model is None:
        logger.info("Loading embedding model: %s", Config.EMBEDDING_MODEL)
        _model = SentenceTransformer(Config.EMBEDDING_MODEL)
    return _model


def _get_client() -> ClientAPI:
    global _chroma_client
    if _chroma_client is None:
        logger.info("Opening ChromaDB at: %s", Config.CHROMA_DB_PATH)
        _chroma_client = chromadb.PersistentClient(
            path=Config.CHROMA_DB_PATH,
            settings=Settings(anonymized_telemetry=False),
        )
    return _chroma_client


def get_collection() -> chromadb.Collection:
    """
    Get (or create) the ChromaDB collection.
    Using cosine similarity — standard for semantic search.
    """
    client = _get_client()
    collection = client.get_or_create_collection(
        name=Config.CHROMA_COLLECTION,
        metadata={"hnsw:space": "cosine"},   # cosine similarity metric
    )
    return collection


# ── Embedding ─────────────────────────────────────────────────────────────────

def embed_chunks(chunks: list[dict]) -> list[dict]:
    """
    Compute embeddings for all chunks in batches.
    Returns the same chunks with an 'embedding' key added.

    Batching is important: sentence-transformers is faster in batches
    due to padding efficiency, and we avoid OOM on large corpora.
    """
    model = _get_model()
    texts = [c["text"] for c in chunks]
    all_embeddings: list[list[float]] = []

    batch_size = Config.EMBEDDING_BATCH_SIZE
    total_batches = (len(texts) + batch_size - 1) // batch_size

    logger.info("Embedding %d chunks in %d batches…", len(texts), total_batches)

    for i in tqdm(range(0, len(texts), batch_size), desc="Embedding", unit="batch"):
        batch = texts[i : i + batch_size]
        # encode() returns a numpy array; convert to plain Python list for Chroma
        vecs = model.encode(
            batch,
            batch_size=batch_size,
            show_progress_bar=False,
            normalize_embeddings=True,  # L2-normalise → cosine = dot product
        )
        all_embeddings.extend(vecs.tolist())

    # Attach embeddings to chunks in-place
    for chunk, embedding in zip(chunks, all_embeddings):
        chunk["embedding"] = embedding

    return chunks


# ── Storage ───────────────────────────────────────────────────────────────────

def _build_chroma_metadata(chunk: dict) -> dict:
    """
    Flatten chunk metadata into a Chroma-compatible dict.
    Chroma only supports str, int, float, bool values — no nested dicts.
    """
    return {
        "url":             chunk["url"],
        "domain":          chunk["domain"],
        "title":           chunk["title"],
        "nearest_heading": chunk["nearest_heading"],
        "chunk_index":     chunk["chunk_index"],
        "total_chunks":    chunk["total_chunks"],
        "doc_id":          chunk["doc_id"],
        "crawled_at":      chunk["metadata"].get("crawled_at", ""),
        "language":        chunk["metadata"].get("language", "en"),
        "word_count":      chunk["metadata"].get("word_count", 0),
    }


def store_chunks(chunks: list[dict], batch_size: int = 500) -> int:
    """
    Upsert embedded chunks into ChromaDB.
    Uses upsert (not add) so re-running the pipeline on updated content
    refreshes existing entries rather than creating duplicates.

    Returns the total number of chunks stored.
    """
    if not chunks:
        logger.warning("No chunks to store")
        return 0

    # Embed if not already done
    if "embedding" not in chunks[0]:
        chunks = embed_chunks(chunks)

    collection = get_collection()
    stored = 0

    for i in tqdm(range(0, len(chunks), batch_size), desc="Storing", unit="batch"):
        batch = chunks[i : i + batch_size]
        collection.upsert(
            ids=[c["chunk_id"] for c in batch],
            embeddings=[c["embedding"] for c in batch],
            documents=[c["text"] for c in batch],       # full text stored in Chroma
            metadatas=[_build_chroma_metadata(c) for c in batch],
        )
        stored += len(batch)

    logger.info("Stored %d chunks in ChromaDB collection '%s'",
                stored, Config.CHROMA_COLLECTION)
    return stored


def embed_and_store(chunks: list[dict]) -> int:
    """Convenience wrapper: embed then store in one call."""
    chunks = embed_chunks(chunks)
    return store_chunks(chunks)
