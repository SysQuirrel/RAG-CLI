"""
retriever.py
------------
Stage 6 — Retrieval

Converts a user query into an embedding and retrieves the most relevant
chunks from ChromaDB. Supports:
  - Basic vector search (cosine similarity)
  - Optional domain/language filtering (metadata pre-filtering)
  - Score thresholding (discard low-confidence results)
  - Result deduplication (same text from different chunk IDs)

The retrieve() function is designed to be called directly by an LLM
orchestration layer — it returns clean dicts ready to be formatted
into a prompt context block.

Hybrid search note:
  ChromaDB does not natively support BM25. For hybrid retrieval at scale,
  use Qdrant (which has built-in sparse+dense hybrid) or add a separate
  BM25 index (rank_bm25 library) and merge results with RRF. The structure
  here makes it easy to slot that in later.
"""

from __future__ import annotations

import logging
from typing import Any, Mapping, Optional

from embedder import _get_model, get_collection
from config import Config

logger = logging.getLogger(__name__)


# ── Result container ──────────────────────────────────────────────────────────

class RetrievalResult:
    """Structured retrieval result for a single chunk."""

    def __init__(self, chunk_id: str, text: str, score: float, metadata: Mapping[str, Any]):
        self.chunk_id = chunk_id
        self.text = text
        self.score = score          # cosine similarity (0–1, higher = more relevant)
        self.url = str(metadata.get("url", ""))
        self.title = str(metadata.get("title", ""))
        self.nearest_heading = str(metadata.get("nearest_heading", ""))
        self.domain = str(metadata.get("domain", ""))
        self.metadata = metadata

    def to_context_block(self) -> str:
        """
        Format this chunk as a context block for LLM consumption.
        The format is designed to give the model clear source attribution.
        """
        heading_line = f"Section: {self.nearest_heading}\n" if self.nearest_heading else ""
        return (
            f"[Source: {self.title} | {self.url}]\n"
            f"{heading_line}"
            f"{self.text}"
        )

    def __repr__(self) -> str:
        return (f"RetrievalResult(score={self.score:.3f}, "
                f"title={self.title!r}, chars={len(self.text)})")


# ── Core retrieval ────────────────────────────────────────────────────────────

def retrieve(
    query: str,
    top_k: int = Config.RETRIEVAL_TOP_K,
    min_score: float = Config.RETRIEVAL_MIN_SCORE,
    domain_filter: Optional[str] = None,
    language_filter: Optional[str] = None,
) -> list[RetrievalResult]:
    """
    Retrieve the most semantically relevant chunks for a query.

    Args:
        query:           Natural language question or search phrase.
        top_k:           Number of results to return (after filtering).
        min_score:       Minimum cosine similarity threshold (0–1).
                         Chunks below this are discarded as irrelevant.
        domain_filter:   If set, only return chunks from this domain.
                         E.g. "docs.python.org"
        language_filter: If set, only return chunks in this language.
                         E.g. "en"

    Returns:
        List of RetrievalResult objects, sorted by descending relevance.
    """
    if not query.strip():
        raise ValueError("Query cannot be empty")

    model = _get_model()
    collection = get_collection()

    # Check if collection has any data
    count = collection.count()
    if count == 0:
        logger.warning("ChromaDB collection is empty — run the pipeline first")
        return []

    logger.info("Retrieving top-%d for query: %r (collection: %d chunks)",
                top_k, query, count)

    # ── Embed query ───────────────────────────────────────────────────────
    query_embedding = model.encode(
        query,
        normalize_embeddings=True,
        show_progress_bar=False,
    ).tolist()

    # ── Build metadata filter ─────────────────────────────────────────────
    where_filter: Optional[dict] = None
    filters = []
    if domain_filter:
        filters.append({"domain": {"$eq": domain_filter}})
    if language_filter:
        filters.append({"language": {"$eq": language_filter}})

    if len(filters) == 1:
        where_filter = filters[0]
    elif len(filters) > 1:
        where_filter = {"$and": filters}

    # ── Query Chroma ──────────────────────────────────────────────────────
    # Fetch more candidates than top_k to allow for post-filtering
    n_candidates = min(top_k * 3, count)

    query_kwargs = {
        "query_embeddings": [query_embedding],
        "n_results": n_candidates,
        "include": ["documents", "metadatas", "distances"],
    }
    if where_filter:
        query_kwargs["where"] = where_filter

    results = collection.query(**query_kwargs)

    # ── Parse and filter results ──────────────────────────────────────────
    ids = (results.get("ids") or [[]])[0]
    documents = (results.get("documents") or [[]])[0]
    metadatas = (results.get("metadatas") or [[]])[0]
    distances = (results.get("distances") or [[]])[0]   # Chroma returns cosine distance (0=identical)

    retrieval_results: list[RetrievalResult] = []
    seen_texts: set[str] = set()

    for chunk_id, text, metadata, distance in zip(ids, documents, metadatas, distances):
        metadata_map = dict(metadata or {})
        # Convert cosine distance → similarity score
        # Chroma cosine distance: 0 = identical, 2 = opposite
        # similarity = 1 - (distance / 2) maps this to [0, 1]
        score = 1.0 - (distance / 2.0)

        if score < min_score:
            logger.debug("Chunk %s below score threshold (%.3f < %.3f)",
                         chunk_id, score, min_score)
            continue

        # Deduplicate: skip near-identical text (same content, different chunk ID)
        text_key = text[:100].strip().lower()
        if text_key in seen_texts:
            logger.debug("Duplicate text skipped: %s", chunk_id)
            continue
        seen_texts.add(text_key)

        retrieval_results.append(RetrievalResult(
            chunk_id=str(chunk_id),
            text=str(text),
            score=score,
            metadata=metadata_map,
        ))

        if len(retrieval_results) >= top_k:
            break

    logger.info(
        "Retrieved %d results (from %d candidates, min_score=%.2f)",
        len(retrieval_results), n_candidates, min_score
    )
    return retrieval_results


def format_context(results: list[RetrievalResult], max_chars: int = 6000) -> str:
    """
    Format retrieval results into a single context string for an LLM prompt.
    Truncates at max_chars to stay within typical context windows.

    Args:
        results:   List of RetrievalResult objects.
        max_chars: Maximum total characters across all context blocks.

    Returns:
        Formatted string with source-attributed context blocks.
    """
    if not results:
        return "No relevant context found."

    blocks: list[str] = []
    total_chars = 0

    for i, result in enumerate(results, 1):
        block = f"[{i}] {result.to_context_block()}"
        if total_chars + len(block) > max_chars:
            logger.debug("Context truncated at result %d (max_chars=%d)", i, max_chars)
            break
        blocks.append(block)
        total_chars += len(block)

    return "\n\n---\n\n".join(blocks)


def collection_stats() -> dict:
    """Return basic statistics about the vector store."""
    collection = get_collection()
    count = collection.count()
    return {
        "collection_name": Config.CHROMA_COLLECTION,
        "total_chunks": count,
        "db_path": Config.CHROMA_DB_PATH,
    }
