"""
modal_pipeline.py
-----------------
Optional: Scale the ingestion pipeline on Modal.com

Modal lets you run the CPU-intensive crawling and embedding steps on
remote machines — useful when ingesting hundreds of URLs or running
on a schedule.

This module wraps the existing pipeline stages as Modal functions.

Requirements:
  pip install modal
  modal token new   # authenticate once

Usage:
  # Deploy to Modal
  modal deploy modal_pipeline.py

  # Run a one-off ingest remotely
  modal run modal_pipeline.py::ingest_remote --query "your topic here"

Architecture note:
  - crawl_batch() runs in parallel across seed URLs (one container per URL)
  - embed_and_store_remote() runs on a single container (Chroma is local)
  - Results are passed between functions as Python objects (Modal serialises them)

Important: ChromaDB is a local store. In a distributed setup you'd replace
it with a remote vector DB (Qdrant Cloud, Pinecone). This demo shows the
parallelism pattern with Chroma by writing to a Modal Volume.
"""

from __future__ import annotations

import os
import sys

# Guard: only import modal if available
try:
    import modal
except ImportError:
    print("modal is not installed. Run: pip install modal")
    sys.exit(1)

# ── Modal app definition ──────────────────────────────────────────────────────

app = modal.App("rag-pipeline")

# Shared volume: persists ChromaDB across runs
volume = modal.Volume.from_name("rag-chroma-db", create_if_missing=True)
VOLUME_PATH = "/data/chroma_db"

# Container image with all dependencies
image = (
    modal.Image.debian_slim(python_version="3.11")
    .pip_install_from_requirements("requirements.txt")
)

# ── Remote functions ──────────────────────────────────────────────────────────

@app.function(
    image=image,
    cpu=2,                          # 2 CPUs per crawl worker
    timeout=120,                    # 2 min per URL batch
    retries=2,
    secrets=[
        modal.Secret.from_name("rag-api-keys"),   # store your API keys in Modal Secrets
    ],
)
def crawl_single_url(url: str, depth: int = 2) -> list[dict]:
    """
    Crawl a single seed URL and return raw page dicts.
    Runs in parallel: Modal spawns one container per URL.
    """
    # Import here (inside Modal container)
    from crawler import crawl
    from extractor import extract_all

    pages = crawl(seed_urls=[url], max_depth=depth, max_pages=10)
    docs = extract_all(pages)
    # Return serialisable dicts (RawPage objects aren't directly serialisable)
    return docs


@app.function(
    image=image,
    cpu=8,                          # Use all cores for embedding
    memory=4096,                    # 4 GB RAM for model + batch
    timeout=600,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("rag-api-keys")],
)
def embed_and_store_remote(documents: list[dict]) -> int:
    """
    Chunk, embed, and store documents in the shared ChromaDB volume.
    Runs on a single high-CPU container to avoid DB write conflicts.
    """
    import os
    os.environ["CHROMA_DB_PATH"] = VOLUME_PATH   # point Chroma at the volume

    from extractor import chunk_all
    from embedder import embed_and_store

    chunks = chunk_all(documents)
    stored = embed_and_store(chunks)
    volume.commit()     # persist volume changes
    return stored


@app.function(
    image=image,
    cpu=4,
    timeout=60,
    volumes={VOLUME_PATH: volume},
    secrets=[modal.Secret.from_name("rag-api-keys")],
)
def query_remote(query: str, top_k: int = 6) -> list[dict]:
    """Run a retrieval query against the remote ChromaDB."""
    import os
    os.environ["CHROMA_DB_PATH"] = VOLUME_PATH

    volume.reload()     # get latest data from volume
    from retriever import retrieve
    results = retrieve(query=query, top_k=top_k)
    return [
        {
            "score": r.score,
            "url": r.url,
            "title": r.title,
            "text": r.text[:500],   # truncate for display
        }
        for r in results
    ]


# ── Orchestrator (local entrypoint) ───────────────────────────────────────────

@app.local_entrypoint()
def ingest_remote(query: str = "Python async programming best practices"):
    """
    Orchestrate the full pipeline remotely.
    Run with: modal run modal_pipeline.py::ingest_remote --query "your topic"
    """
    from crawler import search_web
    print(f"Searching for: {query!r}")
    results = search_web(query, max_results=10)
    urls = [r.url for r in results if r.url]
    print(f"Found {len(urls)} URLs. Crawling in parallel…")

    # Crawl all URLs in parallel — Modal handles the parallelism
    all_documents: list[dict] = []
    for doc_batch in crawl_single_url.map(urls, kwargs={"depth": 2}):
        all_documents.extend(doc_batch)

    print(f"Extracted {len(all_documents)} documents. Embedding and storing…")
    stored = embed_and_store_remote.remote(all_documents)
    print(f"Done. Stored {stored} chunks.")
