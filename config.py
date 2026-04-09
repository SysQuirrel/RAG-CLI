"""
config.py
---------
Central configuration loaded from environment variables.
All pipeline modules import from here — no magic strings scattered around.
"""

import os
from dotenv import load_dotenv

load_dotenv()


class Config:
    # ── API Keys ────────────────────────────────────────────────────────────
    TAVILY_API_KEY: str = os.getenv("TAVILY_API_KEY", "")
    FIRECRAWL_API_KEY: str = os.getenv("FIRECRAWL_API_KEY", "")
    LANGSEARCH_API_KEY: str = os.getenv("LANGSEARCH_API_KEY", "")

    # ── Crawling ─────────────────────────────────────────────────────────────
    CRAWL_MAX_DEPTH: int = int(os.getenv("CRAWL_MAX_DEPTH", 2))
    CRAWL_MAX_PAGES: int = int(os.getenv("CRAWL_MAX_PAGES", 30))
    CRAWL_DELAY_SECONDS: float = float(os.getenv("CRAWL_DELAY_SECONDS", 1.5))
    REQUEST_TIMEOUT: int = int(os.getenv("REQUEST_TIMEOUT", 15))
    USER_AGENT: str = (
        "Mozilla/5.0 (compatible; RAGBot/1.0; +https://example.com/bot)"
    )

    # ── Content extraction ───────────────────────────────────────────────────
    MIN_CONTENT_WORDS: int = 80          # discard pages shorter than this
    NOISE_TAGS: tuple = (               # HTML tags stripped before extraction
        "nav", "header", "footer", "aside", "script",
        "style", "form", "noscript", "iframe",
    )

    # ── Chunking ─────────────────────────────────────────────────────────────
    CHUNK_SIZE: int = int(os.getenv("CHUNK_SIZE", 500))
    CHUNK_OVERLAP: int = int(os.getenv("CHUNK_OVERLAP", 75))

    # ── Embedding ────────────────────────────────────────────────────────────
    # Default to bge-base-en-v1.5 for stronger semantic retrieval quality.
    EMBEDDING_MODEL: str = os.getenv("EMBEDDING_MODEL", "BAAI/bge-base-en-v1.5")
    EMBEDDING_BATCH_SIZE: int = 128     # chunks embedded per batch (larger batches = fewer forward passes)

    # ── Vector store ─────────────────────────────────────────────────────────
    CHROMA_DB_PATH: str = os.path.expanduser(os.getenv("CHROMA_DB_PATH", "~/.rag-cli/chroma_new"))
    CHROMA_COLLECTION: str = "rag_documents"

    # ── Retrieval ────────────────────────────────────────────────────────────
    RETRIEVAL_TOP_K: int = 6            # candidates returned from vector search
    RETRIEVAL_MIN_SCORE: float = 0.30   # cosine similarity floor (0–1)
