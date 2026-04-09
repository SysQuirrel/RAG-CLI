"""
pipeline.py
-----------
Pipeline Orchestrator

Wires together all stages:
  search → crawl → extract → chunk → embed → store

Can be run as a script or imported as a module.

Usage:
  # Ingest content for a query
  python pipeline.py ingest "best practices for API rate limiting"

  # Query the stored knowledge base
  python pipeline.py query "how do I handle rate limits in Python?"

  # Show collection stats
  python pipeline.py stats
"""

from __future__ import annotations

import argparse
import json
import logging
import sys
from pathlib import Path

from config import Config
from crawler import crawl, RawPage, search_web, SearchResult
from extractor import extract_all, chunk_all
from embedder import embed_and_store
from retriever import retrieve, format_context, collection_stats

# ── Logging setup ─────────────────────────────────────────────────────────────

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    datefmt="%H:%M:%S",
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler("pipeline.log", encoding="utf-8"),
    ],
)
logger = logging.getLogger(__name__)


# ── Validation ────────────────────────────────────────────────────────────────

def _validate_config() -> None:
    """Fail fast if required config is missing."""
    if not Config.TAVILY_API_KEY and not Config.LANGSEARCH_API_KEY:
        logger.error(
            "No search API key found. Set TAVILY_API_KEY or "
            "LANGSEARCH_API_KEY in your .env file."
        )
        sys.exit(1)


# ── Ingest pipeline ───────────────────────────────────────────────────────────

def run_ingest(
    query: str,
    max_search_results: int = 10,
    max_pages: int = Config.CRAWL_MAX_PAGES,
    max_depth: int = Config.CRAWL_MAX_DEPTH,
    use_firecrawl: bool = False,
    save_documents: bool = True,
) -> dict:
    """
    Full ingestion pipeline: search → crawl → extract → chunk → embed → store.

    Args:
        query:              Topic to search for and ingest.
        max_search_results: Number of search results to use as seed URLs.
        max_pages:          Total pages to crawl across all seeds.
        max_depth:          Link-follow depth from each seed URL.
        use_firecrawl:      Force Firecrawl for JS-heavy sites.
        save_documents:     Persist structured docs to JSONL for inspection.

    Returns:
        Summary dict with counts at each stage.
    """
    _validate_config()
    logger.info("=" * 60)
    logger.info("INGESTION PIPELINE START")
    logger.info("Query: %r", query)
    logger.info("=" * 60)

    summary = {
        "query": query,
        "search_results": 0,
        "pages_crawled": 0,
        "documents_extracted": 0,
        "chunks_created": 0,
        "chunks_stored": 0,
    }

    # ── Stage 1a: Search ──────────────────────────────────────────────────
    logger.info("STAGE 1a — Web Search")
    search_results: list[SearchResult] = search_web(query, max_results=max_search_results)
    summary["search_results"] = len(search_results)

    if not search_results:
        logger.error("No search results returned. Aborting.")
        return summary

    # Extract seed URLs; some results (Tavily) may already have content
    seed_urls = [r.url for r in search_results if r.url]
    logger.info("Got %d seed URLs", len(seed_urls))

    # ── Stage 1b: Crawl ───────────────────────────────────────────────────
    logger.info("STAGE 1b — Crawling")
    pages: list[RawPage] = crawl(
        seed_urls=seed_urls,
        max_depth=max_depth,
        max_pages=max_pages,
        use_firecrawl=use_firecrawl,
    )
    summary["pages_crawled"] = len(pages)

    # For Tavily results that returned raw_content, inject as synthetic pages
    # This avoids re-fetching pages Tavily already downloaded.
    from crawler import RawPage
    for result in search_results:
        if result.raw_content and result.url:
            pages.append(RawPage(
                url=result.url,
                html=f"<html><body>{result.raw_content}</body></html>",
                status_code=200,
                fetch_method="tavily_prefetch",
                depth=0,
            ))

    # ── Stage 2 & 3: Extract & Structure ─────────────────────────────────
    logger.info("STAGE 2+3 — Content Extraction & Structuring")
    documents = extract_all(pages)
    summary["documents_extracted"] = len(documents)

    if not documents:
        logger.error("No content extracted. Check crawl logs.")
        return summary

    # Optionally persist documents for offline inspection / debugging
    if save_documents:
        out_path = Path("documents.jsonl")
        with out_path.open("w", encoding="utf-8") as f:
            for doc in documents:
                # Exclude embeddings from JSONL (too large)
                record = {k: v for k, v in doc.items() if k != "embedding"}
                f.write(json.dumps(record, ensure_ascii=False) + "\n")
        logger.info("Documents saved to %s", out_path)

    # ── Stage 4: Chunking ─────────────────────────────────────────────────
    logger.info("STAGE 4 — Chunking")
    chunks = chunk_all(documents)
    summary["chunks_created"] = len(chunks)

    if not chunks:
        logger.error("No chunks produced. Check document content.")
        return summary

    # ── Stage 5 & 6: Embed & Store ────────────────────────────────────────
    logger.info("STAGE 5+6 — Embedding & Storing")
    stored = embed_and_store(chunks)
    summary["chunks_stored"] = stored

    logger.info("=" * 60)
    logger.info("INGESTION COMPLETE")
    for key, val in summary.items():
        logger.info("  %-25s %s", key + ":", val)
    logger.info("=" * 60)

    return summary


# ── Query interface ───────────────────────────────────────────────────────────

def run_query(
    query: str,
    top_k: int = Config.RETRIEVAL_TOP_K,
    min_score: float = Config.RETRIEVAL_MIN_SCORE,
    domain_filter: str | None = None,
    print_context: bool = True,
) -> list[dict]:
    """
    Query the vector store and return relevant chunks.

    Args:
        query:         Natural language question.
        top_k:         Number of results to return.
        min_score:     Minimum similarity score (0–1).
        domain_filter: Restrict results to a specific domain.
        print_context: Print formatted context to stdout.

    Returns:
        List of result dicts with text, url, score, etc.
    """
    results = retrieve(
        query=query,
        top_k=top_k,
        min_score=min_score,
        domain_filter=domain_filter,
    )

    if not results:
        logger.info("No results found above threshold.")
        return []

    if print_context:
        context = format_context(results)
        print("\n" + "=" * 60)
        print(f"RETRIEVAL RESULTS FOR: {query!r}")
        print("=" * 60)
        print(context)
        print("=" * 60 + "\n")

    return [
        {
            "chunk_id": r.chunk_id,
            "score": round(r.score, 4),
            "url": r.url,
            "title": r.title,
            "nearest_heading": r.nearest_heading,
            "text": r.text,
        }
        for r in results
    ]


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    parser = argparse.ArgumentParser(
        description="RAG Pipeline — Ingest web content and query it"
    )
    subparsers = parser.add_subparsers(dest="command", required=True)

    # ingest subcommand
    ingest_parser = subparsers.add_parser("ingest", help="Search, crawl, and store content")
    ingest_parser.add_argument("query", help="Topic to search and ingest")
    ingest_parser.add_argument("--max-results", type=int, default=10,
                               help="Number of search results to crawl (default: 10)")
    ingest_parser.add_argument("--max-pages", type=int, default=Config.CRAWL_MAX_PAGES,
                               help=f"Max pages to crawl (default: {Config.CRAWL_MAX_PAGES})")
    ingest_parser.add_argument("--max-depth", type=int, default=Config.CRAWL_MAX_DEPTH,
                               help=f"Crawl depth (default: {Config.CRAWL_MAX_DEPTH})")
    ingest_parser.add_argument("--firecrawl", action="store_true",
                               help="Use Firecrawl for all pages (for JS-heavy sites)")
    ingest_parser.add_argument("--no-save", action="store_true",
                               help="Don't save documents.jsonl")

    # query subcommand
    query_parser = subparsers.add_parser("query", help="Query the stored knowledge base")
    query_parser.add_argument("query", help="Question to answer")
    query_parser.add_argument("--top-k", type=int, default=Config.RETRIEVAL_TOP_K,
                              help=f"Number of results (default: {Config.RETRIEVAL_TOP_K})")
    query_parser.add_argument("--min-score", type=float, default=Config.RETRIEVAL_MIN_SCORE,
                              help=f"Minimum similarity score (default: {Config.RETRIEVAL_MIN_SCORE})")
    query_parser.add_argument("--domain", type=str, default=None,
                              help="Filter results to a specific domain")
    query_parser.add_argument("--json", action="store_true",
                              help="Output results as JSON")

    # stats subcommand
    subparsers.add_parser("stats", help="Show vector store statistics")

    args = parser.parse_args()

    if args.command == "ingest":
        summary = run_ingest(
            query=args.query,
            max_search_results=args.max_results,
            max_pages=args.max_pages,
            max_depth=args.max_depth,
            use_firecrawl=args.firecrawl,
            save_documents=not args.no_save,
        )
        print("\nSummary:", json.dumps(summary, indent=2))

    elif args.command == "query":
        results = run_query(
            query=args.query,
            top_k=args.top_k,
            min_score=args.min_score,
            domain_filter=args.domain,
            print_context=not args.json,
        )
        if args.json:
            print(json.dumps(results, indent=2))

    elif args.command == "stats":
        stats = collection_stats()
        print(json.dumps(stats, indent=2))


if __name__ == "__main__":
    main()
