"""
Standalone web retrieval pipeline for LLM-ready context.

What this script does:
1) Discover recent links with Tavily, LangSearch, and Jina Search.
2) Deduplicate and rank candidate URLs.
3) Enrich top URLs with Firecrawl (or Jina Reader fallback).
4) Produce LLM-ready output (compact facts + citations).

Usage examples:
    uv run python web_search.py --query "latest AI regulation updates"
    uv run python web_search.py --query "latest AI regulation updates" --json-out web_result.json

Environment variables:
    TAVILY_API_KEY
    LANGSEARCH_API_KEY
    FIRECRAWL_API_KEY
    FIRECRAWL_BASE_URL (default: https://api.firecrawl.dev)
    JINA_API_KEY (optional)
"""

from __future__ import annotations

import argparse
import json
import os
import re
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any
from urllib.parse import quote

import httpx


def load_local_env() -> None:
    env_path = Path(__file__).resolve().parent / ".env"
    if not env_path.exists():
        return
    for raw_line in env_path.read_text(encoding="utf-8", errors="replace").splitlines():
        line = raw_line.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, value = line.split("=", 1)
        key = key.strip()
        value = value.strip().strip('"').strip("'")
        if key:
            os.environ.setdefault(key, value)


@dataclass
class Config:
    tavily_api_key: str
    langsearch_api_key: str
    firecrawl_api_key: str
    firecrawl_base_url: str
    jina_api_key: str
    timeout_sec: float = 20.0
    discovery_per_provider: int = 6
    max_urls_to_enrich: int = 5
    max_excerpt_chars: int = 1200


@dataclass
class SearchHit:
    provider: str
    title: str
    url: str
    snippet: str
    score: float


@dataclass
class EnrichedDoc:
    provider: str
    title: str
    url: str
    content: str


URL_RE = re.compile(r"https?://[^\s)]+")


def ensure_http_url(url: str) -> str:
    value = (url or "").strip()
    if not value:
        return ""
    return value if value.startswith(("http://", "https://")) else f"https://{value}"


def extract_urls(text: str, limit: int = 20) -> list[str]:
    found: list[str] = []
    for match in URL_RE.findall(text or ""):
        clean = match.rstrip(".,;)")
        if clean.startswith(("https://s.jina.ai", "https://r.jina.ai")):
            continue
        if clean not in found:
            found.append(clean)
        if len(found) >= limit:
            break
    return found


def query_bonus(text: str) -> float:
    lowered = (text or "").lower()
    bonus = 0.0
    for marker in ("today", "latest", "breaking", "new", "2026", "2025"):
        if marker in lowered:
            bonus += 0.1
    return min(0.5, bonus)


def provider_weight(provider: str) -> float:
    return {
        "tavily": 1.0,
        "langsearch": 0.95,
        "jina": 0.8,
    }.get(provider, 0.7)


def tavily_search(client: httpx.Client, cfg: Config, query: str) -> list[SearchHit]:
    if not cfg.tavily_api_key:
        return []
    try:
        response = client.post(
            "https://api.tavily.com/search",
            json={
                "api_key": cfg.tavily_api_key,
                "query": query,
                "search_depth": "basic",
                "max_results": cfg.discovery_per_provider,
                "topic": "news",
            },
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    hits: list[SearchHit] = []
    for item in payload.get("results", [])[: cfg.discovery_per_provider]:
        title = str(item.get("title") or "(no title)")
        url = ensure_http_url(str(item.get("url") or ""))
        snippet = str(item.get("content") or "")
        if not url:
            continue
        score = provider_weight("tavily") + query_bonus(f"{title} {snippet}")
        hits.append(SearchHit("tavily", title, url, snippet, score))
    return hits


def langsearch_search(client: httpx.Client, cfg: Config, query: str) -> list[SearchHit]:
    if not cfg.langsearch_api_key:
        return []
    try:
        response = client.post(
            "https://api.langsearch.com/v1/search",
            headers={
                "Authorization": f"Bearer {cfg.langsearch_api_key}",
                "Content-Type": "application/json",
            },
            json={"query": query, "top_k": cfg.discovery_per_provider},
        )
        response.raise_for_status()
        payload = response.json()
    except Exception:
        return []

    hits: list[SearchHit] = []
    for item in (payload.get("results") or [])[: cfg.discovery_per_provider]:
        if not isinstance(item, dict):
            continue
        title = str(item.get("title") or item.get("name") or "(no title)")
        url = ensure_http_url(str(item.get("url") or item.get("link") or ""))
        snippet = str(item.get("summary") or item.get("content") or "")
        if not url:
            continue
        score = provider_weight("langsearch") + query_bonus(f"{title} {snippet}")
        hits.append(SearchHit("langsearch", title, url, snippet, score))
    return hits


def jina_search(client: httpx.Client, cfg: Config, query: str) -> list[SearchHit]:
    headers: dict[str, str] = {}
    if cfg.jina_api_key:
        headers["Authorization"] = f"Bearer {cfg.jina_api_key}"
    try:
        response = client.get(f"https://s.jina.ai/{quote(query)}", headers=headers)
        response.raise_for_status()
        body = response.text.strip()
    except Exception:
        return []

    hits: list[SearchHit] = []
    discovered_urls = extract_urls(body, limit=cfg.discovery_per_provider)
    for idx, url in enumerate(discovered_urls, start=1):
        hits.append(
            SearchHit(
                provider="jina",
                title=f"Jina discovered source #{idx}",
                url=url,
                snippet=body[:450],
                score=provider_weight("jina") + query_bonus(body[:600]),
            )
        )
    return hits


def dedupe_and_rank(hits: list[SearchHit]) -> list[SearchHit]:
    best_by_url: dict[str, SearchHit] = {}
    for hit in hits:
        url = hit.url.strip()
        if not url:
            continue
        existing = best_by_url.get(url)
        if existing is None or hit.score > existing.score:
            best_by_url[url] = hit
    ranked = sorted(best_by_url.values(), key=lambda h: h.score, reverse=True)
    return ranked


def firecrawl_scrape(client: httpx.Client, cfg: Config, url: str) -> str:
    if not cfg.firecrawl_api_key:
        return ""
    try:
        response = client.post(
            f"{cfg.firecrawl_base_url}/v1/scrape",
            headers={
                "Authorization": f"Bearer {cfg.firecrawl_api_key}",
                "Content-Type": "application/json",
            },
            json={
                "url": url,
                "formats": ["markdown"],
                "onlyMainContent": True,
            },
        )
        response.raise_for_status()
        payload = response.json()
        data = payload.get("data", {}) if isinstance(payload, dict) else {}
        markdown = data.get("markdown") if isinstance(data, dict) else ""
        return str(markdown or "")
    except Exception:
        return ""


def jina_reader_fetch(client: httpx.Client, cfg: Config, url: str) -> str:
    headers: dict[str, str] = {}
    if cfg.jina_api_key:
        headers["Authorization"] = f"Bearer {cfg.jina_api_key}"
    try:
        response = client.get(f"https://r.jina.ai/{url}", headers=headers)
        response.raise_for_status()
        return response.text.strip()
    except Exception:
        return ""


def enrich_hits(client: httpx.Client, cfg: Config, hits: list[SearchHit]) -> list[EnrichedDoc]:
    enriched: list[EnrichedDoc] = []
    for hit in hits[: cfg.max_urls_to_enrich]:
        content = firecrawl_scrape(client, cfg, hit.url)
        if not content:
            # Firecrawl unavailable/failed: fallback to Jina Reader so pipeline still returns useful text.
            content = jina_reader_fetch(client, cfg, hit.url)
        if not content:
            continue
        enriched.append(
            EnrichedDoc(
                provider=hit.provider,
                title=hit.title,
                url=hit.url,
                content=content[: cfg.max_excerpt_chars],
            )
        )
    return enriched


def build_llm_ready_payload(query: str, ranked_hits: list[SearchHit], docs: list[EnrichedDoc]) -> dict[str, Any]:
    sources = []
    for hit in ranked_hits:
        sources.append(
            {
                "provider": hit.provider,
                "title": hit.title,
                "url": hit.url,
                "snippet": hit.snippet[:300],
                "score": round(hit.score, 3),
            }
        )

    evidence_blocks: list[str] = []
    for idx, doc in enumerate(docs, start=1):
        evidence_blocks.append(
            "\n".join(
                [
                    f"[source#{idx}] provider={doc.provider}",
                    f"title: {doc.title}",
                    f"url: {doc.url}",
                    "excerpt:",
                    doc.content.strip(),
                ]
            )
        )

    llm_ready_context = (
        "Use the following recent web evidence to answer the user query. "
        "Prioritize factual statements that appear in multiple sources and cite source numbers.\n\n"
        f"User query: {query}\n\n"
        + "\n\n".join(evidence_blocks)
    )

    return {
        "query": query,
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "discovered_sources": sources,
        "enriched_count": len(docs),
        "llm_ready_context": llm_ready_context,
    }


def run_pipeline(query: str, cfg: Config) -> dict[str, Any]:
    with httpx.Client(timeout=cfg.timeout_sec, follow_redirects=True) as client:
        discovered: list[SearchHit] = []
        discovered.extend(tavily_search(client, cfg, query))
        discovered.extend(langsearch_search(client, cfg, query))
        discovered.extend(jina_search(client, cfg, query))

        ranked = dedupe_and_rank(discovered)
        enriched = enrich_hits(client, cfg, ranked)
        return build_llm_ready_payload(query, ranked, enriched)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Multi-provider web retrieval to LLM-ready context")
    parser.add_argument("--query", required=True, help="Search query, preferably recent/current-events oriented")
    parser.add_argument("--json-out", default="", help="Optional file path to save JSON payload")
    parser.add_argument("--max-urls", type=int, default=5, help="Max top URLs to enrich")
    return parser.parse_args()


def main() -> None:
    load_local_env()
    args = parse_args()

    cfg = Config(
        tavily_api_key=os.getenv("TAVILY_API_KEY", ""),
        langsearch_api_key=os.getenv("LANGSEARCH_API_KEY", ""),
        firecrawl_api_key=os.getenv("FIRECRAWL_API_KEY", ""),
        firecrawl_base_url=os.getenv("FIRECRAWL_BASE_URL", "https://api.firecrawl.dev").rstrip("/"),
        jina_api_key=os.getenv("JINA_API_KEY", ""),
        max_urls_to_enrich=max(1, args.max_urls),
    )

    payload = run_pipeline(args.query, cfg)

    print("=== Web Retrieval Summary ===")
    print(f"Query: {payload['query']}")
    print(f"Discovered sources: {len(payload['discovered_sources'])}")
    print(f"Enriched sources: {payload['enriched_count']}")
    print("\n=== LLM Ready Context ===")
    print(payload["llm_ready_context"])

    if args.json_out:
        out_path = Path(args.json_out)
        out_path.write_text(json.dumps(payload, indent=2, ensure_ascii=True), encoding="utf-8")
        print(f"\nSaved JSON payload to: {out_path}")


if __name__ == "__main__":
    main()
