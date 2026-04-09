"""
crawler.py
----------
Stage 1b — Crawling & Scraping
Given seed URLs from the search stage, this module:
  - Fetches each page (requests for static HTML)
  - Optionally delegates to Firecrawl for JS-heavy or anti-bot sites
  - Follows internal links up to a configurable depth limit
  - Deduplicates URLs before fetching
  - Respects rate limits with per-domain delays

Architecture choice: We use requests + BeautifulSoup rather than Scrapy
because our workload is moderate (≤30 pages per query) and we want to
avoid the overhead of a full Scrapy project. Firecrawl handles the cases
where plain requests fails (dynamic pages, bot detection).
"""

from __future__ import annotations

import hashlib
import logging
import time
from dataclasses import dataclass, field
from typing import Optional
from urllib.parse import urljoin, urlparse, urlunparse

import requests
from bs4 import BeautifulSoup
from tenacity import retry, stop_after_attempt, wait_exponential

from config import Config

logger = logging.getLogger(__name__)


@dataclass
class RawPage:
    """Raw fetched page before any content extraction."""
    url: str
    html: str
    status_code: int
    fetch_method: str           # "requests" | "firecrawl"
    depth: int = 0
    error: Optional[str] = None


# ── URL utilities ─────────────────────────────────────────────────────────────

def _normalise_url(url: str) -> str:
    """
    Strip fragments and common tracking params so the same logical page
    doesn't get crawled twice under different URLs.

    Example: https://example.com/page?utm_source=google#section
          → https://example.com/page
    """
    parsed = urlparse(url)
    # Drop fragment (#section) — always cosmetic
    clean = parsed._replace(fragment="")
    # Drop query params that don't affect content
    JUNK_PARAMS = {"utm_source", "utm_medium", "utm_campaign", "utm_term",
                   "utm_content", "ref", "source", "fbclid", "gclid"}
    if clean.query:
        kept = "&".join(
            p for p in clean.query.split("&")
            if p.split("=")[0] not in JUNK_PARAMS
        )
        clean = clean._replace(query=kept)
    return urlunparse(clean).rstrip("/").lower()


def _is_same_domain(url: str, base_domain: str) -> bool:
    """Return True if url belongs to base_domain (including subdomains)."""
    host = urlparse(url).netloc.lower()
    return host == base_domain or host.endswith("." + base_domain)


def _is_crawlable(url: str) -> bool:
    """Reject URLs that are almost certainly not useful HTML pages."""
    SKIP_EXTENSIONS = {
        ".pdf", ".zip", ".tar", ".gz", ".png", ".jpg", ".jpeg",
        ".gif", ".svg", ".mp4", ".mp3", ".exe", ".dmg", ".csv",
        ".xml", ".json", ".rss",
    }
    SKIP_PATTERNS = {
        "login", "logout", "signup", "register", "cart", "checkout",
        "account", "privacy", "terms", "cookie", "sitemap",
    }
    path = urlparse(url).path.lower()
    if any(path.endswith(ext) for ext in SKIP_EXTENSIONS):
        return False
    if any(pat in path for pat in SKIP_PATTERNS):
        return False
    return True


# ── Fetching ─────────────────────────────────────────────────────────────────

def _make_session() -> requests.Session:
    """Create a requests Session with sensible defaults."""
    session = requests.Session()
    session.headers.update({
        "User-Agent": Config.USER_AGENT,
        "Accept": "text/html,application/xhtml+xml",
        "Accept-Language": "en-US,en;q=0.9",
    })
    return session


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=8))
def _fetch_with_requests(url: str, session: requests.Session, depth: int) -> RawPage:
    """Fetch a single URL using plain HTTP. Fast and low-overhead."""
    try:
        resp = session.get(url, timeout=Config.REQUEST_TIMEOUT, allow_redirects=True)
        # Only accept HTML responses
        content_type = resp.headers.get("content-type", "")
        if "text/html" not in content_type:
            return RawPage(url=url, html="", status_code=resp.status_code,
                           fetch_method="requests", depth=depth,
                           error=f"Non-HTML content-type: {content_type}")
        return RawPage(
            url=resp.url,           # use final URL after redirects
            html=resp.text,
            status_code=resp.status_code,
            fetch_method="requests",
            depth=depth,
        )
    except requests.RequestException as exc:
        logger.warning("requests fetch failed for %s: %s", url, exc)
        return RawPage(url=url, html="", status_code=0,
                       fetch_method="requests", depth=depth, error=str(exc))


def _fetch_with_firecrawl(url: str, depth: int) -> RawPage:
    """
    Delegate to Firecrawl for JS-heavy or bot-protected pages.
    Firecrawl returns pre-rendered HTML (or clean Markdown).
    We request HTML so our extraction stage handles it uniformly.
    """
    if not Config.FIRECRAWL_API_KEY:
        raise EnvironmentError("FIRECRAWL_API_KEY not set")

    headers = {
        "Authorization": f"Bearer {Config.FIRECRAWL_API_KEY}",
        "Content-Type": "application/json",
    }
    payload = {
        "url": url,
        "formats": ["html"],    # request rendered HTML
    }

    resp = requests.post(
        "https://api.firecrawl.dev/v1/scrape",
        json=payload,
        headers=headers,
        timeout=30,             # Firecrawl needs more time for JS rendering
    )
    resp.raise_for_status()
    data = resp.json()

    html = data.get("data", {}).get("html", "")
    return RawPage(
        url=url,
        html=html,
        status_code=200,
        fetch_method="firecrawl",
        depth=depth,
    )


def _fetch_page(url: str, session: requests.Session, depth: int,
                use_firecrawl: bool = False) -> RawPage:
    """
    Unified fetch dispatcher.
    If use_firecrawl is True (or requests returns empty/error), try Firecrawl.
    """
    if use_firecrawl and Config.FIRECRAWL_API_KEY:
        try:
            return _fetch_with_firecrawl(url, depth)
        except Exception as exc:
            logger.warning("Firecrawl failed for %s: %s — falling back", url, exc)

    page = _fetch_with_requests(url, session, depth)

    # Heuristic: if we got very little HTML, the page is probably dynamic.
    # Try Firecrawl as a fallback if available.
    if (not page.error and len(page.html) < 2000
            and Config.FIRECRAWL_API_KEY):
        logger.info("Short response for %s, retrying with Firecrawl", url)
        fc_page = _fetch_with_firecrawl(url, depth)
        if len(fc_page.html) > len(page.html):
            return fc_page

    return page


# ── Internal link extraction ─────────────────────────────────────────────────

def _extract_internal_links(html: str, base_url: str, base_domain: str) -> list[str]:
    """
    Parse all <a href> links from a page, resolve them relative to base_url,
    and return only those that belong to the same domain.
    """
    soup = BeautifulSoup(html, "lxml")
    links = []
    for tag in soup.find_all("a", href=True):
        href_value = tag.get("href")
        if isinstance(href_value, list):
            href = href_value[0].strip() if href_value else ""
        else:
            href = str(href_value).strip()
        if not href or href.startswith(("mailto:", "tel:", "javascript:")):
            continue
        absolute = urljoin(base_url, href)
        if _is_same_domain(absolute, base_domain) and _is_crawlable(absolute):
            links.append(_normalise_url(absolute))
    return list(set(links))     # deduplicate within the page


# ── Main crawl function ───────────────────────────────────────────────────────

def crawl(
    seed_urls: list[str],
    max_depth: int = Config.CRAWL_MAX_DEPTH,
    max_pages: int = Config.CRAWL_MAX_PAGES,
    use_firecrawl: bool = False,
) -> list[RawPage]:
    """
    BFS crawl starting from seed_urls.

    Args:
        seed_urls:     Entry-point URLs (from search stage or user).
        max_depth:     How many link-hops to follow from each seed.
        max_pages:     Hard cap on total pages fetched across all seeds.
        use_firecrawl: Force Firecrawl for all pages (for JS-heavy sites).

    Returns:
        List of RawPage objects (including failed ones with error set,
        so callers can log or skip them).
    """
    visited: set[str] = set()       # normalised URLs already fetched/queued
    results: list[RawPage] = []
    # Queue entries: (url, depth)
    queue: list[tuple[str, int]] = [
        (_normalise_url(u), 0) for u in seed_urls
    ]
    session = _make_session()

    while queue and len(results) < max_pages:
        url, depth = queue.pop(0)

        if url in visited:
            continue
        visited.add(url)

        logger.info("[depth=%d] Fetching: %s", depth, url)
        page = _fetch_page(url, session, depth, use_firecrawl=use_firecrawl)
        results.append(page)

        # Rate limit: be polite to servers
        time.sleep(Config.CRAWL_DELAY_SECONDS)

        # Don't follow links beyond max depth or from failed pages
        if depth >= max_depth or page.error or not page.html:
            continue

        # Discover internal links and add unseen ones to the queue
        base_domain = urlparse(url).netloc.lower()
        new_links = _extract_internal_links(page.html, url, base_domain)
        for link in new_links:
            if link not in visited:
                queue.append((link, depth + 1))

    logger.info(
        "Crawl complete: %d pages fetched (%d unique URLs seen)",
        len(results), len(visited)
    )
    return results


# ── Stage 1a — Web Search (merged from search.py) ───────────────────────────


@dataclass
class SearchResult:
    """Structured container for a single search result."""
    url: str
    title: str
    snippet: str
    # Tavily optionally returns pre-fetched content — use it if present
    # to skip a redundant HTTP request for that page.
    raw_content: Optional[str] = field(default=None, repr=False)


def _search_tavily(query: str, max_results: int) -> list[SearchResult]:
    """Call Tavily's /search endpoint."""
    if not Config.TAVILY_API_KEY:
        raise EnvironmentError("TAVILY_API_KEY not set")

    payload = {
        "api_key": Config.TAVILY_API_KEY,
        "query": query,
        "max_results": max_results,
        "include_raw_content": True,
        "search_depth": "advanced",
    }

    resp = requests.post(
        "https://api.tavily.com/search",
        json=payload,
        timeout=Config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[SearchResult] = []
    for item in data.get("results", []):
        results.append(SearchResult(
            url=item.get("url", ""),
            title=item.get("title", ""),
            snippet=item.get("content", ""),
            raw_content=item.get("raw_content"),
        ))
    return results


def _search_langsearch(query: str, max_results: int) -> list[SearchResult]:
    """LangSearch API — returns URLs + snippets without raw page content."""
    if not Config.LANGSEARCH_API_KEY:
        raise EnvironmentError("LANGSEARCH_API_KEY not set")

    headers = {"Authorization": f"Bearer {Config.LANGSEARCH_API_KEY}"}
    params = {"q": query, "num": max_results}

    resp = requests.get(
        "https://api.langsearch.com/v1/search",
        headers=headers,
        params=params,
        timeout=Config.REQUEST_TIMEOUT,
    )
    resp.raise_for_status()
    data = resp.json()

    results: list[SearchResult] = []
    for item in data.get("data", {}).get("webPages", {}).get("value", []):
        results.append(SearchResult(
            url=item.get("url", ""),
            title=item.get("name", ""),
            snippet=item.get("snippet", ""),
        ))
    return results


@retry(stop=stop_after_attempt(3), wait=wait_exponential(min=2, max=10))
def search_web(query: str, max_results: int = 10) -> list[SearchResult]:
    """Entry point for web search (Tavily → LangSearch fallback)."""
    logger.info("Searching web for: %r (max %d results)", query, max_results)

    if Config.TAVILY_API_KEY:
        try:
            results = _search_tavily(query, max_results)
            logger.info("Tavily returned %d results", len(results))
            return results
        except Exception as exc:  # noqa: BLE001
            logger.warning("Tavily failed (%s), trying LangSearch…", exc)

    if Config.LANGSEARCH_API_KEY:
        results = _search_langsearch(query, max_results)
        logger.info("LangSearch returned %d results", len(results))
        return results

    raise EnvironmentError(
        "No search API key configured. "
        "Set TAVILY_API_KEY or LANGSEARCH_API_KEY in your .env file."
    )
