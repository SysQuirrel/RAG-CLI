"""
extractor.py
------------
Stage 2 — Content Extraction & Cleaning
Stage 3 — Data Structuring

Takes RawPage objects (HTML) and produces structured Document dicts.

Extraction strategy (in order of priority):
  1. trafilatura  — best general-purpose boilerplate remover
  2. readability  — Mozilla's algorithm as fallback
  3. BS4 heuristic — last resort: grab largest <article>/<main> block

Output schema (per document):
  {
    "id":          sha256 of URL (stable identifier),
    "url":         canonical URL,
    "domain":      hostname,
    "title":       page title,
    "headings":    list of H1-H3 texts (for context),
    "content":     clean plain text (main body only),
    "word_count":  int,
    "metadata": {
      "crawled_at":    ISO timestamp,
      "depth":         crawl depth,
      "fetch_method":  "requests" | "firecrawl",
      "language":      detected language code,
    }
  }
"""

from __future__ import annotations

import hashlib
import logging
import re
import unicodedata
from datetime import datetime, timezone
from typing import Optional

import trafilatura
from bs4 import BeautifulSoup

from config import Config
from crawler import RawPage

logger = logging.getLogger(__name__)


# ── Extraction helpers ────────────────────────────────────────────────────────

def _extract_with_trafilatura(html: str) -> Optional[str]:
    """
    trafilatura is purpose-built for this: it scores text blocks by
    density and position to separate main content from boilerplate.
    favor_precision=True skips borderline blocks (worth the trade-off).
    """
    text = trafilatura.extract(
        html,
        favor_precision=True,
        include_comments=False,
        include_tables=True,        # tables often contain valuable data
        no_fallback=False,
    )
    return text


def _extract_with_bs4_heuristic(html: str) -> Optional[str]:
    """
    Last-resort extractor: look for semantic content containers,
    strip known noise tags, then return inner text.
    """
    soup = BeautifulSoup(html, "lxml")

    # Remove explicit boilerplate elements
    for tag_name in Config.NOISE_TAGS:
        for tag in soup.find_all(tag_name):
            tag.decompose()

    # Try semantic containers in priority order
    for selector in ("article", "main", '[role="main"]', ".content",
                     ".post-content", ".entry-content", "#content"):
        container = soup.select_one(selector)
        if container:
            text = container.get_text(separator="\n", strip=True)
            if len(text.split()) >= Config.MIN_CONTENT_WORDS:
                return text

    # Absolute fallback: whole body
    body = soup.find("body")
    return body.get_text(separator="\n", strip=True) if body else None


def _extract_title(html: str) -> str:
    """Extract page title from <title> or first <h1>."""
    soup = BeautifulSoup(html, "lxml")
    # Try <title>
    title_tag = soup.find("title")
    if title_tag and title_tag.text.strip():
        return title_tag.text.strip()
    # Fallback to first <h1>
    h1 = soup.find("h1")
    if h1:
        return h1.get_text(strip=True)
    return "Untitled"


def _extract_headings(html: str) -> list[str]:
    """
    Collect H1–H3 texts. Headings give the LLM structural context about
    what a chunk came from, even after the document is split.
    """
    soup = BeautifulSoup(html, "lxml")
    headings = []
    for tag in soup.find_all(["h1", "h2", "h3"]):
        text = tag.get_text(strip=True)
        if text:
            headings.append(f"{tag.name.upper()}: {text}")
    return headings


# ── Text cleaning ─────────────────────────────────────────────────────────────

def _clean_text(text: str) -> str:
    """
    Post-extraction text normalisation:
      1. NFC unicode normalisation (combines diacritics)
      2. Replace Windows line endings
      3. Collapse runs of blank lines to a single blank line
      4. Strip leading/trailing whitespace per line
      5. Remove zero-width characters and other invisible Unicode
    """
    # 1. Unicode normalisation
    text = unicodedata.normalize("NFC", text)
    # 2. Normalise line endings
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    # 3. Remove zero-width / invisible characters
    text = re.sub(r"[\u200b\u200c\u200d\ufeff\xa0]", " ", text)
    # 4. Clean up each line
    lines = [line.strip() for line in text.split("\n")]
    # 5. Collapse consecutive blank lines
    cleaned_lines: list[str] = []
    prev_blank = False
    for line in lines:
        is_blank = not line
        if is_blank and prev_blank:
            continue
        cleaned_lines.append(line)
        prev_blank = is_blank
    return "\n".join(cleaned_lines).strip()


def _detect_language(text: str) -> str:
    """
    Very lightweight language detection — checks for common
    non-English patterns. Returns ISO 639-1 code.
    Falls back to 'en' if undetermined (avoids adding langdetect dependency).
    """
    # Sample the first 500 chars — enough for detection
    sample = text[:500].lower()
    # Simple heuristic: high ratio of ASCII printable = likely English
    ascii_chars = sum(1 for c in sample if ord(c) < 128)
    ratio = ascii_chars / max(len(sample), 1)
    return "en" if ratio > 0.85 else "other"


# ── Stable document ID ────────────────────────────────────────────────────────

def _make_doc_id(url: str) -> str:
    """SHA-256 of the URL → 16-char hex prefix. Stable across runs."""
    return hashlib.sha256(url.encode()).hexdigest()[:16]


# ── Main extraction function ──────────────────────────────────────────────────

def extract_document(page: RawPage) -> Optional[dict]:
    """
    Convert a RawPage into a structured Document dict.

    Returns None if:
      - The page had a fetch error
      - Extracted content is too short to be useful
    """
    if page.error:
        logger.debug("Skipping errored page: %s — %s", page.url, page.error)
        return None

    if not page.html:
        logger.debug("Skipping empty page: %s", page.url)
        return None

    # ── Extract content ───────────────────────────────────────────────────
    # Try trafilatura first; fall back to BS4 heuristic
    content = _extract_with_trafilatura(page.html)
    if not content or len(content.split()) < Config.MIN_CONTENT_WORDS:
        logger.debug("trafilatura insufficient for %s, using BS4 fallback", page.url)
        content = _extract_with_bs4_heuristic(page.html)

    if not content or len(content.split()) < Config.MIN_CONTENT_WORDS:
        logger.info("Discarding %s — content too short after extraction", page.url)
        return None

    content = _clean_text(content)

    # ── Structural metadata ───────────────────────────────────────────────
    title = _extract_title(page.html)
    headings = _extract_headings(page.html)
    language = _detect_language(content)

    from urllib.parse import urlparse
    domain = urlparse(page.url).netloc.lower()

    doc = {
        "id": _make_doc_id(page.url),
        "url": page.url,
        "domain": domain,
        "title": title,
        "headings": headings,
        "content": content,
        "word_count": len(content.split()),
        "metadata": {
            "crawled_at": datetime.now(timezone.utc).isoformat(),
            "depth": page.depth,
            "fetch_method": page.fetch_method,
            "language": language,
        },
    }

    logger.info(
        "Extracted: %s | %d words | %d headings",
        page.url, doc["word_count"], len(headings)
    )
    return doc


def extract_all(pages: list[RawPage]) -> list[dict]:
    """
    Extract content from all crawled pages.
    Skips failed pages and logs a summary.
    """
    documents = []
    skipped = 0

    for page in pages:
        doc = extract_document(page)
        if doc:
            documents.append(doc)
        else:
            skipped += 1

    logger.info(
        "Extraction complete: %d documents produced, %d pages skipped",
        len(documents), skipped
    )
    return documents
