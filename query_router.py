"""Smart query router: decides whether to use RAG, LLM knowledge, or web search.

This module replaces the naive keyword-based routing in rag.py with a context-aware
strategy that evaluates retrieval quality in real-time before triggering web search.

Strategy:
    1. Check if query is explicit web/tool command → route to tool
    2. Try RAG retrieval; if high-quality matches found → use RAG
    3. If retrieval is weak/empty → use LLM's internal knowledge
    4. Only route to web if query is time-bound OR retrieval explicitly insufficient
"""

from dataclasses import dataclass
from typing import Literal


@dataclass
class RouterDecision:
    """Routing decision with reasoning."""
    source: Literal["rag", "internal_knowledge", "web_search", "tool"]
    confidence: float  # 0.0–1.0
    reasoning: str
    force_web: bool = False  # Set True if /web command or explicit time-bound query


# ── Hard triggers (explicit commands and critical time-bounds) ──────────────────

HARD_WEB_TRIGGERS = {
    "latest", "breaking", "today", "now", "current", "just announced",
    "this week", "this month", "this year", "2024", "2025", "2026",
    "live", "real-time", "stock price", "weather",
}

TOOL_TRIGGERS = {
    "weather": "weather",
    "cve-": "cve",
    "dns": "dns",
    "patch": "cve",
}

# ── Soft triggers (increase web score, don't force) ────────────────────────────

SOFT_WEB_INDICATORS = {
    "latest", "recent", "news", "trend", "release", "announce",
    "update", "patch", "vulnerability", "github", "stackoverflow",
    "new exploit", "poc", "writeup", "documentation",
}

# ── Signals that the LLM knows this (no retrieval needed) ───────────────────────

INTERNAL_KNOWLEDGE_INDICATORS = {
    "explain", "what is", "define", "concept", "theory", "philosophy",
    "history of", "how does", "why", "fundamental",
}


def is_hard_web_trigger(query: str) -> bool:
    """True if query is explicitly time-bound or asks for live data."""
    q = query.lower()
    return any(trigger in q for trigger in HARD_WEB_TRIGGERS)


def is_tool_trigger(query: str) -> tuple[bool, str | None]:
    """Check if query should route to a specific tool (weather, CVE, DNS)."""
    q = query.lower()
    for pattern, tool in TOOL_TRIGGERS.items():
        if pattern in q:
            return True, tool
    return False, None


def estimate_retrieval_quality(chunks: list[dict], query: str) -> float:
    """
    Estimate how well the retrieved documents answer the query.
    Returns 0.0–1.0 confidence that RAG has good answers.
    """
    if not chunks:
        return 0.0
    
    # Score based on:
    # 1. Max similarity score (top-1 hit)
    # 2. Number of relevant chunks (avg > 0.6)
    # 3. Source diversity
    
    scores = [float(c.get("score", 0.0)) for c in chunks]
    if not scores:
        return 0.0
    
    max_score = max(scores)
    avg_score = sum(scores) / len(scores)
    
    # Count how many chunks meet the relevance threshold (0.6)
    relevant_count = sum(1 for s in scores if s >= 0.6)
    
    # Diversity: do we have multiple sources?
    sources = {str(c.get("source", "")).strip() for c in chunks if c.get("source")}
    source_diversity = min(1.0, len(sources) / 2.0)
    
    # Composite score:
    # - 50% on max score (is the best match relevant?)
    # - 25% on average (are most matches relevant?)
    # - 15% on number of good chunks (do we have enough?)
    # - 10% on source diversity
    quality = (
        0.50 * max_score +
        0.25 * avg_score +
        0.10 * min(1.0, relevant_count / 3.0) +
        0.05 * source_diversity
    )
    
    return min(1.0, quality)


def should_force_internal_knowledge(query: str) -> bool:
    """
    Check if the query is asking for a conceptual/definitional answer
    that the LLM likely knows from pretraining.
    """
    q = query.lower()
    return any(indicator in q for indicator in INTERNAL_KNOWLEDGE_INDICATORS)


def route_query(
    query: str,
    retrieval_chunks: list[dict],
    quality_threshold: float = 0.55,
) -> RouterDecision:
    """
    Main routing function: decide which source to use for this query.
    
    Args:
        query: User query
        retrieval_chunks: Results from RAG retrieval (list of {score, text, source})
        quality_threshold: Minimum retrieval quality (0.0–1.0) to trust RAG
    
    Returns:
        RouterDecision with source, confidence, and reasoning
    """
    
    # ── Check for explicit tool commands ────────────────────────────────────
    is_tool, tool_name = is_tool_trigger(query)
    if is_tool:
        return RouterDecision(
            source="tool",
            confidence=0.95,
            reasoning=f"Explicit {tool_name} query detected.",
            force_web=False,
        )
    
    # ── Check for hard web triggers (time-bound questions) ───────────────────
    if is_hard_web_trigger(query):
        return RouterDecision(
            source="web_search",
            confidence=0.90,
            reasoning="Hard web trigger detected (time-bound or live data requested).",
            force_web=True,
        )
    
    # ── Evaluate retrieval quality ──────────────────────────────────────────
    retrieval_quality = estimate_retrieval_quality(retrieval_chunks, query)
    
    # ── Check if query is asking for knowledge the LLM likely knows ─────────
    if should_force_internal_knowledge(query) and retrieval_quality < 0.50:
        return RouterDecision(
            source="internal_knowledge",
            confidence=0.85,
            reasoning=(
                f"Conceptual question; retrieval quality low ({retrieval_quality:.2f}). "
                "Using LLM knowledge."
            ),
            force_web=False,
        )
    
    # ── Good retrieval → use RAG ──────────────────────────────────────────────
    if retrieval_quality >= quality_threshold:
        return RouterDecision(
            source="rag",
            confidence=retrieval_quality,
            reasoning=f"RAG retrieval quality is good ({retrieval_quality:.2f}).",
            force_web=False,
        )
    
    # ── Weak retrieval → decide between internal knowledge and web search ────
    # Check for soft web indicators
    q = query.lower()
    has_soft_indicators = any(indicator in q for indicator in SOFT_WEB_INDICATORS)
    
    if has_soft_indicators:
        # Soft web indicators + weak retrieval → try web
        return RouterDecision(
            source="web_search",
            confidence=0.70,
            reasoning=(
                f"Weak RAG retrieval ({retrieval_quality:.2f}) + soft web indicators detected. "
                "Attempting web search."
            ),
            force_web=False,
        )
    
    # ── Default: fall back to LLM knowledge ─────────────────────────────────
    return RouterDecision(
        source="internal_knowledge",
        confidence=0.65,
        reasoning=(
            f"Weak RAG retrieval ({retrieval_quality:.2f}). "
            "Using LLM's internal knowledge."
        ),
        force_web=False,
    )


# ── Integration helper ────────────────────────────────────────────────────────

def should_skip_web_auto_routing(query: str) -> bool:
    """
    Check if a query should NOT be auto-routed to web, even if the old
    _should_web_search() returns True. Use this in rag.py to suppress
    overeager web search for generic questions.
    
    Return True → skip web search, use RAG/internal knowledge instead.
    """
    q = query.lower()
    
    # Generic question patterns that the LLM can answer from knowledge
    # Covers: conceptual, definitional, educational, teaching, how-to queries
    skip_patterns = [
        "what is",
        "how does",
        "how is",
        "how can i",
        "how to",
        "explain",
        "define",
        "tell me about",
        "describe",
        "help me with",
        "teach me",
        "tutori",  # tutorial, tutoring
        "guide on",
        "help with",
        "can you help",
        "what are",
        "what does",
        "what's the",
        "why is",
        "why do",
        "difference between",
        "compare",
    ]
    
    # Don't skip if there's a hard time-bound indicator
    if is_hard_web_trigger(query):
        return False
    
    # Skip if it's a generic educational question
    if any(pattern in q for pattern in skip_patterns):
        return True
    
    return False
