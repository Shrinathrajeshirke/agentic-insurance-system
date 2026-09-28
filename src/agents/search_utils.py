"""Pure helpers for the live-web fallback path.

Kept free of LangChain / Qdrant imports so they can be unit-tested in isolation.
"""
import re
import time
from typing import Any, Callable, Dict, List, Optional, Tuple

# Prefix that `web_search` puts on its output when live search failed and it is
# returning the static benchmark table instead. Callers use it to avoid
# presenting static data to the LLM as if it were a live finding.
STATIC_FALLBACK_MARKER = "[STATIC_BENCHMARK_FALLBACK]"

# Insurers that are NOT expected to be in the local corpus. If the user names
# one and no retrieved chunk mentions it, we go to the web.
EXTERNAL_INSURER_TERMS = [
    "axis", "lic", "sbi", "kotak", "bajaj", "pnb", "canara", "star", "care", "aditya birla",
]

_FILLER = re.compile(
    r"(?i)\b(tell me about|what is|what are|how does|can you explain|explain|"
    r"details on|details about|please|the)\b"
)


def mentions_term(text: str, term: str) -> bool:
    """Whole-word / whole-phrase match. `lic` must not match inside `policy`."""
    pattern = rf"(?<!\w){re.escape(term.lower())}(?!\w)"
    return re.search(pattern, (text or "").lower()) is not None


def insurers_named_in_query(query: str) -> List[str]:
    return [t for t in EXTERNAL_INSURER_TERMS if mentions_term(query, t)]


def needs_web_fallback(query: str, contexts: List[Dict[str, Any]]) -> bool:
    """True when local retrieval cannot answer the question.

    - No chunks retrieved -> fall back.
    - Query names an external insurer and no retrieved chunk mentions it -> fall back.
    - Otherwise the local chunks are good enough.
    """
    if not contexts:
        return True

    named = insurers_named_in_query(query)
    if not named:
        return False

    for c in contexts:
        haystack = f"{c.get('insurer', '')} {c.get('policy_name', '')} {c.get('text', '')}"
        if any(mentions_term(haystack, t) for t in named):
            return False
    return True


def clean_search_query(query: str) -> str:
    """Strip conversational filler so the search engine sees keywords."""
    cleaned = _FILLER.sub(" ", query or "")
    cleaned = re.sub(r"[?!]+", " ", cleaned)
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned or (query or "").strip()


def build_search_queries(query: str) -> List[str]:
    """Primary (India-scoped) query first, then a relaxed keyword-only variant."""
    cleaned = clean_search_query(query)
    if re.search(r"(?i)\binsurance\b", cleaned):
        return [cleaned]
    return [f"{cleaned} term insurance India", cleaned]


def format_search_results(results: List[Dict[str, Any]]) -> str:
    blocks = []
    for r in results:
        body = r.get("body")
        if not body:
            continue
        title = r.get("title", "Untitled")
        url = r.get("href") or r.get("url") or ""
        source = f"{title} ({url})" if url else title
        blocks.append(f"Source: {source}\nExcerpt: {body}")
    return "\n\n---\n\n".join(blocks)


def run_live_search(
    query: str,
    ddgs_cls: Any,
    *,
    max_results: int = 5,
    region: str = "in-en",
    timeout: int = 8,
    pause_seconds: float = 1.0,
    sleep: Callable[[float], None] = time.sleep,
) -> Tuple[List[Dict[str, Any]], Optional[str]]:
    """Try each query variant once. Returns (results, error_summary).

    `error_summary` is None on success, otherwise a description of what went
    wrong (exception types, empty result sets) so callers can log it.
    """
    errors: List[str] = []
    for i, q in enumerate(build_search_queries(query)):
        if i > 0:
            sleep(pause_seconds)
        try:
            with ddgs_cls(timeout=timeout) as ddgs:
                raw = list(ddgs.text(q, region=region, max_results=max_results))
            usable = [r for r in raw if r.get("body")]
            if usable:
                return usable, None
            errors.append(f"no results for {q!r} (possible rate limiting)")
        except Exception as exc:  # network errors, rate limits, library changes
            errors.append(f"{type(exc).__name__}: {exc}")
    return [], "; ".join(errors)