"""Unit tests for web-fallback routing and resilient live search."""
from src.agents.search_utils import (
    build_search_queries,
    clean_search_query,
    format_search_results,
    insurers_named_in_query,
    mentions_term,
    needs_web_fallback,
    run_live_search,
)

MAX_LIFE_CHUNK = {
    "insurer": "Max Life",
    "policy_name": "Max Life Smart Secure Plus",
    "text": "Suicide within 12 months of issuance returns 80% of premiums paid.",
}


# ---------------------------------------------------------------- word matching
def test_mentions_term_is_whole_word():
    assert mentions_term("Is LIC a good insurer?", "lic")
    assert not mentions_term("What is the free-look period for this policy?", "lic")
    assert not mentions_term("How do I start a claim?", "star")
    assert not mentions_term("Is healthcare cover included?", "care")


def test_mentions_term_handles_multiword_and_case():
    assert mentions_term("Compare with ADITYA BIRLA Sun Life", "aditya birla")


def test_ordinary_questions_name_no_external_insurer():
    for q in [
        "What is the free-look period for this policy?",
        "How do I start a claim?",
        "Is healthcare cover included?",
    ]:
        assert insurers_named_in_query(q) == [], q


def test_real_insurer_is_detected():
    assert insurers_named_in_query("Tell me about Axis Max Life Smart Term Plan Plus") == ["axis"]


# --------------------------------------------------------------- fallback routing
def test_no_contexts_always_falls_back():
    assert needs_web_fallback("anything", [])


def test_named_external_insurer_missing_from_chunks_falls_back():
    q = "Tell me about Axis Max Life Smart Term Plan Plus"
    assert needs_web_fallback(q, [MAX_LIFE_CHUNK])


def test_named_external_insurer_present_in_chunks_stays_local():
    chunk = dict(MAX_LIFE_CHUNK, insurer="Axis Max Life")
    assert not needs_web_fallback("Tell me about Axis Max Life plans", [chunk])


def test_plain_policy_question_no_longer_triggers_web_search():
    # Regression: "lic" used to match inside "policy".
    assert not needs_web_fallback("What is the free-look period for this policy?", [MAX_LIFE_CHUNK])


# ------------------------------------------------------------------ query building
def test_clean_search_query_strips_filler_and_punctuation():
    assert clean_search_query("Tell me about Axis Max Life Smart Term Plan Plus?") == \
        "Axis Max Life Smart Term Plan Plus"


def test_clean_search_query_never_returns_empty():
    assert clean_search_query("please") == "please"


def test_build_search_queries_adds_india_scope_then_relaxed_variant():
    assert build_search_queries("Tell me about Axis Max Life Smart Term Plan Plus") == [
        "Axis Max Life Smart Term Plan Plus term insurance India",
        "Axis Max Life Smart Term Plan Plus",
    ]


def test_build_search_queries_does_not_double_up_insurance():
    assert build_search_queries("Axis Max Life term insurance premium") == [
        "Axis Max Life term insurance premium"
    ]


# --------------------------------------------------------------- result formatting
def test_format_search_results_includes_source_url_and_skips_empty_bodies():
    text = format_search_results([
        {"title": "Axis Max Life", "href": "https://example.com/plan", "body": "Plan details."},
        {"title": "Empty", "href": "https://example.com/x", "body": ""},
    ])
    assert "https://example.com/plan" in text
    assert "Plan details." in text
    assert "Empty" not in text


# ------------------------------------------------------------------- live search
def make_ddgs(script):
    """Fake DDGS. `script` is a list of results-or-exceptions, one per .text() call."""
    calls = []

    class FakeDDGS:
        def __init__(self, timeout=None):
            self.timeout = timeout

        def __enter__(self):
            return self

        def __exit__(self, *exc):
            return False

        def text(self, q, region=None, max_results=None):
            calls.append((q, region, max_results))
            outcome = script[len(calls) - 1]
            if isinstance(outcome, Exception):
                raise outcome
            return iter(outcome)

    return FakeDDGS, calls


GOOD = [{"title": "T", "href": "https://x.test", "body": "useful excerpt"}]


def test_live_search_success_first_try():
    cls, calls = make_ddgs([GOOD])
    results, error = run_live_search("Axis plan", cls, sleep=lambda s: None)
    assert results == GOOD and error is None
    assert len(calls) == 1
    assert calls[0][1] == "in-en"


def test_live_search_retries_relaxed_query_after_failure():
    cls, calls = make_ddgs([RuntimeError("boom"), GOOD])
    results, error = run_live_search("Axis plan", cls, sleep=lambda s: None)
    assert results == GOOD and error is None
    assert len(calls) == 2
    assert calls[0][0] != calls[1][0]


def test_live_search_reports_rate_limit_style_failures():
    cls, _ = make_ddgs([RuntimeError("202 Ratelimit"), RuntimeError("202 Ratelimit")])
    results, error = run_live_search("Axis plan", cls, sleep=lambda s: None)
    assert results == []
    assert "RuntimeError" in error and "Ratelimit" in error


def test_live_search_empty_results_are_an_error_not_success():
    cls, _ = make_ddgs([[], []])
    results, error = run_live_search("Axis plan", cls, sleep=lambda s: None)
    assert results == []
    assert "no results" in error


def test_live_search_filters_results_without_body():
    cls, _ = make_ddgs([[{"title": "no body", "href": "u", "body": ""}] + GOOD])
    results, error = run_live_search("Axis plan", cls, sleep=lambda s: None)
    assert results == GOOD and error is None


def test_live_search_pauses_between_attempts():
    slept = []
    cls, _ = make_ddgs([[], GOOD])
    run_live_search("Axis plan", cls, pause_seconds=1.5, sleep=slept.append)
    assert slept == [1.5]