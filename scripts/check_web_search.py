"""Diagnose whether live DuckDuckGo search works from THIS machine.

Run from the project root, locally and again in a Render shell:
    python scripts/check_web_search.py "Axis Max Life Smart Term Plan Plus"
"""
import sys

sys.path.insert(0, ".")

try:
    from ddgs import DDGS
    package = "ddgs (current)"
except ImportError:
    from duckduckgo_search import DDGS
    package = "duckduckgo_search (DEPRECATED - run: pip install -U ddgs)"

from src.agents.search_utils import build_search_queries, run_live_search

query = " ".join(sys.argv[1:]) or "Axis Max Life Smart Term Plan Plus"
print(f"Package : {package}")
print(f"Queries : {build_search_queries(query)}")

results, error = run_live_search(query, DDGS)
if results:
    print(f"\nOK: {len(results)} results")
    for r in results:
        print(f" - {r.get('title')}  {r.get('href')}")
else:
    print(f"\nFAILED: {error}")
    print("If this works on your laptop but fails on Render, the host's IP is being blocked/rate-limited.")
    sys.exit(1)