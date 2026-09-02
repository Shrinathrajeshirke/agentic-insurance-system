from src.agents.tools import search_policy_contracts, web_search

def test_tools():
    print("\n--- 1. Testing Internal Qdrant Search ---")
    results = search_policy_contracts.invoke(
        {"query": "What is the suicide exclusion rule?"}
    )
    for idx, r in enumerate(results, 1):
        print(f"[{idx}] Insurer: {r['insurer']} | Policy: {r['policy_name']} | Score: {r['score']:.4f}")
        print(f"    Excerpt: {r['text'][:140]}...\n")

    print("--- 2. Testing External Web Search ---")
    web_res = web_search.invoke({"query": "HDFC Life claim settlement ratio IRDAI latest"})
    print(f"Web Search Excerpt: {web_res[:250]}...\n")

if __name__ == "__main__":
    test_tools()