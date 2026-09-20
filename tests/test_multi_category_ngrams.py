"""
Multi-Category N-gram & Context Alignment Test
"""
import urllib.request
import json

queries = [
    ("running shoes", ["Shoes", "Footwear", "Running", "Sports"]),
    ("red dress", ["Dress", "Gown", "Red"]),
    ("analog watch", ["Watch", "Analog", "Titan", "Fastrack"]),
    ("casual black shirt", ["Shirt", "Black", "Casual"])
]

for query, expected_keywords in queries:
    print(f"\n==========================================")
    print(f"Testing Query: '{query}'")
    print(f"==========================================")
    data = json.dumps({"query": query, "eliminate_noise": True, "max_results": 5}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/search/text",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    res = json.loads(resp.read().decode("utf-8"))
    results = res.get("results", [])
    print(f"Found {len(results)} items (eliminated {res.get('noise_metadata', {}).get('eliminated_count', 0)}):")
    for r in results:
        ngram_info = r.get("ngram_score", "N/A")
        print(f"  Rank #{r['rank']}: {r['display_name']} [{r['articleType']}] | Blended: {r['score']} | N-gram: {ngram_info}")
    
    # Assert top result has at least one expected keyword or matching category
    top = results[0]
    matched = any(kw.lower() in (top['display_name'] + " " + top['articleType']).lower() for kw in expected_keywords)
    print(f"Top Item '{top['display_name']}' matched category expectation? {matched}")
    assert matched, f"Expected top item to match one of {expected_keywords}, got: {top['display_name']}"

print("\nALL CATEGORY N-GRAM & CONTEXT TESTS PASSED SUCCESSFULLY!")
