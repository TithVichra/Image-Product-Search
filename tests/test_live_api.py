import requests
import json
import os

BASE_URL = "http://127.0.0.1:8000"

def test_text_queries():
    queries = [
        "ball",
        "basketball shoes",
        "red dress",
        "dress for women",
        "dog",
        "men t-shirt"
    ]
    
    print("=== TESTING TEXT QUERIES ===")
    for q in queries:
        resp = requests.post(f"{BASE_URL}/api/search/text", json={"query": q, "eliminate_noise": True})
        assert resp.status_code == 200, f"Query '{q}' failed with {resp.status_code}: {resp.text}"
        data = resp.json()
        print(f"\n--- Query: '{q}' ---")
        print(f"is_out_of_domain: {data.get('is_out_of_domain')}")
        if data.get("is_out_of_domain"):
            print(f"Reason: {data.get('out_of_domain_reason')}")
        print(f"Total found: {data.get('total_found')}")
        for idx, item in enumerate(data.get("results", [])[:3]):
            print(f"  {idx+1}. {item.get('name')} | Color: {item.get('base_colour')} | Gender: {item.get('gender')} | Cat: {item.get('article_type')} | Score: {item.get('score'):.3f}")

def test_image_queries():
    print("\n=== TESTING IMAGE QUERIES ===")
    ball_image = r"C:\Users\User\.gemini\antigravity-ide\brain\4c2847d3-8f4e-4809-a016-c56c7b9625c5\test_sports_ball_1789912639900.jpg"
    if not os.path.exists(ball_image):
        print(f"Image not found at {ball_image}")
        return

    # Test /api/detect
    with open(ball_image, "rb") as f:
        resp = requests.post(f"{BASE_URL}/api/detect", files={"file": f})
    assert resp.status_code == 200, f"Detect failed: {resp.text}"
    detect_data = resp.json()
    print("\n--- Image Detect (Ball) ---")
    print(f"is_out_of_domain: {detect_data.get('is_out_of_domain')}")
    print(f"detected_entity: {detect_data.get('detected_entity')}")
    print(f"reason: {detect_data.get('out_of_domain_reason')}")

    # Test /api/search/image
    with open(ball_image, "rb") as f:
        resp = requests.post(f"{BASE_URL}/api/search/image", files={"file": f})
    assert resp.status_code == 200, f"Search image failed: {resp.text}"
    search_data = resp.json()
    print("\n--- Image Search (Ball) ---")
    print(f"is_out_of_domain: {search_data.get('is_out_of_domain')}")
    print(f"reason: {search_data.get('out_of_domain_reason')}")
    print(f"total_found: {search_data.get('total_found')}")
    print(f"results count: {len(search_data.get('results', []))}")

if __name__ == "__main__":
    test_text_queries()
    test_image_queries()
