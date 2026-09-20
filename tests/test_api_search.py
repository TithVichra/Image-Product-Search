"""
End-to-end Search Pipeline Test
"""

import os
import sys

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def test_text_search():
    import urllib.request
    import json

    query = "blue jeans"
    print(f"\n--- Testing Text Search for: '{query}' ---")
    data = json.dumps({"query": query, "eliminate_noise": True}).encode("utf-8")
    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/search/text",
        data=data,
        headers={"Content-Type": "application/json"}
    )
    resp = urllib.request.urlopen(req)
    search_res = json.loads(resp.read().decode("utf-8"))

    results = search_res["results"]
    noise_meta = search_res["noise_metadata"]

    print(f"Kept: {len(results)} items, Eliminated Noise: {noise_meta.get('eliminated_count', 0)}")
    for r in results[:5]:
        print(f"Rank #{r['rank']}: {r['display_name']} | Blended: {r['score']} | Text: {r['text_score']} | Img: {r['image_score']}")

    assert len(results) > 0, "Expected at least 1 result"
    top_item = results[0]
    print(f"Top match ID: {top_item['id']}, Title: {top_item['display_name']}")
    assert "Jeans" in top_item["display_name"] or "Blue" in top_item["display_name"], f"Expected jeans or blue item, got: {top_item['display_name']}"
    print("SUCCESS: Text search returns top relevant fashion item!")

def test_image_search():
    import urllib.request
    import json

    img_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "media", "images", "39386.jpg")
    print(f"\n--- Testing HTTP Image Search for: {img_path} ---")

    boundary = "----TestBoundary12345"
    fn = os.path.basename(img_path)
    with open(img_path, "rb") as f:
        file_bytes = f.read()

    body = (
        f"--{boundary}\r\n"
        f'Content-Disposition: form-data; name="file"; filename="{fn}"\r\n'
        f"Content-Type: image/jpeg\r\n\r\n"
    ).encode("utf-8") + file_bytes + (
        f"\r\n--{boundary}\r\n"
        f'Content-Disposition: form-data; name="use_crop"\r\n\r\n'
        f"true\r\n"
        f"--{boundary}--\r\n"
    ).encode("utf-8")

    req = urllib.request.Request(
        "http://127.0.0.1:8000/api/search/image",
        data=body,
        headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
    )
    resp = urllib.request.urlopen(req)
    search_res = json.loads(resp.read().decode("utf-8"))

    results = search_res["results"]
    print(f"Detection: crop_applied={search_res.get('crop_applied')}, boxes={len(search_res.get('detected_boxes', []))}")
    print(f"Found {len(results)} matching items:")
    for r in results[:5]:
        print(f"Rank #{r['rank']}: {r['display_name']} | Blended: {r['score']} | Img: {r['image_score']} | Text: {r['text_score']}")

    assert len(results) > 0
    top_item = results[0]
    print(f"Top match: {top_item['id']} - {top_item['display_name']}")
    assert top_item["id"] == 39386, f"Expected product 39386 as top match, got {top_item['id']}"
    print("SUCCESS: Image search returns exact matching Kaggle item with high similarity!")

if __name__ == "__main__":
    test_text_search()
    test_image_search()
