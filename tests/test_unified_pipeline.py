import os
import io
import requests
from PIL import Image

BASE_URL = "http://127.0.0.1:8000"


def test_preprocessing_and_dataset():
    """
    Task 1 Verification:
    Verify styles.csv exists, productDisplayName is standardized, and referenced images exist.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    styles_csv = os.path.join(base_dir, "data", "fashion-dataset", "styles.csv")
    images_dir = os.path.join(base_dir, "data", "fashion-dataset", "images")
    
    assert os.path.exists(styles_csv), f"styles.csv not found at {styles_csv}"
    assert os.path.exists(images_dir), f"images/ dir not found at {images_dir}"


def test_api_status_unified_count():
    """
    Task 2 Verification:
    Verify the unified collection exists and is reported online by /api/status.
    """
    resp = requests.get(f"{BASE_URL}/api/status", timeout=10)
    assert resp.status_code == 200, f"Status check failed: {resp.text}"
    data = resp.json()
    assert data.get("status") == "online"
    vstats = data.get("vector_stats", {})
    assert "unified_count" in vstats, "unified_count not found in vector_stats"
    assert vstats["unified_count"] > 0, "Unified collection is empty"


def test_api_search_text_contract():
    """
    Task 4 & Task 3 Verification:
    POST /search/text
    Input: {"query": "navy blue sports shoes", "top_k": 10}
    Output: Top 10 unique items with product_id, name, image_url, final_score, breakdown.
    """
    payload = {
        "query": "navy blue sports shoes",
        "top_k": 10
    }
    resp = requests.post(f"{BASE_URL}/search/text", json=payload, timeout=15)
    assert resp.status_code == 200, f"Search text failed: {resp.text}"
    results = resp.json()
    assert isinstance(results, list), "Response must be a JSON list"
    assert len(results) > 0, "Should return at least 1 match"
    assert len(results) <= 10, "Should return at most top_k items"

    # Verify unique product IDs
    seen_ids = set()
    prev_score = float("inf")
    for idx, item in enumerate(results):
        pid = item.get("product_id")
        assert pid is not None, f"Item {idx} missing product_id"
        assert pid not in seen_ids, f"Duplicate product_id found: {pid}"
        seen_ids.add(pid)

        assert "name" in item, f"Item {idx} missing name"
        assert "image_url" in item, f"Item {idx} missing image_url"
        assert "final_score" in item, f"Item {idx} missing final_score"
        assert "breakdown" in item, f"Item {idx} missing breakdown"
        
        breakdown = item["breakdown"]
        assert "image_score" in breakdown, f"Item {idx} missing image_score in breakdown"
        assert "title_score" in breakdown, f"Item {idx} missing title_score in breakdown"

        # Verify descending order of final_score
        assert item["final_score"] <= prev_score + 1e-6, "Results not sorted by final_score descending"
        prev_score = item["final_score"]


def test_api_search_image_contract():
    """
    Task 4 & Task 3 Verification:
    POST /search/image
    Input: multipart/form-data image file
    Output: Top 10 unique items with product_id, name, image_url, final_score, breakdown.
    """
    # Create an in-memory sample image
    img = Image.new("RGB", (224, 224), color=(30, 60, 150))
    buf = io.BytesIO()
    img.save(buf, format="JPEG")
    buf.seek(0)

    files = {"file": ("test_query.jpg", buf, "image/jpeg")}
    data = {"top_k": 10}

    resp = requests.post(f"{BASE_URL}/search/image", files=files, data=data, timeout=20)
    assert resp.status_code == 200, f"Search image failed: {resp.text}"
    results = resp.json()
    assert isinstance(results, list), "Response must be a JSON list"
    assert len(results) > 0, "Should return at least 1 match"
    assert len(results) <= 10, "Should return at most top_k items"

    seen_ids = set()
    prev_score = float("inf")
    for idx, item in enumerate(results):
        pid = item.get("product_id")
        assert pid is not None, f"Item {idx} missing product_id"
        assert pid not in seen_ids, f"Duplicate product_id found: {pid}"
        seen_ids.add(pid)

        assert "name" in item, f"Item {idx} missing name"
        assert "image_url" in item, f"Item {idx} missing image_url"
        assert "final_score" in item, f"Item {idx} missing final_score"
        assert "breakdown" in item, f"Item {idx} missing breakdown"
        
        breakdown = item["breakdown"]
        assert "image_score" in breakdown, f"Item {idx} missing image_score in breakdown"
        assert "title_score" in breakdown, f"Item {idx} missing title_score in breakdown"

        # Verify descending order
        assert item["final_score"] <= prev_score + 1e-6, "Results not sorted by final_score descending"
        prev_score = item["final_score"]


if __name__ == "__main__":
    tests = [
        test_preprocessing_and_dataset,
        test_api_status_unified_count,
        test_api_search_text_contract,
        test_api_search_image_contract
    ]
    passed = 0
    print("=== RUNNING UNIFIED MULTIMODAL PIPELINE TESTS ===")
    for test in tests:
        try:
            test()
            print(f"PASS: {test.__name__}")
            passed += 1
        except Exception as e:
            err_msg = str(e).split("\n")[0][:200]
            print(f"FAIL: {test.__name__} -> {err_msg}")
    print(f"\nResult: {passed}/{len(tests)} tests passed.")
    if passed != len(tests):
        exit(1)
