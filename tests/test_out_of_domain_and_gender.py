"""
Automated Test for:
1. Out-of-domain image rejection (e.g. Dog)
2. Category isolation (e.g. Red Dress vs T-shirt / Bra / Camisole)
3. Gender enforcement (e.g. Women queries excluding Men products)
"""

import os
import json
import urllib.request
import unittest


class TestDomainCategoryAndGender(unittest.TestCase):
    BASE_URL = "http://127.0.0.1:8000"

    def test_dog_image_rejected_as_out_of_domain(self):
        dog_path = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "scripts", "dog_test.jpg")
        self.assertTrue(os.path.exists(dog_path), f"dog_test.jpg must exist at {dog_path}")

        boundary = "----TestDogBoundary"
        with open(dog_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="dog.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="use_crop"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.BASE_URL}/api/search/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode("utf-8"))

        print("\n[Test] Dog Image Search Result:")
        print(f"  is_out_of_domain: {data.get('is_out_of_domain')}")
        print(f"  detected_entity: {data.get('detected_entity')}")
        print(f"  results count: {len(data.get('results', []))}")

        self.assertTrue(data.get("is_out_of_domain"), "Dog image must be flagged as out-of-domain")
        self.assertEqual(len(data.get("results", [])), 0, "Out-of-domain image must return 0 results")
        self.assertIn("dog", (data.get("detected_entity") or "").lower())

    def test_red_dress_category_isolation(self):
        data = json.dumps({"query": "red dress", "eliminate_noise": True, "max_results": 10}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/search/text",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        res = json.loads(resp.read().decode("utf-8"))
        results = res.get("results", [])

        self.assertGreater(len(results), 0, "Expected at least 1 dress result")
        print("\n[Test] 'red dress' Search Results:")
        for r in results:
            article = r.get("articleType", "")
            display_name = r.get("display_name", "")
            print(f"  Rank #{r['rank']}: {display_name} [{article}] | Score: {r['score']}")

            # Assert no bras or camisoles or t-shirts
            self.assertNotIn("camisole", article.lower())
            self.assertNotIn("bra", article.lower())
            self.assertNotIn("tshirt", article.lower())
            self.assertNotIn("t-shirt", article.lower())
            self.assertIn(r.get("gender"), ["Women", "Girls"], "Expected Women's or Girls' garments")
            
            # Color strictness: ensure color is in red family or title has red
            color = (r.get("baseColour") or r.get("base_colour") or "").lower()
            name = (r.get("display_name") or r.get("name") or "").lower()
            red_words = {"red", "crimson", "maroon", "burgundy", "ruby", "multi coloured"}
            has_red = any(rw in color or rw in name for rw in red_words)
            self.assertTrue(has_red, f"Result '{name}' [{color}] must match red color family")

    def test_ball_text_query_rejected(self):
        data = json.dumps({"query": "ball", "eliminate_noise": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/search/text",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        res = json.loads(resp.read().decode("utf-8"))
        print("\n[Test] Query 'ball' Result:")
        print(f"  is_out_of_domain: {res.get('is_out_of_domain')}")
        print(f"  total_found: {res.get('total_found')}")
        self.assertTrue(res.get("is_out_of_domain"), "Pure 'ball' query must be rejected as out of domain")
        self.assertEqual(len(res.get("results", [])), 0, "No results (e.g. shoes) should match 'ball'")

    def test_basketball_shoes_allowed(self):
        data = json.dumps({"query": "basketball shoes", "eliminate_noise": True}).encode("utf-8")
        req = urllib.request.Request(
            f"{self.BASE_URL}/api/search/text",
            data=data,
            headers={"Content-Type": "application/json"}
        )
        resp = urllib.request.urlopen(req)
        res = json.loads(resp.read().decode("utf-8"))
        print("\n[Test] Query 'basketball shoes' Result:")
        print(f"  is_out_of_domain: {res.get('is_out_of_domain')}")
        print(f"  total_found: {res.get('total_found')}")
        self.assertFalse(res.get("is_out_of_domain"), "'basketball shoes' is valid fashion query")
        self.assertGreater(len(res.get("results", [])), 0, "Should find sports shoes for 'basketball shoes'")

    def test_ball_image_rejected(self):
        ball_path = r"C:\Users\User\.gemini\antigravity-ide\brain\4c2847d3-8f4e-4809-a016-c56c7b9625c5\test_sports_ball_1789912639900.jpg"
        if not os.path.exists(ball_path):
            self.skipTest("Sports ball test image not found")

        boundary = "----TestBallBoundary"
        with open(ball_path, "rb") as f:
            file_bytes = f.read()

        body = (
            f"--{boundary}\r\n"
            f'Content-Disposition: form-data; name="file"; filename="ball.jpg"\r\n'
            f"Content-Type: image/jpeg\r\n\r\n"
        ).encode("utf-8") + file_bytes + (
            f"\r\n--{boundary}\r\n"
            f'Content-Disposition: form-data; name="use_crop"\r\n\r\n'
            f"true\r\n"
            f"--{boundary}--\r\n"
        ).encode("utf-8")

        req = urllib.request.Request(
            f"{self.BASE_URL}/api/search/image",
            data=body,
            headers={"Content-Type": f"multipart/form-data; boundary={boundary}"}
        )
        resp = urllib.request.urlopen(req)
        data = json.loads(resp.read().decode("utf-8"))

        print("\n[Test] Sports Ball Image Search Result:")
        print(f"  is_out_of_domain: {data.get('is_out_of_domain')}")
        print(f"  results count: {len(data.get('results', []))}")
        self.assertTrue(data.get("is_out_of_domain"), "Sports ball image must be flagged out-of-domain")
        self.assertEqual(len(data.get("results", [])), 0, "No results for sports ball")

    def test_women_gender_isolation(self):
        queries = ["black shoes for women", "red dress for women"]
        for q in queries:
            data = json.dumps({"query": q, "eliminate_noise": True, "max_results": 10}).encode("utf-8")
            req = urllib.request.Request(
                f"{self.BASE_URL}/api/search/text",
                data=data,
                headers={"Content-Type": "application/json"}
            )
            resp = urllib.request.urlopen(req)
            res = json.loads(resp.read().decode("utf-8"))
            results = res.get("results", [])

            self.assertGreater(len(results), 0, f"Expected results for query '{q}'")
            print(f"\n[Test] Query '{q}' Gender Results:")
            for r in results:
                gender = r.get("gender", "")
                display_name = r.get("display_name", "")
                print(f"  {display_name} | Gender: {gender}")
                self.assertNotIn(gender, ["Men", "Boys"], f"Men's items must not appear for '{q}', found: {display_name}")


if __name__ == "__main__":
    unittest.main()
