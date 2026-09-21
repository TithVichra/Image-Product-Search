import os
import sys
import unittest
import numpy as np
from PIL import Image

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine.chunking import chunk_product_description
from search_engine.scoring import compute_dot_product, blend_scores, eliminate_rank_gap_noise, calibrate_cross_modal


class TestChunkingAndScoring(unittest.TestCase):
    def test_chunk_product_description(self):
        sample = {
            "id": 42431,
            "productDisplayName": "Turtle Check Men Navy Blue Shirt",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Shirts",
            "baseColour": "Navy Blue",
            "season": "Fall",
            "year": "2012",
            "usage": "Casual",
            "description": "Navy blue check casual shirt. Has a spread collar, button placket, long sleeves, curved hem."
        }
        chunks = chunk_product_description(sample)
        self.assertGreaterEqual(len(chunks), 2)
        
        # Check chunk types
        chunk_types = [c["chunk_type"] for c in chunks]
        self.assertIn("identity_category", chunk_types)
        self.assertIn("style_attributes", chunk_types)
        print(f"\n[Test] Generated {len(chunks)} chunks:")
        for c in chunks:
            print(f"  - [{c['chunk_type']}]: {c['text']}")

    def test_dot_product_and_blend_scores(self):
        # L2 normalized vectors
        u = np.array([1.0, 0.0, 0.0])
        v = np.array([0.8, 0.6, 0.0])
        dot = compute_dot_product(u, v)
        self.assertAlmostEqual(dot, 0.8, places=4)

        # Mode: image (75% image, 25% calibrated text)
        img_score = 0.90
        txt_score = 0.40
        score_img_mode = blend_scores(img_score, txt_score, search_mode="image")
        self.assertAlmostEqual(score_img_mode, 0.75 * img_score + 0.25 * calibrate_cross_modal(txt_score), places=4)

        # Mode: text (75% text, 25% calibrated image)
        score_txt_mode = blend_scores(img_score, txt_score, search_mode="text")
        self.assertAlmostEqual(score_txt_mode, 0.75 * txt_score + 0.25 * calibrate_cross_modal(img_score), places=4)
        print(f"\n[Test] Blend scores: Image mode={score_img_mode}, Text mode={score_txt_mode}")

    def test_rank_gap_noise_elimination(self):
        """
        Tests that when a sharp drop occurs in candidate ranks,
        subsequent low-confidence noise is trimmed.
        """
        ranked_candidates = [
            {"id": 1, "score": 0.92, "name": "Item 1"},
            {"id": 2, "score": 0.88, "name": "Item 2"},
            {"id": 3, "score": 0.85, "name": "Item 3"},
            {"id": 4, "score": 0.65, "name": "Item 4 (cliff drop: 0.85 - 0.65 = 0.20)"},
            {"id": 5, "score": 0.40, "name": "Noise 5"},
            {"id": 6, "score": 0.25, "name": "Noise 6"}
        ]

        filtered, meta = eliminate_rank_gap_noise(
            ranked_items=ranked_candidates,
            gap_threshold=0.15,
            max_results=10
        )

        # Should cut off at Item 4 because drop was 0.20 >= 0.15
        self.assertEqual(len(filtered), 3)
        self.assertEqual(meta["eliminated_count"], 3)
        self.assertTrue(meta["gap_detected"])
        print(f"\n[Test] Noise filter eliminated {meta['eliminated_count']} items after rank {meta['gap_rank']}")


if __name__ == "__main__":
    unittest.main()
