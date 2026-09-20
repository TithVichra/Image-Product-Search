import os
import sys
import unittest
import numpy as np

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from search_engine.chunking import (
    chunk_product_description,
    generate_attribute_ngrams,
    extract_ngrams,
    find_matching_lexicon_entry
)
from search_engine.scoring import (
    compute_ngram_context_score,
    blend_scores
)


class TestNgramAndContext(unittest.TestCase):
    def setUp(self):
        self.jeans_product = {
            "id": 1001,
            "productDisplayName": "Peter England Men Blue Jeans",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Bottomwear",
            "articleType": "Jeans",
            "baseColour": "Blue",
            "season": "Summer",
            "year": "2016",
            "usage": "Casual",
            "description": "Mid-rise blue denim jeans for men with washed finish."
        }
        self.shirt_product = {
            "id": 1002,
            "productDisplayName": "United Colors of Benetton Men Blue Casual Shirt",
            "gender": "Men",
            "masterCategory": "Apparel",
            "subCategory": "Topwear",
            "articleType": "Shirts",
            "baseColour": "Blue",
            "season": "Summer",
            "year": "2016",
            "usage": "Casual",
            "description": "Blue cotton casual long-sleeve shirt."
        }

    def test_extract_ngrams(self):
        tokens = ["peter", "england", "men", "blue", "jeans"]
        bigrams = extract_ngrams(tokens, 2)
        self.assertIn("peter england", bigrams)
        self.assertIn("blue jeans", bigrams)

        trigrams = extract_ngrams(tokens, 3)
        self.assertIn("men blue jeans", trigrams)

    def test_generate_attribute_ngrams(self):
        ngrams = generate_attribute_ngrams(self.jeans_product)
        self.assertIn("blue jeans", ngrams["bigrams"])
        self.assertIn("men blue jeans", ngrams["trigrams"])
        # Should include synonym bigrams from lexicon
        self.assertTrue(any("denim" in bg for bg in ngrams["bigrams"]))

    def test_chunk_has_ngram_and_semantic_types(self):
        chunks = chunk_product_description(self.jeans_product)
        chunk_types = [c["chunk_type"] for c in chunks]
        self.assertIn("ngram_context", chunk_types)
        self.assertIn("semantic_context", chunk_types)
        
        ngram_chunk = next(c for c in chunks if c["chunk_type"] == "ngram_context")
        self.assertIn("blue jeans", ngram_chunk["text"])

        semantic_chunk = next(c for c in chunks if c["chunk_type"] == "semantic_context")
        self.assertIn("denim", semantic_chunk["text"].lower())

    def test_compute_ngram_context_score_exact_vs_mismatch(self):
        query = "blue jeans"
        
        # Test jeans product (exact category match & bigram match)
        score_jeans, meta_jeans = compute_ngram_context_score(query, self.jeans_product)
        self.assertGreaterEqual(score_jeans, 0.85)
        self.assertEqual(meta_jeans["category_target"], "jeans")
        self.assertGreaterEqual(meta_jeans["context_multiplier"], 1.0)

        # Test shirt product (unigram 'blue' matches, but 'jeans' is missing and category contradicts)
        score_shirt, meta_shirt = compute_ngram_context_score(query, self.shirt_product)
        self.assertLess(score_shirt, 0.40)
        self.assertLess(meta_shirt["context_multiplier"], 0.50)

        print(f"\n[Test] N-gram Context Score for query '{query}':")
        print(f"  - Peter England Blue Jeans: {score_jeans:.4f} (meta: {meta_jeans})")
        print(f"  - Blue Casual Shirt:        {score_shirt:.4f} (meta: {meta_shirt})")
        self.assertGreater(score_jeans, score_shirt * 2.0)

    def test_hybrid_blend_scores(self):
        # Jeans candidate: high vector sim + high ngram sim
        score_jeans = blend_scores(
            image_sim=0.70,
            text_sim=0.75,
            ngram_sim=0.95,
            search_mode="text"
        )
        # Shirt candidate: high vector sim due to color + apparel, but low ngram sim
        score_shirt = blend_scores(
            image_sim=0.68,
            text_sim=0.72,
            ngram_sim=0.20,
            search_mode="text"
        )
        self.assertGreater(score_jeans, score_shirt)
        print(f"\n[Test] Hybrid Blended Scores: Jeans={score_jeans:.4f}, Shirt={score_shirt:.4f}")


if __name__ == "__main__":
    unittest.main()
