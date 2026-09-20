"""
Embedded Qdrant Vector Database Service
Configured with Distance.DOT (Dot Product) similarity.
Stores product images and description chunks in local storage.
"""

import os
from typing import List, Dict, Any, Tuple, Optional
import numpy as np
from qdrant_client import QdrantClient
from qdrant_client.http import models
from .scoring import blend_scores, eliminate_rank_gap_noise, compute_ngram_context_score

COLLECTION_IMAGES = "fashion_products_image"
COLLECTION_CHUNKS = "fashion_products_text_chunks"
VECTOR_DIM = 512

_vector_db_instance = None


class VectorDB:
    def __init__(self, storage_path: str = None):
        if storage_path is None:
            # Default to qdrant_storage inside the project directory
            base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
            storage_path = os.path.join(base_dir, "qdrant_storage")

        os.makedirs(storage_path, exist_ok=True)
        self.storage_path = storage_path
        print(f"[VectorDB] Initializing embedded Qdrant at: {self.storage_path}")
        self.client = QdrantClient(path=self.storage_path)
        self._init_collections()

    def _init_collections(self):
        """
        Creates collections with Distance.DOT if they don't already exist.
        """
        existing = [c.name for c in self.client.get_collections().collections]

        if COLLECTION_IMAGES not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_IMAGES,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.DOT)
            )
            print(f"[VectorDB] Created collection '{COLLECTION_IMAGES}' with Distance.DOT")

        if COLLECTION_CHUNKS not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_CHUNKS,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.DOT)
            )
            print(f"[VectorDB] Created collection '{COLLECTION_CHUNKS}' with Distance.DOT")

    def reset_collections(self):
        """
        Clears and recreates collections for fresh dataset indexing.
        """
        print("[VectorDB] Resetting collections for fresh Kaggle dataset indexing...")
        try:
            self.client.delete_collection(COLLECTION_IMAGES)
        except Exception:
            pass
        try:
            self.client.delete_collection(COLLECTION_CHUNKS)
        except Exception:
            pass
        self._init_collections()

    def upsert_products_batch(
        self,
        products_data: List[Dict[str, Any]],
        image_embeddings: np.ndarray,
        chunk_embeddings_map: Dict[int, np.ndarray]
    ):
        """
        Upserts a batch of products into both image and text collections.
        products_data: List of dicts, each with keys 'id', 'name', 'category', etc.
        image_embeddings: (N, 512) float32 array
        chunk_embeddings_map: { product_id: (num_chunks, 512) float32 array }
        """
        img_points = []
        chunk_points = []
        global_chunk_idx = 0

        for i, prod in enumerate(products_data):
            pid = int(prod["id"])
            img_vec = image_embeddings[i].tolist()
            
            # Point for product image
            img_points.append(
                models.PointStruct(
                    id=pid,
                    vector=img_vec,
                    payload=prod
                )
            )

            # Points for text chunks
            if pid in chunk_embeddings_map:
                chunk_vecs = chunk_embeddings_map[pid]
                chunks_info = prod.get("chunks", [])
                for c_idx, c_info in enumerate(chunks_info):
                    c_vec = chunk_vecs[c_idx].tolist()
                    chunk_payload = {
                        "product_id": pid,
                        "product_name": prod.get("name", ""),
                        "chunk_id": c_info.get("chunk_id", c_idx),
                        "chunk_type": c_info.get("chunk_type", "description"),
                        "chunk_text": c_info.get("text", ""),
                        "product_metadata": prod
                    }
                    # Generate deterministic integer ID for chunk
                    chunk_point_id = pid * 1000 + c_idx
                    chunk_points.append(
                        models.PointStruct(
                            id=chunk_point_id,
                            vector=c_vec,
                            payload=chunk_payload
                        )
                    )

        if img_points:
            self.client.upsert(collection_name=COLLECTION_IMAGES, points=img_points)
        if chunk_points:
            self.client.upsert(collection_name=COLLECTION_CHUNKS, points=chunk_points)

    def search(
        self,
        query_vector: np.ndarray,
        search_mode: str = "text",
        query_text: Optional[str] = None,
        top_candidates: int = 50,
        gap_threshold: float = 0.12,
        min_confidence_ratio: float = 0.5,
        max_results: int = 10,
        eliminate_noise: bool = True,
        is_out_of_domain: bool = False,
        domain_reason: Optional[str] = None
    ) -> Dict[str, Any]:
        """
        Executes multi-modal search using Dot Product similarity and N-Gram context alignment:
        1. Validates domain relevance (rejects out-of-domain non-fashion inputs).
        2. Searches image collection and chunk collection with query_vector.
        3. Merges results per product, extracting S_image and S_text.
        4. Applies category and gender intent filtering.
        5. Computes N-gram context coherence score and blended multi-modal score.
        6. Applies rank gap noise elimination with absolute quality floor.
        """
        # Guardrail: Out of domain input (e.g. dog, animal, vehicle, ball in image mode)
        if is_out_of_domain:
            return {
                "results": [],
                "noise_metadata": {
                    "total_candidates": 0,
                    "kept_count": 0,
                    "eliminated_count": 0,
                    "reason": "out_of_domain",
                    "details": domain_reason or "Non-fashion subject detected"
                },
                "search_mode": search_mode,
                "is_out_of_domain": True,
                "out_of_domain_reason": domain_reason,
                "total_found": 0
            }

        # Guardrail: Check if user text query targets non-fashion entity (e.g. 'ball', 'dog', 'car')
        from .chunking import extract_search_intent
        intent = extract_search_intent(query_text) if (search_mode == "text" and query_text) else {}

        if search_mode == "text" and intent.get("is_non_fashion_query"):
            term = intent.get("non_fashion_term", "non-fashion item")
            return {
                "results": [],
                "noise_metadata": {
                    "total_candidates": 0,
                    "kept_count": 0,
                    "eliminated_count": 0,
                    "reason": "non_fashion_query",
                    "details": f"Query targets non-fashion item '{term}'."
                },
                "search_mode": search_mode,
                "is_out_of_domain": True,
                "out_of_domain_reason": f"No fashion products found for '{term}'. FashionLens only searches clothing, footwear, and accessories.",
                "total_found": 0,
                "intent": intent
            }

        query_list = query_vector.tolist()
        # Retrieve more candidates in text mode to ensure category candidates are retrieved
        effective_limit = max(top_candidates, 120) if search_mode == "text" else top_candidates

        # Step 1: Query both collections
        img_results = self.client.query_points(
            collection_name=COLLECTION_IMAGES,
            query=query_list,
            limit=effective_limit,
            with_payload=True,
            with_vectors=True
        ).points

        chunk_results = self.client.query_points(
            collection_name=COLLECTION_CHUNKS,
            query=query_list,
            limit=effective_limit * 2,
            with_payload=True
        ).points

        # Step 2: Organize candidate products
        candidates: Dict[int, Dict[str, Any]] = {}

        # Process image search results
        for pt in img_results:
            pid = int(pt.id)
            score = float(pt.score)
            payload = pt.payload or {}
            vector = pt.vector

            candidates[pid] = {
                "id": pid,
                "name": payload.get("name", f"Product #{pid}"),
                "display_name": payload.get("productDisplayName", payload.get("name", "")),
                "gender": payload.get("gender", ""),
                "masterCategory": payload.get("masterCategory", ""),
                "subCategory": payload.get("subCategory", ""),
                "articleType": payload.get("articleType", ""),
                "baseColour": payload.get("baseColour", ""),
                "season": payload.get("season", ""),
                "usage": payload.get("usage", ""),
                "description": payload.get("description", ""),
                "image_filename": payload.get("image_filename", f"{pid}.jpg"),
                "image_url": payload.get("image_url", f"/media/images/{pid}.jpg"),
                "image_score": max(0.0, score),
                "text_score": 0.0,
                "best_matching_chunk": "",
                "image_vector": vector
            }

        # Process text chunk results
        for pt in chunk_results:
            payload = pt.payload or {}
            pid = int(payload.get("product_id", 0))
            if not pid:
                continue

            score = float(pt.score)
            chunk_text = payload.get("chunk_text", "")

            if pid not in candidates:
                prod_meta = payload.get("product_metadata", {})
                candidates[pid] = {
                    "id": pid,
                    "name": prod_meta.get("name", f"Product #{pid}"),
                    "display_name": prod_meta.get("productDisplayName", prod_meta.get("name", "")),
                    "gender": prod_meta.get("gender", ""),
                    "masterCategory": prod_meta.get("masterCategory", ""),
                    "subCategory": prod_meta.get("subCategory", ""),
                    "articleType": prod_meta.get("articleType", ""),
                    "baseColour": prod_meta.get("baseColour", ""),
                    "season": prod_meta.get("season", ""),
                    "usage": prod_meta.get("usage", ""),
                    "description": prod_meta.get("description", ""),
                    "image_filename": prod_meta.get("image_filename", f"{pid}.jpg"),
                    "image_url": prod_meta.get("image_url", f"/media/images/{pid}.jpg"),
                    "image_score": 0.0,
                    "text_score": max(0.0, score),
                    "best_matching_chunk": chunk_text,
                    "image_vector": None
                }
            else:
                if score > candidates[pid]["text_score"]:
                    candidates[pid]["text_score"] = max(0.0, score)
                    candidates[pid]["best_matching_chunk"] = chunk_text

        # For candidates retrieved via chunk search that lack image_score, compute dot product if vector exists
        for pid, cand in candidates.items():
            if cand["image_score"] == 0.0 and cand.get("image_vector"):
                cand["image_score"] = max(0.0, float(np.dot(query_vector, np.array(cand["image_vector"]))))

        # Guardrail: Check image similarity floor in image mode
        if search_mode == "image":
            top_img_score = max([c["image_score"] for c in candidates.values()], default=0.0)
            if top_img_score < 0.38:
                return {
                    "results": [],
                    "noise_metadata": {
                        "total_candidates": len(candidates),
                        "kept_count": 0,
                        "eliminated_count": len(candidates),
                        "reason": "low_similarity_floor",
                        "details": f"Best similarity ({top_img_score:.3f}) below required fashion relevance threshold (0.38)."
                    },
                    "search_mode": search_mode,
                    "is_out_of_domain": True,
                    "out_of_domain_reason": f"No relevant fashion products found (top image similarity {top_img_score:.2f} is too low).",
                    "total_found": 0
                }

        # Step 3: Compute weighted blended score with N-gram context coherence
        from .chunking import extract_search_intent
        intent = extract_search_intent(query_text) if (search_mode == "text" and query_text) else {}

        primary_w = 0.75
        secondary_w = 0.25

        ranked_list = []
        for cand in candidates.values():
            ngram_sim = None
            ngram_meta = None
            is_disq = False
            if search_mode == "text" and query_text:
                ngram_sim, ngram_meta = compute_ngram_context_score(query_text, cand)
                if ngram_meta:
                    is_disq = ngram_meta.get("is_disqualified", False)

            blended = blend_scores(
                image_sim=cand["image_score"],
                text_sim=cand["text_score"],
                ngram_sim=ngram_sim,
                search_mode=search_mode,
                primary_weight=primary_w,
                secondary_weight=secondary_w,
                ngram_weight=0.30,
                is_disqualified=is_disq
            )
            cand["score"] = round(blended, 4)
            cand["image_score"] = round(cand["image_score"], 4)
            cand["text_score"] = round(cand["text_score"], 4)
            if ngram_sim is not None:
                cand["ngram_score"] = round(ngram_sim, 4)
                cand["ngram_meta"] = ngram_meta
            cand["primary_weight"] = primary_w
            cand["secondary_weight"] = secondary_w
            cand["search_mode"] = search_mode

            cand.pop("image_vector", None)
            ranked_list.append(cand)

        # Sort descending by blended score
        ranked_list.sort(key=lambda x: x["score"], reverse=True)

        # Step 4: Noise Elimination & Quality Cutoff
        if eliminate_noise and len(ranked_list) > 0:
            filtered_results, noise_meta = eliminate_rank_gap_noise(
                ranked_items=ranked_list,
                gap_threshold=gap_threshold,
                min_confidence_ratio=min_confidence_ratio,
                max_results=max_results,
                search_mode=search_mode
            )
        else:
            # Filter out zero score items
            valid_only = [item for item in ranked_list if item.get("score", 0.0) > 0.0]
            filtered_results = valid_only[:max_results]
            noise_meta = {
                "total_candidates": len(ranked_list),
                "kept_count": len(filtered_results),
                "eliminated_count": max(0, len(ranked_list) - len(filtered_results)),
                "gap_detected": False
            }

        # Format ranks (1-indexed)
        for i, item in enumerate(filtered_results):
            item["rank"] = i + 1

        return {
            "results": filtered_results,
            "noise_metadata": noise_meta,
            "search_mode": search_mode,
            "total_found": len(filtered_results),
            "intent": intent,
            "is_out_of_domain": False
        }

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns count of items in both collections.
        """
        try:
            img_count = self.client.count(collection_name=COLLECTION_IMAGES).count
            chunk_count = self.client.count(collection_name=COLLECTION_CHUNKS).count
        except Exception as e:
            img_count = 0
            chunk_count = 0
            
        return {
            "image_count": img_count,
            "chunk_count": chunk_count,
            "storage_path": self.storage_path
        }


def get_vector_db() -> VectorDB:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDB()
    return _vector_db_instance
