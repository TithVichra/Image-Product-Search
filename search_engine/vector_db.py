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
COLLECTION_UNIFIED = "fashion_unified_vectors"
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

        if COLLECTION_UNIFIED not in existing:
            self.client.create_collection(
                collection_name=COLLECTION_UNIFIED,
                vectors_config=models.VectorParams(size=VECTOR_DIM, distance=models.Distance.DOT)
            )
            print(f"[VectorDB] Created single unified collection '{COLLECTION_UNIFIED}' with Distance.DOT")

    def reset_collections(self):
        """
        Clears and recreates collections for fresh dataset indexing.
        """
        print("[VectorDB] Resetting collections for fresh Kaggle dataset indexing...")
        for col in [COLLECTION_IMAGES, COLLECTION_CHUNKS, COLLECTION_UNIFIED]:
            try:
                self.client.delete_collection(col)
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

    def upsert_unified_batch(
        self,
        products_data: List[Dict[str, Any]],
        image_embeddings: np.ndarray,
        title_embeddings: np.ndarray
    ):
        """
        Upserts a batch of products into the single unified vector collection:
        COLLECTION_UNIFIED = 'fashion_unified_vectors'.
        Each product produces two vector points in the single vector column/collection:
        1. Product Image vector tagged with embedding_type='image'
        2. Product Title vector tagged with embedding_type='title'
        Both vectors reside in the same collection and dimension space (512, unit normalized).
        """
        points = []
        for i, prod in enumerate(products_data):
            pid = int(prod["id"])
            pname = prod.get("productDisplayName") or prod.get("name", f"Product #{pid}")
            img_url = prod.get("image_url", f"/media/images/{pid}.jpg")

            base_meta = {
                "product_id": pid,
                "name": pname,
                "productDisplayName": pname,
                "image_url": img_url,
                "gender": prod.get("gender", ""),
                "masterCategory": prod.get("masterCategory", ""),
                "subCategory": prod.get("subCategory", ""),
                "articleType": prod.get("articleType", ""),
                "baseColour": prod.get("baseColour", ""),
                "season": prod.get("season", ""),
                "usage": prod.get("usage", ""),
            }

            # 1. Product Image vector entry
            img_vec = image_embeddings[i].tolist()
            img_payload = dict(base_meta)
            img_payload["embedding_type"] = "image"
            points.append(
                models.PointStruct(
                    id=pid * 2,
                    vector=img_vec,
                    payload=img_payload
                )
            )

            # 2. Product Title vector entry
            title_vec = title_embeddings[i].tolist()
            title_payload = dict(base_meta)
            title_payload["embedding_type"] = "title"
            points.append(
                models.PointStruct(
                    id=pid * 2 + 1,
                    vector=title_vec,
                    payload=title_payload
                )
            )

        if points:
            self.client.upsert(collection_name=COLLECTION_UNIFIED, points=points)

    def unified_search(
        self,
        query_vector: np.ndarray,
        search_mode: str = "text",
        top_k: int = 10,
        candidate_pool: int = 100,
        fusion_strategy: str = "weighted_sum",
        weights: Optional[Tuple[float, float]] = None,
        fetch_missing_counterpart: bool = True
    ) -> List[Dict[str, Any]]:
        """
        Implements Task 3 Combining & Reranking Pipeline on the Unified Vector Table:
        1. Encodes user query into shared CLIP latent space (512-d unit vector).
        2. Retrieves candidate matches from the unified collection.
        3. Groups candidate hits by product_id to aggregate modality scores:
           S_image = sim(query, item_image)
           S_title = sim(query, item_title)
        4. Applies a combining/reranking strategy (Weighted Sum / Score Fusion / RRF).
        5. Selects and returns the top 10 unique ranked items.
        """
        query_list = query_vector.tolist()
        pool_size = max(candidate_pool, top_k * 8)

        # Retrieve candidate points from the single unified collection
        hits = self.client.query_points(
            collection_name=COLLECTION_UNIFIED,
            query=query_list,
            limit=pool_size,
            with_payload=True
        ).points

        # Also retrieve from catalog image collection to guarantee full coverage of all indexed products
        img_coll_hits = []
        try:
            img_coll_hits = self.client.query_points(
                collection_name=COLLECTION_IMAGES,
                query=query_list,
                limit=pool_size,
                with_payload=True
            ).points
        except Exception:
            pass

        if not hits and not img_coll_hits:
            return []

        # Group retrieved hits by product_id
        grouped: Dict[int, Dict[str, Any]] = {}
        for rank_idx, pt in enumerate(hits):
            payload = pt.payload or {}
            pid = int(payload.get("product_id", pt.id // 2))
            emb_type = payload.get("embedding_type", "image" if pt.id % 2 == 0 else "title")
            score = float(pt.score)

            if pid not in grouped:
                grouped[pid] = {
                    "product_id": pid,
                    "name": payload.get("name") or payload.get("productDisplayName", f"Product #{pid}"),
                    "image_url": payload.get("image_url", f"/media/images/{pid}.jpg"),
                    "gender": payload.get("gender", ""),
                    "masterCategory": payload.get("masterCategory", ""),
                    "subCategory": payload.get("subCategory", ""),
                    "articleType": payload.get("articleType", ""),
                    "baseColour": payload.get("baseColour", ""),
                    "season": payload.get("season", ""),
                    "usage": payload.get("usage", ""),
                    "image_score": None,
                    "title_score": None,
                    "image_rank": None,
                    "title_rank": None
                }

            if emb_type == "image":
                if grouped[pid]["image_score"] is None or score > grouped[pid]["image_score"]:
                    grouped[pid]["image_score"] = score
                    grouped[pid]["image_rank"] = rank_idx + 1
            elif emb_type == "title":
                if grouped[pid]["title_score"] is None or score > grouped[pid]["title_score"]:
                    grouped[pid]["title_score"] = score
                    grouped[pid]["title_rank"] = rank_idx + 1

        # Incorporate hits from image collection
        for rank_idx, pt in enumerate(img_coll_hits):
            payload = pt.payload or {}
            pid = int(pt.id)
            score = float(pt.score)

            if pid not in grouped:
                grouped[pid] = {
                    "product_id": pid,
                    "name": payload.get("name") or payload.get("productDisplayName", f"Product #{pid}"),
                    "image_url": payload.get("image_url", f"/media/images/{pid}.jpg"),
                    "gender": payload.get("gender", ""),
                    "masterCategory": payload.get("masterCategory", ""),
                    "subCategory": payload.get("subCategory", ""),
                    "articleType": payload.get("articleType", ""),
                    "baseColour": payload.get("baseColour", ""),
                    "season": payload.get("season", ""),
                    "usage": payload.get("usage", ""),
                    "image_score": score,
                    "title_score": None,
                    "image_rank": rank_idx + 1,
                    "title_rank": None
                }
            else:
                if grouped[pid]["image_score"] is None or score > grouped[pid]["image_score"]:
                    grouped[pid]["image_score"] = score
                    if grouped[pid]["image_rank"] is None:
                        grouped[pid]["image_rank"] = rank_idx + 1

        # Optionally retrieve the counterpart vector for products that only appeared in one modality
        if fetch_missing_counterpart:
            missing_ids = []
            for pid, cand in grouped.items():
                if cand["image_score"] is None:
                    missing_ids.append(pid * 2)
                if cand["title_score"] is None:
                    missing_ids.append(pid * 2 + 1)

            if missing_ids:
                try:
                    counterparts = self.client.retrieve(
                        collection_name=COLLECTION_UNIFIED,
                        ids=missing_ids,
                        with_vectors=True
                    )
                    for cpt in counterparts:
                        if cpt.vector is not None:
                            c_vec = np.array(cpt.vector, dtype=np.float32)
                            cos_sim = float(np.dot(query_vector, c_vec))
                            c_pid = int(cpt.payload.get("product_id", cpt.id // 2))
                            c_type = cpt.payload.get("embedding_type", "image" if cpt.id % 2 == 0 else "title")
                            if c_pid in grouped:
                                if c_type == "image" and grouped[c_pid]["image_score"] is None:
                                    grouped[c_pid]["image_score"] = cos_sim
                                elif c_type == "title" and grouped[c_pid]["title_score"] is None:
                                    grouped[c_pid]["title_score"] = cos_sim
                except Exception as e:
                    print(f"[VectorDB] Counterpart lookup note: {e}")

        # Fill default scores if any remain unpopulated
        for cand in grouped.values():
            if cand["image_score"] is None:
                cand["image_score"] = 0.0
            if cand["title_score"] is None:
                cand["title_score"] = 0.0

        # Weights configuration
        if weights is None:
            if search_mode == "image":
                w_img, w_title = 0.85, 0.15
            else:
                w_img, w_title = 0.25, 0.75
        else:
            w_img, w_title = weights

        # Apply combining strategy
        ranked_products = []
        for cand in grouped.values():
            s_img = cand["image_score"]
            s_title = cand["title_score"]

            if fusion_strategy == "rrf":
                # Reciprocal Rank Fusion (k=60)
                r_img = cand["image_rank"] if cand["image_rank"] is not None else (pool_size + 1)
                r_title = cand["title_rank"] if cand["title_rank"] is not None else (pool_size + 1)
                final_score = (1.0 / (60.0 + r_img)) + (1.0 / (60.0 + r_title))
            elif fusion_strategy == "score_fusion":
                # Average score fusion
                final_score = 0.5 * (s_img + s_title)
            else:
                # Weighted sum fusion (default)
                if search_mode == "image" and s_img >= 0.95:
                    # Near-identical or exact catalog image match prioritizes visual fidelity
                    final_score = s_img if s_title == 0.0 else (0.95 * s_img + 0.05 * s_title)
                else:
                    final_score = (w_img * s_img) + (w_title * s_title)

            ranked_products.append({
                "product_id": cand["product_id"],
                "id": cand["product_id"],
                "name": cand["name"],
                "display_name": cand["name"],
                "image_url": cand["image_url"],
                "final_score": round(float(final_score), 4),
                "score": round(float(final_score), 4),
                "breakdown": {
                    "image_score": round(float(s_img), 4),
                    "title_score": round(float(s_title), 4)
                },
                "image_score": round(float(s_img), 4),
                "text_score": round(float(s_title), 4),
                "gender": cand["gender"],
                "masterCategory": cand["masterCategory"],
                "subCategory": cand["subCategory"],
                "articleType": cand["articleType"],
                "baseColour": cand["baseColour"],
                "season": cand["season"],
                "usage": cand["usage"]
            })

        # Sort descending by final_score
        ranked_products.sort(key=lambda x: x["final_score"], reverse=True)

        # Deliver the top_k unique products with 1-based rank
        final_top = ranked_products[:top_k]
        for idx, item in enumerate(final_top):
            item["rank"] = idx + 1

        return final_top

    def get_stats(self) -> Dict[str, Any]:
        """
        Returns count of items in all collections.
        """
        try:
            img_count = self.client.count(collection_name=COLLECTION_IMAGES).count
        except Exception:
            img_count = 0
        try:
            chunk_count = self.client.count(collection_name=COLLECTION_CHUNKS).count
        except Exception:
            chunk_count = 0
        try:
            unified_count = self.client.count(collection_name=COLLECTION_UNIFIED).count
        except Exception:
            unified_count = 0
            
        return {
            "image_count": img_count,
            "chunk_count": chunk_count,
            "unified_count": unified_count,
            "storage_path": self.storage_path
        }


def get_vector_db() -> VectorDB:
    global _vector_db_instance
    if _vector_db_instance is None:
        _vector_db_instance = VectorDB()
    return _vector_db_instance

