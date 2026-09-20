"""
Dataset Ingestion and Indexing Pipeline
Loads fashion products from 'paramaggarwal/fashion-product-images-small',
generates semantic chunks, encodes images and text using CLIP in configurable batches,
and indexes into embedded Qdrant.
"""

import os
import csv
import json
import glob
import shutil
from typing import List, Dict, Any, Optional
from PIL import Image
import numpy as np

from .chunking import chunk_product_description
from .clip_service import get_clip_service
from .vector_db import get_vector_db


class DatasetIngestion:
    def __init__(self, data_dir: str = None, media_dir: str = None):
        base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
        self.data_dir = data_dir or os.path.join(base_dir, "data", "fashion-dataset")
        self.media_images_dir = media_dir or os.path.join(base_dir, "media", "images")
        os.makedirs(self.data_dir, exist_ok=True)
        os.makedirs(self.media_images_dir, exist_ok=True)

    def download_or_locate_dataset(self) -> str:
        """
        Downloads and locates the official Kaggle dataset via kagglehub:
        'paramaggarwal/fashion-product-images-small'
        """
        # 1. Fast check if already extracted in kagglehub cache
        cache_dir = os.path.expanduser("~/.cache/kagglehub/datasets/paramaggarwal/fashion-product-images-small")
        if os.path.exists(cache_dir):
            for root, dirs, files in os.walk(cache_dir):
                if "styles.csv" in files and "images" in dirs:
                    print(f"[Ingestion] Found extracted Kaggle dataset in: {root}")
                    return root

        # 2. Otherwise download via kagglehub
        try:
            print("[Ingestion] Downloading official Kaggle dataset via kagglehub: paramaggarwal/fashion-product-images-small...")
            import kagglehub
            path = kagglehub.dataset_download("paramaggarwal/fashion-product-images-small")
            print(f"[Ingestion] Kaggle dataset available at: {path}")

            candidates = [
                path,
                os.path.join(path, "fashion-dataset"),
                os.path.join(path, "myntradataset"),
            ]
            for c in candidates:
                if os.path.exists(os.path.join(c, "styles.csv")) and os.path.exists(os.path.join(c, "images")):
                    print(f"[Ingestion] Located Kaggle styles.csv and images in: {c}")
                    return c
            return path
        except Exception as e:
            print(f"[Ingestion] Note on kagglehub: {e}")

        if os.path.exists(os.path.join(self.data_dir, "styles.csv")):
            print(f"[Ingestion] Using styles.csv in {self.data_dir}")
            return self.data_dir

        return self.data_dir

    def index_dataset(
        self,
        dataset_path: str = None,
        limit: int = 1000,
        batch_size: int = 32,
        clear_existing: bool = True,
        progress_callback = None
    ) -> Dict[str, Any]:
        """
        Indexes products from styles.csv up to `limit` items using batching.
        batch_size: Configurable batch size for CLIP encoding and Qdrant upserts.
        """
        dataset_path = dataset_path or self.download_or_locate_dataset()
        styles_csv_path = os.path.join(dataset_path, "styles.csv")
        images_dir = os.path.join(dataset_path, "images")

        if not os.path.exists(styles_csv_path) or not os.path.exists(images_dir):
            return {
                "success": False,
                "error": f"styles.csv or images/ not found in {dataset_path}"
            }

        vector_db = get_vector_db()
        if clear_existing:
            vector_db.reset_collections()

        clip_service = get_clip_service()

        print(f"[Ingestion] Reading styles.csv from {styles_csv_path}...")
        products = []
        with open(styles_csv_path, mode="r", encoding="utf-8", errors="replace") as f:
            reader = csv.DictReader(f)
            for row in reader:
                pid = row.get("id")
                if not pid:
                    continue
                img_path = os.path.join(images_dir, f"{pid}.jpg")
                if not os.path.exists(img_path):
                    continue

                # Copy/symlink image to media_images_dir for web serving
                dest_img = os.path.join(self.media_images_dir, f"{pid}.jpg")
                if not os.path.exists(dest_img):
                    try:
                        shutil.copyfile(img_path, dest_img)
                    except Exception as e:
                        pass

                # Check for styles json if available
                desc_text = ""
                json_path = os.path.join(dataset_path, "styles", f"{pid}.json")
                if os.path.exists(json_path):
                    try:
                        with open(json_path, "r", encoding="utf-8", errors="ignore") as jf:
                            jdata = json.load(jf)
                            data_info = jdata.get("data", {})
                            desc_info = data_info.get("productDescriptors", {}).get("description", {})
                            if isinstance(desc_info, dict):
                                desc_text = desc_info.get("value", "")
                    except Exception:
                        pass

                row["description"] = desc_text
                row["name"] = row.get("productDisplayName", f"Product #{pid}")
                row["image_path"] = dest_img
                row["image_url"] = f"/media/images/{pid}.jpg"
                
                # Pre-generate chunks
                row["chunks"] = chunk_product_description(row)
                products.append(row)

                if limit and len(products) >= limit:
                    break

        total_products = len(products)
        print(f"[Ingestion] Loaded {total_products} valid products to index.")

        indexed_count = 0
        for i in range(0, total_products, batch_size):
            batch_prods = products[i:i + batch_size]
            
            # Load batch images
            batch_images = []
            for p in batch_prods:
                try:
                    img = Image.open(p["image_path"]).convert("RGB")
                except Exception:
                    img = Image.new("RGB", (224, 224), color="gray")
                batch_images.append(img)

            # 1. Encode images with CLIP in batch
            batch_img_vecs = clip_service.encode_images(batch_images, batch_size=batch_size)

            # 2. Encode all description chunks for batch
            chunk_map = {}
            for p in batch_prods:
                pid = int(p["id"])
                p_chunks = p["chunks"]
                chunk_texts = [c["text"] for c in p_chunks]
                if chunk_texts:
                    chunk_vecs = clip_service.encode_texts(chunk_texts, batch_size=batch_size)
                    chunk_map[pid] = chunk_vecs

            # 3. Upsert into Qdrant
            vector_db.upsert_products_batch(
                products_data=batch_prods,
                image_embeddings=batch_img_vecs,
                chunk_embeddings_map=chunk_map
            )

            indexed_count += len(batch_prods)
            print(f"[Ingestion] Processed {indexed_count}/{total_products} products (Batch size: {batch_size})")

            if progress_callback:
                progress_callback(indexed_count, total_products)

        stats = vector_db.get_stats()
        return {
            "success": True,
            "indexed_count": indexed_count,
            "total_products": total_products,
            "stats": stats
        }
