"""
FastAPI Search & Detection Application
Exposes REST endpoints for:
- Text-based product search
- Image-based product search with YOLOv8 object detection & focus-crop
- Interactive object detection preview
- Vector DB status & batch indexing
"""

import os
import io
import uuid
import time
from typing import Optional, List, Dict, Any
from pydantic import BaseModel
from fastapi import FastAPI, UploadFile, File, Form, HTTPException, BackgroundTasks
from fastapi.middleware.cors import CORSMiddleware
from fastapi.staticfiles import StaticFiles
from PIL import Image

fastapi_app = FastAPI(
    title="Fashion Product Multimodal Search API",
    description="CLIP + YOLOv8 + Embedded Qdrant Vector Search",
    version="1.0.0"
)

fastapi_app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
UPLOADS_DIR = os.path.join(MEDIA_DIR, "uploads")
CROPS_DIR = os.path.join(MEDIA_DIR, "crops")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


class TextSearchRequest(BaseModel):
    query: str
    eliminate_noise: bool = True
    gap_threshold: float = 0.12
    min_confidence_ratio: float = 0.50
    max_results: int = 10


class IndexRequest(BaseModel):
    limit: int = 1000
    batch_size: int = 32


@fastapi_app.get("/api/status")
def get_status():
    from search_engine.vector_db import get_vector_db
    import torch
    
    vdb = get_vector_db()
    stats = vdb.get_stats()
    return {
        "status": "online",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vector_stats": stats
    }


class SearchTextContractRequest(BaseModel):
    query: str
    top_k: int = 10


@fastapi_app.post("/search/text")
def search_text_unified_contract(req: SearchTextContractRequest):
    """
    Task 4 Required Endpoint:
    POST /search/text: Accepts {"query": "navy blue sports shoes", "top_k": 10}.
    Delivers top 10 unique items with product_id, name, image_url, final_score, and breakdown.
    """
    clean_q = req.query.strip() if req.query else ""
    if not clean_q:
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    clip_service = get_clip_service()
    vector_db = get_vector_db()

    query_vec = clip_service.encode_text_single(clean_q)
    return vector_db.unified_search(
        query_vector=query_vec,
        search_mode="text",
        top_k=req.top_k,
        candidate_pool=100
    )


@fastapi_app.post("/search/image")
async def search_image_unified_contract(
    file: UploadFile = File(...),
    top_k: int = Form(10)
):
    """
    Task 4 Required Endpoint:
    POST /search/image: Accepts multipart/form-data image file upload.
    Delivers top 10 unique items with product_id, name, image_url, final_score, and breakdown.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"query_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    clip_service = get_clip_service()
    vector_db = get_vector_db()

    query_vec = clip_service.encode_image_single(image)
    return vector_db.unified_search(
        query_vector=query_vec,
        search_mode="image",
        top_k=top_k,
        candidate_pool=100
    )


@fastapi_app.post("/api/search/text")
def search_by_text(req: TextSearchRequest):
    if not req.query or not req.query.strip():
        raise HTTPException(status_code=400, detail="Query string cannot be empty.")

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    t0 = time.time()
    clip_service = get_clip_service()
    vector_db = get_vector_db()

    # 1. Encode text query into normalized CLIP vector
    query_vec = clip_service.encode_text_single(req.query)

    # 2. Search Qdrant with search_mode="text" (75% text score, 25% image score)
    search_res = vector_db.search(
        query_vector=query_vec,
        search_mode="text",
        top_candidates=50,
        gap_threshold=req.gap_threshold,
        min_confidence_ratio=req.min_confidence_ratio,
        max_results=req.max_results,
        eliminate_noise=req.eliminate_noise
    )

    elapsed = round(time.time() - t0, 3)
    search_res["query"] = req.query
    search_res["latency_sec"] = elapsed
    return search_res


@fastapi_app.post("/api/detect")
async def detect_objects(file: UploadFile = File(...)):
    """
    Runs YOLOv8 object detection on an uploaded image, saves a crop preview,
    and returns detected bounding boxes with labels.
    """
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    # Save uploaded query image
    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"upload_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    from search_engine.detector import get_detector
    detector = get_detector()

    det_res = detector.detect_and_crop(image)

    # Save cropped image preview
    crop_filename = f"crop_{file_id}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    det_res["cropped_image"].save(crop_path, "JPEG")

    return {
        "upload_url": f"/media/uploads/{raw_filename}",
        "crop_url": f"/media/crops/{crop_filename}",
        "detected_boxes": det_res["detected_boxes"],
        "crop_applied": det_res["crop_applied"],
        "crop_box": det_res["crop_box"],
        "dimensions": det_res["dimensions"]
    }


@fastapi_app.post("/api/search/image")
async def search_by_image(
    file: UploadFile = File(...),
    box_index: Optional[int] = Form(None),
    use_crop: bool = Form(True),
    eliminate_noise: bool = Form(True),
    gap_threshold: float = Form(0.12),
    max_results: int = Form(10)
):
    if not file.content_type.startswith("image/"):
        raise HTTPException(status_code=400, detail="Uploaded file must be an image.")

    content = await file.read()
    try:
        image = Image.open(io.BytesIO(content)).convert("RGB")
    except Exception as e:
        raise HTTPException(status_code=400, detail=f"Invalid image format: {e}")

    t0 = time.time()
    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"query_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    from search_engine.detector import get_detector
    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    detector = get_detector()
    det_res = detector.detect_and_crop(image, target_box_index=box_index)

    # Save cropped image preview
    crop_filename = f"crop_{file_id}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    det_res["cropped_image"].save(crop_path, "JPEG")

    clip_service = get_clip_service()
    vector_db = get_vector_db()

    # 1. Always encode uncropped original image vector
    orig_vec = clip_service.encode_image_single(image)
    orig_results = vector_db.unified_search(
        query_vector=orig_vec,
        search_mode="image",
        top_k=max_results,
        candidate_pool=100
    )

    # 2. Check if cropping should be tested
    crop_results = None
    if use_crop and det_res["crop_applied"]:
        crop_vec = clip_service.encode_image_single(det_res["cropped_image"])
        crop_results = vector_db.unified_search(
            query_vector=crop_vec,
            search_mode="image",
            top_k=max_results,
            candidate_pool=100
        )

    # Compare uncropped vs cropped results
    orig_top_sim = orig_results[0]["breakdown"]["image_score"] if orig_results else 0.0
    crop_top_sim = crop_results[0]["breakdown"]["image_score"] if crop_results else 0.0

    # If original image has high match similarity (>= 0.65) or beats cropped image, preserve original
    if crop_results is None or orig_top_sim >= 0.65 or orig_top_sim >= crop_top_sim:
        final_results = orig_results
        selected_top_sim = orig_top_sim
    else:
        final_results = crop_results
        selected_top_sim = crop_top_sim

    # 3. Domain validation: An image that matches a product with similarity >= 0.80 is confirmed in-domain!
    is_out_of_domain = False
    domain_reason = None

    if selected_top_sim < 0.80:
        if det_res.get("is_out_of_domain"):
            is_out_of_domain = True
            domain_reason = det_res.get("out_of_domain_reason")
        else:
            domain_check = clip_service.classify_domain(image)
            if domain_check.get("is_out_of_domain"):
                is_out_of_domain = True
                domain_reason = domain_check.get("out_of_domain_reason")

    elapsed = round(time.time() - t0, 3)

    if is_out_of_domain or not final_results:
        return {
            "results": [],
            "noise_metadata": {
                "total_candidates": 0,
                "kept_count": 0,
                "eliminated_count": 0,
                "reason": "out_of_domain" if is_out_of_domain else "no_matches",
                "details": domain_reason or "No matching fashion products found."
            },
            "search_mode": "image",
            "is_out_of_domain": is_out_of_domain,
            "out_of_domain_reason": domain_reason,
            "detected_entity": det_res.get("detected_entity"),
            "total_found": 0,
            "query_image_url": f"/media/uploads/{raw_filename}",
            "crop_image_url": f"/media/crops/{crop_filename}",
            "detected_boxes": det_res["detected_boxes"],
            "crop_applied": det_res["crop_applied"],
            "crop_box": det_res["crop_box"],
            "dimensions": det_res["dimensions"],
            "latency_sec": elapsed
        }

    # 4. Optional rank gap noise elimination
    if eliminate_noise and len(final_results) > 0:
        from search_engine.scoring import eliminate_rank_gap_noise
        filtered_results, noise_meta = eliminate_rank_gap_noise(
            ranked_items=final_results,
            gap_threshold=gap_threshold,
            min_confidence_ratio=0.50,
            max_results=max_results,
            search_mode="image"
        )
        for idx, itm in enumerate(filtered_results):
            itm["rank"] = idx + 1
    else:
        filtered_results = final_results[:max_results]
        noise_meta = {
            "total_candidates": len(final_results),
            "kept_count": len(filtered_results),
            "eliminated_count": 0,
            "gap_detected": False
        }

    return {
        "results": filtered_results,
        "noise_metadata": noise_meta,
        "search_mode": "image",
        "total_found": len(filtered_results),
        "is_out_of_domain": False,
        "query_image_url": f"/media/uploads/{raw_filename}",
        "crop_image_url": f"/media/crops/{crop_filename}",
        "detected_boxes": det_res["detected_boxes"],
        "crop_applied": det_res["crop_applied"],
        "crop_box": det_res["crop_box"],
        "dimensions": det_res["dimensions"],
        "detected_entity": det_res.get("detected_entity"),
        "latency_sec": elapsed
    }


indexing_in_progress = False


@fastapi_app.post("/api/index")
def trigger_indexing(req: IndexRequest, background_tasks: BackgroundTasks):
    global indexing_in_progress
    if indexing_in_progress:
        return {"status": "already_running", "message": "Indexing is already running."}

    def run_index():
        global indexing_in_progress
        indexing_in_progress = True
        try:
            from search_engine.ingestion import DatasetIngestion
            ingestion = DatasetIngestion()
            ingestion.index_dataset(limit=req.limit, batch_size=req.batch_size)
        finally:
            indexing_in_progress = False

    background_tasks.add_task(run_index)
    return {
        "status": "started",
        "message": f"Dataset indexing started for {req.limit} items with batch_size={req.batch_size}."
    }
