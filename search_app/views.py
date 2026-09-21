import os
import io
import time
import uuid
import json
from django.shortcuts import render
from django.http import JsonResponse, HttpResponseBadRequest
from django.views.decorators.csrf import csrf_exempt
from PIL import Image

BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
MEDIA_DIR = os.path.join(BASE_DIR, "media")
UPLOADS_DIR = os.path.join(MEDIA_DIR, "uploads")
CROPS_DIR = os.path.join(MEDIA_DIR, "crops")
os.makedirs(UPLOADS_DIR, exist_ok=True)
os.makedirs(CROPS_DIR, exist_ok=True)


def index(request):
    """
    Renders the main search UI template.
    """
    return render(request, "index.html")


def api_status(request):
    """
    Returns vector database and model status.
    """
    from search_engine.vector_db import get_vector_db
    import torch
    vdb = get_vector_db()
    stats = vdb.get_stats()
    return JsonResponse({
        "status": "online",
        "device": "cuda" if torch.cuda.is_available() else "cpu",
        "vector_stats": stats
    })


@csrf_exempt
def search_text_unified(request):
    """
    Task 4 Endpoint:
    POST /search/text
    Input: {"query": "navy blue sports shoes", "top_k": 10}
    Output: JSON list containing the top 10 unique ranked items.
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        body = json.loads(request.body)
    except Exception:
        body = {}

    query = body.get("query", "").strip()
    if not query:
        return JsonResponse({"detail": "Query string cannot be empty."}, status=400)

    top_k = int(body.get("top_k", 10))

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    clip_service = get_clip_service()
    vector_db = get_vector_db()

    query_vec = clip_service.encode_text_single(query)
    results = vector_db.unified_search(
        query_vector=query_vec,
        search_mode="text",
        top_k=top_k,
        candidate_pool=100
    )
    return JsonResponse(results, safe=False)


@csrf_exempt
def search_image_unified(request):
    """
    Task 4 Endpoint:
    POST /search/image
    Input: multipart/form-data image file upload
    Output: JSON list containing the top 10 unique ranked items.
    """
    if request.method != "POST" or "file" not in request.FILES:
        return HttpResponseBadRequest("Image file upload required")

    file_obj = request.FILES["file"]
    try:
        image = Image.open(file_obj).convert("RGB")
    except Exception as e:
        return JsonResponse({"detail": f"Invalid image: {e}"}, status=400)

    top_k = int(request.POST.get("top_k", 10))

    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"query_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    clip_service = get_clip_service()
    vector_db = get_vector_db()

    query_vec = clip_service.encode_image_single(image)
    results = vector_db.unified_search(
        query_vector=query_vec,
        search_mode="image",
        top_k=top_k,
        candidate_pool=100
    )
    return JsonResponse(results, safe=False)


@csrf_exempt
def api_search_text(request):
    """
    Handles POST /api/search/text
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        body = json.loads(request.body)
    except Exception:
        body = {}

    query = body.get("query", "").strip()
    if not query:
        return JsonResponse({"detail": "Query string cannot be empty."}, status=400)

    eliminate_noise = body.get("eliminate_noise", True)
    gap_threshold = float(body.get("gap_threshold", 0.12))
    min_confidence_ratio = float(body.get("min_confidence_ratio", 0.50))
    max_results = int(body.get("max_results", 10))

    from search_engine.clip_service import get_clip_service
    from search_engine.vector_db import get_vector_db

    t0 = time.time()
    clip_service = get_clip_service()
    vector_db = get_vector_db()

    query_vec = clip_service.encode_query_with_context(query)

    search_res = vector_db.search(
        query_vector=query_vec,
        search_mode="text",
        query_text=query,
        top_candidates=50,
        gap_threshold=gap_threshold,
        min_confidence_ratio=min_confidence_ratio,
        max_results=max_results,
        eliminate_noise=eliminate_noise
    )

    search_res["query"] = query
    search_res["latency_sec"] = round(time.time() - t0, 3)
    return JsonResponse(search_res)


@csrf_exempt
def api_detect(request):
    """
    Handles POST /api/detect
    """
    if request.method != "POST" or "file" not in request.FILES:
        return HttpResponseBadRequest("Image file upload required")

    file_obj = request.FILES["file"]
    try:
        image = Image.open(file_obj).convert("RGB")
    except Exception as e:
        return JsonResponse({"detail": f"Invalid image: {e}"}, status=400)

    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"upload_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    from search_engine.detector import get_detector
    detector = get_detector()
    det_res = detector.detect_and_crop(image)

    crop_filename = f"crop_{file_id}.jpg"
    crop_path = os.path.join(CROPS_DIR, crop_filename)
    det_res["cropped_image"].save(crop_path, "JPEG")

    return JsonResponse({
        "upload_url": f"/media/uploads/{raw_filename}",
        "crop_url": f"/media/crops/{crop_filename}",
        "detected_boxes": det_res["detected_boxes"],
        "crop_applied": det_res["crop_applied"],
        "crop_box": det_res["crop_box"],
        "dimensions": det_res["dimensions"],
        "is_out_of_domain": det_res.get("is_out_of_domain", False),
        "detected_entity": det_res.get("detected_entity"),
        "out_of_domain_reason": det_res.get("out_of_domain_reason")
    })


@csrf_exempt
def api_search_image(request):
    """
    Handles POST /api/search/image
    Executes unified multi-modal search powered by the unified vector table with:
    1. Uncropped original image priority to prevent degradation of exact catalog images.
    2. Adaptive focus-cropping for noisy images.
    3. Overrule out-of-domain rejections when high similarity catalog matches exist.
    """
    if request.method != "POST" or "file" not in request.FILES:
        return HttpResponseBadRequest("Image file upload required")

    file_obj = request.FILES["file"]
    try:
        image = Image.open(file_obj).convert("RGB")
    except Exception as e:
        return JsonResponse({"detail": f"Invalid image: {e}"}, status=400)

    t0 = time.time()
    file_id = str(uuid.uuid4())[:8]
    raw_filename = f"query_{file_id}.jpg"
    raw_path = os.path.join(UPLOADS_DIR, raw_filename)
    image.save(raw_path, "JPEG")

    box_index = request.POST.get("box_index")
    if box_index is not None and box_index != "" and box_index != "null":
        box_index = int(box_index)
    else:
        box_index = None

    use_crop = request.POST.get("use_crop", "true").lower() in ("true", "1")
    eliminate_noise = request.POST.get("eliminate_noise", "true").lower() in ("true", "1")
    gap_threshold = float(request.POST.get("gap_threshold", 0.12))
    max_results = int(request.POST.get("max_results", 10))

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
        # Not an exact catalog match, evaluate out-of-domain detection
        if det_res.get("is_out_of_domain"):
            is_out_of_domain = True
            domain_reason = det_res.get("out_of_domain_reason")
        else:
            domain_check = clip_service.classify_domain(image)
            if domain_check.get("is_out_of_domain"):
                is_out_of_domain = True
                domain_reason = domain_check.get("out_of_domain_reason")

    if is_out_of_domain or not final_results:
        return JsonResponse({
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
            "latency_sec": round(time.time() - t0, 3)
        })

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

    return JsonResponse({
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
        "latency_sec": round(time.time() - t0, 3)
    })


@csrf_exempt
def api_index(request):
    """
    Handles POST /api/index
    """
    if request.method != "POST":
        return HttpResponseBadRequest("POST required")

    try:
        body = json.loads(request.body)
    except Exception:
        body = {}

    limit = int(body.get("limit", 1000))
    batch_size = int(body.get("batch_size", 32))
    clear_existing = bool(body.get("clear_existing", False))

    from search_engine.ingestion import DatasetIngestion
    import threading

    def run_index():
        ing = DatasetIngestion()
        ing.index_dataset(limit=limit, batch_size=batch_size, clear_existing=clear_existing)

    t = threading.Thread(target=run_index, daemon=True)
    t.start()

    return JsonResponse({
        "status": "started",
        "message": f"Dataset indexing started for {limit} items with batch_size={batch_size}."
    })
