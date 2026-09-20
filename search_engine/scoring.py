"""
Multi-Modal Dual Scoring, N-Gram Context Coherence, and Rank Gap Noise Elimination Module
"""

import re
import numpy as np
from typing import List, Dict, Any, Tuple, Optional


def clean_text(text: str) -> str:
    if not text:
        return ""
    text = re.sub(r"<[^>]+>", " ", str(text))
    text = re.sub(r"[^\w\s\-']", " ", text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()


def extract_word_tokens(text: str) -> List[str]:
    cleaned = clean_text(text).lower()
    return [w for w in cleaned.split() if len(w) > 1]


def extract_ngrams(tokens: List[str], n: int) -> List[str]:
    if len(tokens) < n or n < 1:
        return []
    return [" ".join(tokens[i:i + n]) for i in range(len(tokens) - n + 1)]


def compute_dot_product(vec_a: np.ndarray, vec_b: np.ndarray) -> float:
    """
    Computes dot product between two vectors.
    If vectors are L2-normalized, dot product is identical to cosine similarity.
    """
    return float(np.dot(vec_a, vec_b))


def calibrate_cross_modal(score: float) -> float:
    """
    CLIP cross-modal similarities (e.g. text-to-image dot products) naturally fall
    in the 0.12 - 0.35 range due to cross-space projection spread.
    Calibrates this score into an intuitive 0.0 - 1.0 confidence distribution.
    """
    if score <= 0.12:
        return max(0.0, score * 2.0)
    # Map [0.12, 0.34] to [0.24, 0.95]
    scaled = 0.24 + ((score - 0.12) / (0.34 - 0.12)) * 0.71
    return float(np.clip(scaled, 0.0, 1.0))


def compute_ngram_context_score(
    query: str,
    product_meta: Dict[str, Any]
) -> Tuple[float, Dict[str, Any]]:
    """
    Evaluates N-gram lexical overlap, phrase structure, and semantic category context
    between user search query and candidate product metadata.

    Returns:
      (score: float in [0.0, 1.0], details: dict)
    """
    from search_engine.chunking import (
        clean_text, extract_word_tokens, extract_normalized_token_set,
        normalize_stem, extract_search_intent, FASHION_LEXICON,
        INCOMPATIBLE_CATEGORIES, COLOR_FAMILIES
    )

    q_clean = clean_text(query).lower()
    q_tokens = extract_word_tokens(q_clean)
    if not q_tokens:
        return 0.5, {"match_type": "empty_query", "is_disqualified": False}

    # Extract candidate text fields
    name = clean_text(product_meta.get("display_name") or product_meta.get("name") or "").lower()
    article = clean_text(product_meta.get("articleType") or "").lower()
    sub_cat = clean_text(product_meta.get("subCategory") or "").lower()
    master_cat = clean_text(product_meta.get("masterCategory") or "").lower()
    color = clean_text(product_meta.get("baseColour") or "").lower()
    usage = clean_text(product_meta.get("usage") or "").lower()
    chunk_text = clean_text(product_meta.get("best_matching_chunk") or "").lower()

    # Aggregate candidate representation
    cand_full_text = f"{name} {article} {sub_cat} {master_cat} {color} {usage} {chunk_text}".lower()
    cand_tokens = extract_normalized_token_set(cand_full_text)

    # Candidate N-grams
    cand_name_tokens = extract_word_tokens(name)
    cand_chunk_tokens = extract_word_tokens(chunk_text)
    cand_bigrams = set(
        extract_ngrams(cand_name_tokens, 2) +
        extract_ngrams(cand_chunk_tokens, 2) +
        [f"{color} {article}", f"{usage} {article}"]
    )
    cand_trigrams = set(
        extract_ngrams(cand_name_tokens, 3) +
        extract_ngrams(cand_chunk_tokens, 3)
    )

    # 1. Unigram Recall & Precision with stem normalization
    q_unigrams = extract_normalized_token_set(q_clean)
    matched_unigrams = q_unigrams.intersection(cand_tokens)
    num_query_terms = max(1, len(q_tokens))
    unigram_recall = min(1.0, len(matched_unigrams) / num_query_terms)

    # 2. Bigram Matching
    q_bigrams = extract_ngrams(q_tokens, 2)
    if q_bigrams:
        matched_bigrams = [
            bg for bg in q_bigrams 
            if bg in cand_bigrams or bg in cand_full_text or
            f"{normalize_stem(bg.split()[0])} {normalize_stem(bg.split()[1])}" in cand_full_text
        ]
        bigram_score = len(matched_bigrams) / len(q_bigrams)
    else:
        # If single-word query, bigram defaults to unigram recall
        bigram_score = unigram_recall

    # 3. Trigram Matching
    q_trigrams = extract_ngrams(q_tokens, 3)
    if q_trigrams:
        matched_trigrams = [tg for tg in q_trigrams if tg in cand_trigrams or tg in cand_full_text]
        trigram_score = len(matched_trigrams) / len(q_trigrams)
    else:
        trigram_score = bigram_score

    # 4. Exact Phrase Match Bonus (requires whole-word boundary or exact unigram match)
    exact_match = 1.0 if re.search(r"\b" + re.escape(q_clean) + r"\b", cand_full_text) else 0.0

    # 5. Intent Extraction: Category, Gender & Color Semantic Alignment
    intent = extract_search_intent(query)
    target_category = intent.get("target_category")
    target_gender = intent.get("target_gender")
    target_color = intent.get("target_color")
    color_shades = intent.get("color_shades", set())
    cand_gender = product_meta.get("gender", "").strip()
    cand_color = product_meta.get("baseColour", "").strip().lower()

    # --- Gender Compatibility Check ---
    gender_multiplier = 1.0
    gender_mismatch = False

    if target_gender == "Women":
        if cand_gender in ["Men", "Boys"]:
            gender_multiplier = 0.0
            gender_mismatch = True
        elif cand_gender in ["Women", "Girls"]:
            gender_multiplier = 1.10
    elif target_gender == "Men":
        if cand_gender in ["Women", "Girls"]:
            gender_multiplier = 0.0
            gender_mismatch = True
        elif cand_gender in ["Men", "Boys"]:
            gender_multiplier = 1.10

    # --- Color Compatibility Check ---
    color_multiplier = 1.0
    color_mismatch = False

    if target_color:
        is_color_match = (
            cand_color in color_shades or
            bool(re.search(r"\b" + re.escape(target_color) + r"\b", name)) or
            bool(re.search(r"\b" + re.escape(target_color) + r"\b", chunk_text))
        )

        if is_color_match:
            color_multiplier = 1.35
        else:
            # Check if product has an explicit conflicting color
            is_conflicting_color = False
            for other_color, other_shades in COLOR_FAMILIES.items():
                if other_color != target_color and cand_color in other_shades:
                    is_conflicting_color = True
                    break

            if is_conflicting_color:
                color_multiplier = 0.0
                color_mismatch = True
            else:
                color_multiplier = 0.25

    # --- Category Compatibility Check ---
    category_multiplier = 1.0
    category_mismatch = False
    incompatible_found = None

    if target_category:
        target_info = FASHION_LEXICON.get(target_category, {})
        valid_terms = set(
            [target_category, normalize_stem(target_category)] +
            target_info.get("synonyms", []) +
            [normalize_stem(s) for s in target_info.get("synonyms", [])]
        )
        incompatible_categories = INCOMPATIBLE_CATEGORIES.get(target_category, set())

        # Candidate tokens
        product_terms = extract_normalized_token_set(f"{article} {sub_cat} {master_cat} {name}")
        product_cat_terms = extract_normalized_token_set(f"{article} {sub_cat}")

        # Check if product belongs to an incompatible category (primarily check structured category fields)
        for incomp_key in incompatible_categories:
            incomp_info = FASHION_LEXICON.get(incomp_key, {})
            incomp_terms = set(
                [incomp_key, normalize_stem(incomp_key)] +
                incomp_info.get("synonyms", []) +
                [normalize_stem(s) for s in incomp_info.get("synonyms", [])]
            )
            if incomp_terms.intersection(product_cat_terms):
                if not valid_terms.intersection(product_cat_terms):
                    incompatible_found = incomp_key
                    category_multiplier = 0.0
                    category_mismatch = True
                    break

        if not category_mismatch:
            # MasterCategory integrity
            apparel_categories = {
                "dresses", "tshirts", "shirts", "jeans", "trousers", "shorts", "skirts",
                "jackets", "kurtas", "sarees", "tops", "sweatshirts", "sweaters",
                "jumpsuits", "leggings", "nightwear", "swimwear", "bra", "briefs"
            }
            footwear_categories = {
                "shoes", "sports shoes", "casual shoes", "formal shoes",
                "heels", "flats", "sandals", "flip flops"
            }
            accessory_categories = {
                "watches", "sunglasses", "handbags", "backpacks", "wallets",
                "belts", "clutches", "caps", "ties", "earrings", "necklaces",
                "bracelets", "rings", "cufflinks", "scarves", "dupattas", "socks"
            }
            personal_care_categories = {"perfumes", "deodorants", "lipstick", "nail polish"}

            if target_category in apparel_categories and master_cat and master_cat != "apparel":
                category_multiplier = 0.0
                category_mismatch = True
            elif target_category in footwear_categories and master_cat and master_cat != "footwear":
                category_multiplier = 0.0
                category_mismatch = True
            elif target_category in accessory_categories and master_cat and master_cat not in ["accessories", "bags"]:
                category_multiplier = 0.0
                category_mismatch = True
            elif target_category in personal_care_categories and master_cat and master_cat not in ["personal care", "fragrance", "lips", "nails", "makeup"]:
                category_multiplier = 0.0
                category_mismatch = True

            if not category_mismatch:
                # Special check for dresses (must not match tops/shirts unless title/article clearly indicates dress)
                if target_category == "dresses":
                    is_dress = ("dress" in article or "dress" in sub_cat or "dresses" in name or "gown" in name or "sundress" in name or "frock" in name)
                    if not is_dress:
                        category_multiplier = 0.0
                        category_mismatch = True
                    else:
                        category_multiplier = 1.30
                elif valid_terms.intersection(product_terms):
                    category_multiplier = 1.25
                else:
                    category_multiplier = 0.60

    is_disqualified = bool(gender_mismatch or category_mismatch or color_mismatch)

    # If hard mismatch on gender, category, or color, return 0.0 score with flag
    if is_disqualified:
        return 0.0, {
            "unigram_recall": round(unigram_recall, 3),
            "category_target": target_category,
            "target_gender": target_gender,
            "target_color": target_color,
            "cand_gender": cand_gender,
            "cand_color": cand_color,
            "gender_mismatch": gender_mismatch,
            "category_mismatch": category_mismatch,
            "color_mismatch": color_mismatch,
            "incompatible_found": incompatible_found,
            "is_disqualified": True,
            "context_multiplier": 0.0
        }

    # Compute raw N-gram score
    raw_ngram = (
        0.35 * unigram_recall +
        0.35 * bigram_score +
        0.15 * trigram_score +
        0.15 * exact_match
    )

    combined_multiplier = gender_multiplier * category_multiplier * color_multiplier
    final_ngram_score = float(np.clip(raw_ngram * combined_multiplier, 0.0, 1.0))

    details = {
        "unigram_recall": round(unigram_recall, 3),
        "bigram_score": round(bigram_score, 3),
        "trigram_score": round(trigram_score, 3),
        "exact_match": bool(exact_match),
        "category_target": target_category,
        "target_gender": target_gender,
        "target_color": target_color,
        "cand_gender": cand_gender,
        "cand_color": cand_color,
        "is_disqualified": False,
        "context_multiplier": round(combined_multiplier, 2)
    }

    return final_ngram_score, details


def blend_scores(
    image_sim: float,
    text_sim: float,
    ngram_sim: Optional[float] = None,
    search_mode: str = "text",
    primary_weight: float = 0.75,
    secondary_weight: float = 0.25,
    ngram_weight: float = 0.30,
    is_disqualified: bool = False
) -> float:
    """
    Computes the calibrated blended multi-modal score with N-gram contextual coherence:
    - In 'image' mode: 75% image-image similarity + 25% calibrated image-text similarity.
    - In 'text' mode:
      If is_disqualified is True (hard category/gender contradiction), return 0.0.
      Otherwise:
      Vector score = 75% text-text similarity + 25% calibrated text-image similarity.
      Final = (1 - ngram_weight) * Vector score + ngram_weight * effective_ngram.
    """
    if search_mode == "image":
        calibrated_secondary = calibrate_cross_modal(text_sim)
        return primary_weight * image_sim + secondary_weight * calibrated_secondary
    else:
        # If hard category, gender, or color contradiction occurred, disqualify item
        if is_disqualified:
            return 0.0

        calibrated_secondary = calibrate_cross_modal(image_sim)
        vec_score = primary_weight * text_sim + secondary_weight * calibrated_secondary

        if ngram_sim is not None:
            # If ngram_sim has positive lexical score, blend it; if 0 (e.g. synonym without exact lexical overlap),
            # rely on calibrated vec_score rather than arbitrarily zeroing out
            effective_ngram = ngram_sim if ngram_sim > 0.0 else vec_score
            blended = (1.0 - ngram_weight) * vec_score + ngram_weight * effective_ngram
            return float(np.clip(blended, 0.0, 1.0))
        return vec_score


def eliminate_rank_gap_noise(
    ranked_items: List[Dict[str, Any]],
    gap_threshold: float = 0.12,
    min_confidence_ratio: float = 0.50,
    max_results: int = 10,
    search_mode: str = "text"
) -> Tuple[List[Dict[str, Any]], Dict[str, Any]]:
    """
    Eliminates noise when there is a significant score gap (cliff) between adjacent ranks,
    when scores fall below the adaptive floor, or when items fail domain/category validity (score <= 0.0).
    """
    if not ranked_items:
        return [], {"eliminated_count": 0, "reason": "empty_input"}

    # Filter out any disqualified items (score <= 0.0)
    valid_items = [item for item in ranked_items if item.get("score", 0.0) > 0.0]

    # Absolute floor: in image mode, min 0.38; in text mode, min 0.28
    abs_floor = 0.38 if search_mode == "image" else 0.28
    valid_items = [item for item in valid_items if item.get("score", 0.0) >= abs_floor]

    if not valid_items:
        return [], {
            "total_candidates": len(ranked_items),
            "kept_count": 0,
            "eliminated_count": len(ranked_items),
            "reason": "all_below_quality_or_incompatible"
        }

    top_score = valid_items[0].get("score", 0.0)
    cutoff_index = len(valid_items)
    gap_found = False
    gap_rank = None
    gap_size = 0.0

    min_score_allowed = max(abs_floor, top_score * min_confidence_ratio)

    for i in range(len(valid_items) - 1):
        curr_score = valid_items[i].get("score", 0.0)
        next_score = valid_items[i + 1].get("score", 0.0)
        diff = curr_score - next_score

        if diff >= gap_threshold and i >= 1:
            cutoff_index = i + 1
            gap_found = True
            gap_rank = i + 1
            gap_size = diff
            break

        if next_score < min_score_allowed and i >= 2:
            cutoff_index = i + 1
            break

    effective_cutoff = min(cutoff_index, max_results)
    filtered = valid_items[:effective_cutoff]
    eliminated_count = len(ranked_items) - len(filtered)

    noise_meta = {
        "total_candidates": len(ranked_items),
        "kept_count": len(filtered),
        "eliminated_count": eliminated_count,
        "gap_detected": gap_found,
        "gap_rank": gap_rank,
        "gap_size": round(gap_size, 4),
        "top_score": round(top_score, 4)
    }

    return filtered, noise_meta
