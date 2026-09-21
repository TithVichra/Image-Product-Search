"""
CLIP Embedding Service
Provides unified multimodal text and image embeddings using OpenAI CLIP (ViT-B/32).
All embeddings are L2-normalized so that Dot Product == Cosine Similarity.
"""

import os
from typing import List, Union, Dict, Any, Tuple, Optional
import numpy as np
from PIL import Image
import torch
from transformers import CLIPProcessor, CLIPModel

# Default model
DEFAULT_CLIP_MODEL = "openai/clip-vit-base-patch32"

_clip_instance = None


class CLIPService:
    def __init__(self, model_name: str = DEFAULT_CLIP_MODEL, device: str = None):
        if device is None:
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
            
        print(f"[CLIPService] Loading CLIP model '{model_name}' on device '{self.device}'...")
        self.model = CLIPModel.from_pretrained(model_name).to(self.device)
        self.processor = CLIPProcessor.from_pretrained(model_name)
        self.model.eval()
        print("[CLIPService] CLIP model loaded successfully.")

    def encode_texts(self, texts: List[str], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of text strings into L2-normalized 512-dimensional vectors.
        """
        if not texts:
            return np.empty((0, 512), dtype=np.float32)

        all_features = []
        for i in range(0, len(texts), batch_size):
            batch_texts = texts[i:i + batch_size]
            # Handle empty/whitespace text safely
            clean_batch = [t if t and t.strip() else "fashion product" for t in batch_texts]
            
            inputs = self.processor(
                text=clean_batch,
                return_tensors="pt",
                padding=True,
                truncation=True,
                max_length=77
            ).to(self.device)

            with torch.no_grad():
                features = self.model.get_text_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                # L2 normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy().astype(np.float32))

        return np.vstack(all_features)

    def encode_text_single(self, text: str) -> np.ndarray:
        """
        Encodes a single text query into a 1D normalized float32 vector.
        """
        features = self.encode_texts([text], batch_size=1)
        return features[0]

    def encode_query_with_context(self, query: str) -> np.ndarray:
        """
        Encodes a user text search query using contextual prompt ensembling.
        Applies OpenAI CLIP prompt engineering templates and fashion synonym context
        to stabilize word meaning and eliminate isolated-token variance.
        """
        clean_q = query.strip()
        if not clean_q:
            return self.encode_text_single("fashion item")

        # Base prompt ensemble templates
        prompts = [
            clean_q,
            f"a photo of {clean_q}",
            f"a close-up photo of {clean_q}",
            f"{clean_q}, stylish fashion apparel product",
        ]

        # Domain contextual expansion using fashion lexicon
        q_lower = clean_q.lower()
        try:
            from search_engine.chunking import find_matching_lexicon_entry
            lex = find_matching_lexicon_entry(q_lower)
            if lex:
                syns = [s for s in lex.get("synonyms", []) if s not in q_lower][:2]
                if syns:
                    prompts.append(f"a photo of {clean_q}, also known as {', '.join(syns)}")
                if lex.get("context"):
                    prompts.append(f"{clean_q}, {lex['context']}")
        except Exception:
            pass

        inputs = self.processor(
            text=prompts,
            return_tensors="pt",
            padding=True,
            truncation=True,
            max_length=77
        ).to(self.device)

        with torch.no_grad():
            features = self.model.get_text_features(**inputs)
            if hasattr(features, "pooler_output"):
                features = features.pooler_output
            # L2 normalize individual prompt features
            features = features / features.norm(p=2, dim=-1, keepdim=True)
            # Compute mean vector across prompt ensemble
            ensemble_vec = torch.mean(features, dim=0, keepdim=True)
            # Re-normalize to unit sphere
            ensemble_vec = ensemble_vec / ensemble_vec.norm(p=2, dim=-1, keepdim=True)

        return ensemble_vec.cpu().numpy()[0].astype(np.float32)

    def encode_images(self, images: List[Union[Image.Image, str]], batch_size: int = 32) -> np.ndarray:
        """
        Encodes a list of PIL Images or image file paths into L2-normalized 512-dimensional vectors.
        """
        if not images:
            return np.empty((0, 512), dtype=np.float32)

        all_features = []
        for i in range(0, len(images), batch_size):
            batch_items = images[i:i + batch_size]
            pil_batch = []
            for item in batch_items:
                if isinstance(item, str):
                    try:
                        img = Image.open(item).convert("RGB")
                    except Exception as e:
                        print(f"[CLIPService] Error loading image {item}: {e}")
                        img = Image.new("RGB", (224, 224), color="gray")
                elif isinstance(item, Image.Image):
                    img = item.convert("RGB")
                else:
                    img = Image.new("RGB", (224, 224), color="gray")
                pil_batch.append(img)

            inputs = self.processor(
                images=pil_batch,
                return_tensors="pt"
            ).to(self.device)

            with torch.no_grad():
                features = self.model.get_image_features(**inputs)
                if hasattr(features, "pooler_output"):
                    features = features.pooler_output
                # L2 normalize
                features = features / features.norm(p=2, dim=-1, keepdim=True)
                all_features.append(features.cpu().numpy().astype(np.float32))

        return np.vstack(all_features)

    def encode_image_single(self, image: Union[Image.Image, str]) -> np.ndarray:
        """
        Encodes a single PIL Image or image file path into a 1D normalized float32 vector.
        """
        features = self.encode_images([image], batch_size=1)
        return features[0]

    def classify_domain(self, image: Union[Image.Image, str]) -> Dict[str, Any]:
        """
        Evaluates whether an image belongs to the fashion domain or is out-of-domain
        (e.g., dog, cat, animal, vehicle, food, landscape) using CLIP zero-shot classification.
        """
        domain_labels = [
            ("fashion", "a photo of wearable fashion clothing, apparel, shirts, t-shirts, tops, dresses, skirts, jeans, trousers, shorts, jackets, sweaters, shoes, sports shoes, sneakers, sandals, watches, bags, backpacks, wallets, sunglasses, or accessories"),
            ("sports_ball", "a photo of a sports ball, soccer ball, football, basketball, baseball, tennis ball, or golf ball"),
            ("animal", "a photo of an animal, pet, dog, cat, puppy, kitten, bird, wildlife, or mammal"),
            ("vehicle", "a photo of a vehicle, car, motorcycle, truck, bus, or bicycle"),
            ("food", "a photo of food, cooked meal, burger, pizza, salad, fruit, or edible dish"),
            ("other", "a photo of household furniture, electronics, landscape, or room interior")
        ]

        # Reset cache if labels changed
        if not hasattr(self, "_domain_text_vectors") or self._domain_text_vectors is None or len(self._domain_text_vectors) != len(domain_labels):
            texts = [label_desc for _, label_desc in domain_labels]
            self._domain_text_vectors = self.encode_texts(texts)

        img_vec = self.encode_image_single(image)
        # Dot products with L2 normalized vectors
        sims = np.dot(self._domain_text_vectors, img_vec)
        
        # Softmax temperature scaling
        temperature = 0.05
        exp_sims = np.exp(sims / temperature)
        probs = exp_sims / np.sum(exp_sims)

        domain_probs = {domain_labels[i][0]: float(probs[i]) for i in range(len(domain_labels))}
        top_idx = int(np.argmax(probs))
        top_domain = domain_labels[top_idx][0]
        top_prob = float(probs[top_idx])
        fashion_prob = domain_probs["fashion"]

        # Only out-of-domain if non-fashion class dominates with high confidence
        is_out_of_domain = False
        out_of_domain_reason = None

        if top_domain in ["sports_ball", "animal", "vehicle", "food"] and top_prob >= 0.50 and fashion_prob < 0.25:
            is_out_of_domain = True
            domain_name = top_domain.replace("_", " ")
            out_of_domain_reason = f"Image classified as {domain_name} ({top_prob*100:.1f}%) rather than fashion product ({fashion_prob*100:.1f}%)"

        return {
            "is_out_of_domain": is_out_of_domain,
            "top_domain": top_domain,
            "top_confidence": round(top_prob, 3),
            "fashion_probability": round(fashion_prob, 3),
            "domain_probs": domain_probs,
            "domain_probabilities": {k: round(v, 3) for k, v in domain_probs.items()},
            "out_of_domain_reason": out_of_domain_reason
        }


def get_clip_service() -> CLIPService:
    global _clip_instance
    if _clip_instance is None:
        _clip_instance = CLIPService()
    return _clip_instance

