from __future__ import annotations

from dataclasses import dataclass, field
from typing import Protocol

import numpy as np
from PIL import Image
import torch
import torch.nn.functional as F
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor

from v2.nlp_utils import get_processor
from skimage import color as skcolor
from skimage import measure
from skimage.morphology import (
    closing,
    disk,
    opening,
    remove_small_holes,
    remove_small_objects,
)


class MaskProvider(Protocol):
    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        """Return a foreground mask where True indicates garment pixels."""


@dataclass
class HeuristicMaskProvider:
    max_side: int = 512
    corner_lab_threshold: float = 18.0
    min_coverage: float = 0.20
    max_coverage: float = 0.98
    center_fallback_ratio: float = 0.72
    center_focus_ratio: float = 0.50
    min_component_area_ratio: float = 0.002
    use_skin_mask: bool = False

    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3)")

        original_h, original_w = image_rgb.shape[:2]
        work = self._resize_for_processing(image_rgb)

        lab = skcolor.rgb2lab(work / 255.0)
        candidate_background = self._corner_background_mask(lab)
        connected_background = self._edge_connected_mask(candidate_background)
        if self.use_skin_mask:
            skin_mask = self._skin_mask(work)
        else:
            skin_mask = np.zeros(work.shape[:2], dtype=bool)

        foreground = ~(connected_background | skin_mask)
        foreground = opening(foreground, disk(2))
        foreground = closing(foreground, disk(3))
        foreground = remove_small_holes(foreground, max_size=128)
        foreground = remove_small_objects(foreground, max_size=96)
        foreground = self._retain_centered_components(foreground)

        coverage = float(np.mean(foreground))
        if coverage < self.min_coverage or coverage > self.max_coverage:
            foreground = self._center_box_mask(
                work.shape[0], work.shape[1], ratio=self.center_fallback_ratio
            )

        mask = self._resize_mask_to_original(foreground, original_h, original_w)

        if float(np.mean(mask)) < self.min_coverage:
            mask = self._center_box_mask(
                original_h, original_w, ratio=self.center_fallback_ratio
            )

        return mask.astype(bool)

    def _resize_for_processing(self, image_rgb: np.ndarray) -> np.ndarray:
        height, width = image_rgb.shape[:2]
        longest = max(height, width)
        if longest <= self.max_side:
            return image_rgb

        scale = self.max_side / float(longest)
        new_h = max(1, int(round(height * scale)))
        new_w = max(1, int(round(width * scale)))
        resized = Image.fromarray(image_rgb, mode="RGB").resize(
            (new_w, new_h), Image.Resampling.BILINEAR
        )
        return np.asarray(resized, dtype=np.uint8)

    def _corner_background_mask(self, lab: np.ndarray) -> np.ndarray:
        h, w = lab.shape[:2]
        patch_h = max(1, int(h * 0.04))
        patch_w = max(1, int(w * 0.04))

        corners = [
            lab[:patch_h, :patch_w],
            lab[:patch_h, -patch_w:],
            lab[-patch_h:, :patch_w],
            lab[-patch_h:, -patch_w:],
        ]
        corner_colors = np.array(
            [np.mean(corner.reshape(-1, 3), axis=0) for corner in corners]
        )

        flat_lab = lab.reshape(-1, 3)
        distances = np.linalg.norm(
            flat_lab[:, None, :] - corner_colors[None, :, :], axis=2
        )
        min_dist = np.min(distances, axis=1)
        return (min_dist <= self.corner_lab_threshold).reshape(h, w)

    def _edge_connected_mask(self, candidate: np.ndarray) -> np.ndarray:
        labels = measure.label(candidate, connectivity=1)
        if labels.max() == 0:
            return candidate

        edge_labels = set(np.unique(labels[0, :]))
        edge_labels.update(np.unique(labels[-1, :]))
        edge_labels.update(np.unique(labels[:, 0]))
        edge_labels.update(np.unique(labels[:, -1]))
        edge_labels.discard(0)

        if not edge_labels:
            return np.zeros_like(candidate, dtype=bool)

        connected = np.isin(labels, list(edge_labels))
        return connected.astype(bool)

    def _skin_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        hsv = skcolor.rgb2hsv(image_rgb / 255.0)
        h = hsv[..., 0]
        s = hsv[..., 1]
        v = hsv[..., 2]

        skin = (
            ((h <= 0.13) | (h >= 0.95))
            & (s >= 0.15)
            & (s <= 0.75)
            & (v >= 0.20)
            & (v <= 0.98)
        )
        skin = opening(skin, disk(1))
        return skin.astype(bool)

    def _center_box_mask(self, height: int, width: int, ratio: float) -> np.ndarray:
        mask = np.zeros((height, width), dtype=bool)
        box_h = max(1, int(round(height * ratio)))
        box_w = max(1, int(round(width * ratio)))
        top = max(0, (height - box_h) // 2)
        left = max(0, (width - box_w) // 2)
        mask[top : top + box_h, left : left + box_w] = True
        return mask

    def _retain_centered_components(self, foreground: np.ndarray) -> np.ndarray:
        labels = measure.label(foreground, connectivity=2)
        if labels.max() == 0:
            return foreground

        center = self._center_box_mask(
            foreground.shape[0],
            foreground.shape[1],
            ratio=self.center_focus_ratio,
        )

        center_ids = set(np.unique(labels[center]))
        center_ids.discard(0)
        component_ids = np.unique(labels)
        component_ids = component_ids[component_ids != 0]

        areas = {
            int(component_id): int(np.count_nonzero(labels == component_id))
            for component_id in component_ids
        }
        if not areas:
            return foreground

        min_area_pixels = max(
            1, int(round(foreground.size * self.min_component_area_ratio))
        )

        keep_ids: set[int] = {
            component_id
            for component_id in center_ids
            if areas.get(component_id, 0) >= min_area_pixels
        }

        largest_component = max(areas.items(), key=lambda item: item[1])[0]
        keep_ids.add(int(largest_component))

        keep_mask = np.isin(labels, list(keep_ids))
        if np.count_nonzero(keep_mask) == 0:
            return foreground
        return keep_mask

    def _resize_mask_to_original(
        self, mask: np.ndarray, original_h: int, original_w: int
    ) -> np.ndarray:
        if mask.shape == (original_h, original_w):
            return mask.astype(bool)

        image = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
        upscaled = image.resize((original_w, original_h), Image.Resampling.NEAREST)
        return np.asarray(upscaled, dtype=np.uint8) > 0


LABELS = [
    "background",
    "hat",
    "hair",
    "sunglasses",
    "upper-clothes",
    "skirt",
    "pants",
    "dress",
    "belt",
    "left-shoe",
    "right-shoe",
    "face",
    "left-leg",
    "right-leg",
    "left-arm",
    "right-arm",
    "bag",
    "scarf",
]

CANONICAL_CATEGORIES = {
    "tshirt": "upper-clothes",
    "t-shirt": "upper-clothes",
    "tee": "upper-clothes",
    "top": "upper-clothes",
    "tops": "upper-clothes",
    "topwear": "upper-clothes",
    "shirt": "upper-clothes",
    "blouse": "upper-clothes",
    "upper": "upper-clothes",
    "upper-clothes": "upper-clothes",
    "polo": "upper-clothes",
    "hoodie": "upper-clothes",
    "sweater": "upper-clothes",
    "sweatshirt": "upper-clothes",
    "cardigan": "upper-clothes",
    "jacket": "upper-clothes",
    "coat": "upper-clothes",
    "blazer": "upper-clothes",
    "kurta": "upper-clothes",
    "kurti": "upper-clothes",
    "tunic": "upper-clothes",
    "shrug": "upper-clothes",
    "waistcoat": "upper-clothes",
    "pants": "pants",
    "bottomwear": "bottomwear",
    "trousers": "pants",
    "jeans": "pants",
    "shorts": "pants",
    "leggings": "pants",
    "joggers": "pants",
    "trackpants": "pants",
    "chinos": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "jumpsuit": "dress",
    "dungarees": "dress",
    "romper": "dress",
    "saree": "dress",
    "sari": "dress",
    "lehenga": "dress",
    "hat": "hat",
    "hair": "hair",
    "sunglasses": "sunglasses",
    "belt": "belt",
    "shoe": "shoe",
    "shoes": "shoe",
    "sneakers": "shoe",
    "boots": "shoe",
    "sandals": "shoe",
    "heels": "shoe",
    "flats": "shoe",
    "loafers": "shoe",
    "left-shoe": "left-shoe",
    "right-shoe": "right-shoe",
    "bag": "bag",
    "scarf": "scarf",
    "face": "face",
    "arm": "arm",
    "arms": "arm",
    "left-arm": "left-arm",
    "right-arm": "right-arm",
    "leg": "leg",
    "legs": "leg",
    "left-leg": "left-leg",
    "right-leg": "right-leg",
}


def _normalize_categories(categories: list[str]) -> list[str]:
    normalized: list[str] = []
    for category in categories:
        canonical = CANONICAL_CATEGORIES.get(category.strip().lower())
        if canonical is None:
            continue
        if canonical == "bottomwear":
            normalized.extend(["pants", "skirt", "dress"])
        elif canonical == "shoe":
            normalized.extend(["left-shoe", "right-shoe"])
        elif canonical == "arm":
            normalized.extend(["left-arm", "right-arm"])
        elif canonical == "leg":
            normalized.extend(["left-leg", "right-leg"])
        else:
            normalized.append(canonical)
    return list(dict.fromkeys(normalized))


def _build_mask(prediction: torch.Tensor, categories: list[str]) -> np.ndarray:
    category_indices = [
        LABELS.index(category) for category in categories if category in LABELS
    ]
    if not category_indices:
        return np.zeros(prediction.shape, dtype=bool)
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    for index in category_indices:
        mask |= prediction == index
    return mask.cpu().numpy()


@dataclass
class SegformerMaskProvider:
    title: str
    model_id: str = "mattmdjaga/segformer_b2_clothes"
    min_coverage: float = 0.02
    max_coverage: float = 0.98
    device: torch.device | None = None
    processor: SegformerImageProcessor | None = field(default=None, init=False)
    model: AutoModelForSemanticSegmentation | None = field(default=None, init=False)

    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        if image_rgb.ndim != 3 or image_rgb.shape[2] != 3:
            raise ValueError("image_rgb must have shape (H, W, 3)")

        categories = self._extract_categories()
        if not categories:
            return np.zeros(image_rgb.shape[:2], dtype=bool)

        device = self._get_device()
        self._ensure_model(device)
        processor = self.processor
        model = self.model
        if processor is None or model is None:
            raise RuntimeError("SegFormer model failed to initialize")
        image = Image.fromarray(image_rgb, mode="RGB")
        inputs = processor(images=image, return_tensors="pt").to(device)

        with torch.no_grad():
            outputs = model(**inputs)
            logits = outputs.logits
            upsampled_logits = F.interpolate(
                logits,
                size=image.size[::-1],
                mode="bilinear",
                align_corners=False,
            )
            prediction = upsampled_logits.argmax(dim=1)[0]

        mask = _build_mask(prediction, categories)
        return mask.astype(bool)

    def has_reasonable_coverage(self, mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        coverage = float(mask.mean())
        return self.min_coverage <= coverage <= self.max_coverage

    def _get_device(self) -> torch.device:
        if self.device is not None:
            return self.device
        return torch.device("cuda" if torch.cuda.is_available() else "cpu")

    def _ensure_model(self, device: torch.device) -> None:
        if self.processor is None:
            self.processor = SegformerImageProcessor.from_pretrained(self.model_id)
        if self.model is None:
            self.model = AutoModelForSemanticSegmentation.from_pretrained(
                self.model_id
            ).to(device)
            model = self.model
            if model is None:
                raise RuntimeError("SegFormer model failed to load")
            model.eval()

    def _extract_categories(self) -> list[str]:
        processor = get_processor()
        entities = processor.extract_entities(self.title)
        category_terms = entities.get("CATEGORY", []) if entities else []
        return _normalize_categories(category_terms)


@dataclass
class CombinedMaskProvider:
    title: str
    heuristic: HeuristicMaskProvider = field(default_factory=HeuristicMaskProvider)
    segformer: SegformerMaskProvider | None = None
    min_coverage: float = 0.02
    max_coverage: float = 0.98

    def __post_init__(self) -> None:
        if self.segformer is None:
            self.segformer = SegformerMaskProvider(title=self.title)

    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        segformer = self.segformer
        if segformer is None:
            return self.heuristic.get_mask(image_rgb)

        heuristic_mask = self.heuristic.get_mask(image_rgb)
        segformer_mask = segformer.get_mask(image_rgb)
        combined = heuristic_mask & segformer_mask

        if self._has_reasonable_coverage(combined):
            return combined

        if segformer.has_reasonable_coverage(segformer_mask):
            return segformer_mask

        if self._has_reasonable_coverage(heuristic_mask):
            return heuristic_mask

        return heuristic_mask

    def _has_reasonable_coverage(self, mask: np.ndarray) -> bool:
        if mask.size == 0:
            return False
        coverage = float(mask.mean())
        return self.min_coverage <= coverage <= self.max_coverage
