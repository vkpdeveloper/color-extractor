from __future__ import annotations

import argparse
from io import BytesIO

import requests
import torch
import torch.nn.functional as F
import numpy as np
from PIL import Image
from transformers import AutoModelForSemanticSegmentation, SegformerImageProcessor


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
    "topwear": "upper-clothes",
    "shirt": "upper-clothes",
    "blouse": "upper-clothes",
    "upper": "upper-clothes",
    "upper-clothes": "upper-clothes",
    "pants": "pants",
    "bottomwear": "bottomwear",
    "trousers": "pants",
    "jeans": "pants",
    "skirt": "skirt",
    "dress": "dress",
    "hat": "hat",
    "hair": "hair",
    "sunglasses": "sunglasses",
    "belt": "belt",
    "shoe": "shoe",
    "shoes": "shoe",
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


def _normalize_category(category: str) -> list[str]:
    normalized = category.strip().lower()
    canonical = CANONICAL_CATEGORIES.get(normalized)
    if canonical is None:
        raise ValueError(
            f"unknown category '{category}'. Supported categories: {', '.join(sorted(CANONICAL_CATEGORIES))}"
        )

    if canonical == "bottomwear":
        return ["pants", "skirt", "dress"]
    if canonical == "shoe":
        return ["left-shoe", "right-shoe"]
    if canonical == "arm":
        return ["left-arm", "right-arm"]
    if canonical == "leg":
        return ["left-leg", "right-leg"]
    return [canonical]


def _download_image(url: str) -> Image.Image:
    response = requests.get(url, timeout=15)
    response.raise_for_status()
    return Image.open(BytesIO(response.content)).convert("RGB")


def _build_mask(prediction: torch.Tensor, categories: list[str]) -> np.ndarray:
    category_indices = [LABELS.index(category) for category in categories]
    mask = torch.zeros_like(prediction, dtype=torch.bool)
    for index in category_indices:
        mask |= prediction == index
    return mask.cpu().numpy()


def _overlay_mask(image: Image.Image, mask: np.ndarray, alpha: float) -> Image.Image:
    image_arr = np.asarray(image).astype(np.float32)
    mask_bool = mask.astype(bool)
    overlay_color = np.array([0.0, 200.0, 255.0], dtype=np.float32)

    highlighted = image_arr.copy()
    highlighted[mask_bool] = (
        (1.0 - alpha) * image_arr[mask_bool] + alpha * overlay_color
    )
    highlighted[~mask_bool] = image_arr[~mask_bool] * 0.2

    return Image.fromarray(highlighted.clip(0, 255).astype(np.uint8))


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run SegFormer B2 clothes segmentation and export a category mask.",
    )
    parser.add_argument("--image-url", required=True, help="HTTP(S) image URL")
    parser.add_argument("--category", required=True, help="Target category, e.g. tshirt")
    parser.add_argument(
        "--model",
        default="mattmdjaga/segformer_b2_clothes",
        help="Hugging Face model id",
    )
    parser.add_argument(
        "--mask-out",
        required=True,
        help="Output PNG path for the binary mask",
    )
    parser.add_argument(
        "--overlay-out",
        required=True,
        help="Output PNG path for the highlighted overlay",
    )
    parser.add_argument(
        "--alpha",
        type=float,
        default=0.5,
        help="Overlay blend strength (0-1)",
    )
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    categories = _normalize_category(args.category)

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    processor = SegformerImageProcessor.from_pretrained(args.model)
    model = AutoModelForSemanticSegmentation.from_pretrained(args.model).to(device)
    model.eval()

    image = _download_image(args.image_url)
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
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    overlay_img = _overlay_mask(image, mask, alpha=args.alpha)

    mask_img.save(args.mask_out)
    overlay_img.save(args.overlay_out)


if __name__ == "__main__":
    main()
