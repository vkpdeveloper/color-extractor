from __future__ import annotations

import json
from pathlib import Path
import io
import requests

import numpy as np
from PIL import Image

from .models import ExtractionResult


def read_image_rgb(image_path: str | Path) -> np.ndarray:
    path_str = str(image_path)
    if path_str.startswith(("http://", "https://")):
        response = requests.get(path_str, timeout=10)
        response.raise_for_status()
        image_data = io.BytesIO(response.content)
        with Image.open(image_data) as image:
            rgb = image.convert("RGB")
            return np.asarray(rgb, dtype=np.uint8)

    path = Path(image_path)
    with Image.open(path) as image:
        rgb = image.convert("RGB")
        return np.asarray(rgb, dtype=np.uint8)


def save_mask_image(mask: np.ndarray, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    mask_img = Image.fromarray((mask.astype(np.uint8) * 255), mode="L")
    mask_img.save(path)


def save_masked_image(
    image_rgb: np.ndarray, mask: np.ndarray, output_path: str | Path
) -> None:
    if mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError("mask shape must match image dimensions")

    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    masked = image_rgb.copy()
    masked[~mask] = 0
    masked_img = Image.fromarray(masked, mode="RGB")
    masked_img.save(path)


def write_result_json(result: ExtractionResult, output_path: str | Path) -> None:
    path = Path(output_path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = json.dumps(result.to_dict(), indent=2)
    path.write_text(payload + "\n", encoding="utf-8")
