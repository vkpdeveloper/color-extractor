from __future__ import annotations

import numpy as np
from PIL import Image

from v2.src.color_pipeline.localize import HeuristicMaskProvider
from v2.src.color_pipeline.pipeline import ColorExtractionPipeline


class FullMaskProvider:
    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        return np.ones(image_rgb.shape[:2], dtype=bool)


class TinyMaskProvider:
    def get_mask(self, image_rgb: np.ndarray) -> np.ndarray:
        mask = np.zeros(image_rgb.shape[:2], dtype=bool)
        mask[0:1, 0:1] = True
        return mask


def _write_image(path, array):
    Image.fromarray(array.astype(np.uint8), mode="RGB").save(path)


def test_solid_color_maps_to_palette_match(tmp_path):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = [180, 50, 40]
    image_path = tmp_path / "solid.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nSolid Tone,PMS-001,#B43228\nOther Tone,PMS-002,#0022AA\n",
        encoding="utf-8",
    )

    pipeline = ColorExtractionPipeline(mask_provider=FullMaskProvider(), random_state=7)
    result = pipeline.run(
        str(image_path), title="solid tee", palette_path=str(palette_path), top_k=3
    )

    assert result.primary.matched_name == "Solid Tone"
    assert result.primary.delta_e is not None
    assert result.primary.delta_e < 0.5


def test_bicolor_returns_stable_proportions(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :70] = [200, 30, 30]
    image[:, 70:] = [30, 60, 200]
    image_path = tmp_path / "bicolor.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nCrimson,PMS-CRIMSON,#C81E1E\nDenim,PMS-DENIM,#1E3CC8\n",
        encoding="utf-8",
    )

    pipeline = ColorExtractionPipeline(mask_provider=FullMaskProvider(), random_state=7)
    result = pipeline.run(
        str(image_path), title="bicolor shirt", palette_path=str(palette_path), top_k=2
    )

    assert len(result.dominant_colors) == 2
    names = [c.matched_name for c in result.dominant_colors]
    assert names[0] == "Crimson"
    assert names[1] == "Denim"
    assert abs(result.dominant_colors[0].proportion - 0.7) < 0.10
    assert abs(result.dominant_colors[1].proportion - 0.3) < 0.10


def test_highlight_does_not_override_primary_color(tmp_path):
    image = np.zeros((120, 120, 3), dtype=np.uint8)
    image[:, :] = [20, 60, 180]
    image[0:35, 0:35] = [255, 255, 255]
    image_path = tmp_path / "highlight.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nBlue Base,PMS-BLUE,#143CB4\nWhite,PMS-WHITE,#FFFFFF\n",
        encoding="utf-8",
    )

    pipeline = ColorExtractionPipeline(mask_provider=FullMaskProvider(), random_state=3)
    result = pipeline.run(
        str(image_path), title="blue shirt", palette_path=str(palette_path), top_k=2
    )

    assert result.primary.matched_name == "Blue Base"


def test_missing_palette_uses_fallback_source(tmp_path):
    image = np.zeros((64, 64, 3), dtype=np.uint8)
    image[:, :] = [255, 0, 0]
    image_path = tmp_path / "fallback.png"
    _write_image(image_path, image)

    fallback = tmp_path / "fallback_palette.csv"
    fallback.write_text(
        "name,code,hex\nFallback Red,FALLBACK-RED,#FF0000\n", encoding="utf-8"
    )

    pipeline = ColorExtractionPipeline(
        mask_provider=FullMaskProvider(), fallback_palette_path=fallback
    )
    result = pipeline.run(
        str(image_path), title="red shirt", palette_path=None, top_k=1
    )

    assert result.palette_source == "open_fallback"
    assert result.primary.matched_name == "Fallback Red"


def test_tiny_mask_still_produces_output(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [90, 160, 90]
    image_path = tmp_path / "tiny-mask.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nGreen,PMS-GREEN,#5AA05A\n", encoding="utf-8"
    )

    pipeline = ColorExtractionPipeline(mask_provider=TinyMaskProvider())
    result = pipeline.run(
        str(image_path), title="green shirt", palette_path=str(palette_path), top_k=1
    )

    assert result.primary.matched_name == "Green"


def test_heuristic_provider_center_fallback_on_flat_image():
    image = np.full((80, 120, 3), 255, dtype=np.uint8)
    provider = HeuristicMaskProvider()
    mask = provider.get_mask(image)

    assert mask.dtype == bool
    assert mask.shape == image.shape[:2]
    assert 0.05 <= float(mask.mean()) <= 0.60


def test_heuristic_provider_recovers_centered_shirt_like_region(tmp_path):
    image = np.zeros((220, 160, 3), dtype=np.uint8)
    image[:, :] = [232, 221, 205]  # plain warm background
    image[30:215, 45:115] = [44, 58, 132]  # navy shirt body
    image[70:205, 20:45] = [222, 180, 160]  # skin-toned arm-like area
    image[70:205, 115:140] = [222, 180, 160]

    image_path = tmp_path / "shirt-like.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\n"
        "Navy Shirt,PMS-NAVY,#2C3A84\n"
        "Warm Beige,PMS-BEIGE,#E8DDCD\n"
        "Skin Tone,PMS-SKIN,#DEB4A0\n",
        encoding="utf-8",
    )

    pipeline = ColorExtractionPipeline(
        mask_provider=HeuristicMaskProvider(), random_state=11
    )
    result = pipeline.run(
        str(image_path), title="navy shirt", palette_path=str(palette_path), top_k=2
    )

    assert result.primary.matched_name == "Navy Shirt"
    assert result.mask_coverage > 0.20


def test_repeated_runs_are_deterministic(tmp_path):
    image = np.zeros((100, 100, 3), dtype=np.uint8)
    image[:, :] = [160, 40, 140]
    image_path = tmp_path / "repeat.png"
    _write_image(image_path, image)

    palette_path = tmp_path / "palette.csv"
    palette_path.write_text(
        "name,code,hex\nPurple,PMS-PURPLE,#A0288C\n", encoding="utf-8"
    )

    pipeline = ColorExtractionPipeline(
        mask_provider=FullMaskProvider(), random_state=123
    )
    result_a = pipeline.run(
        str(image_path), title="purple top", palette_path=str(palette_path), top_k=1
    )
    result_b = pipeline.run(
        str(image_path), title="purple top", palette_path=str(palette_path), top_k=1
    )

    assert result_a.primary.hex == result_b.primary.hex
    assert result_a.primary.matched_name == result_b.primary.matched_name
    assert result_a.primary.delta_e == result_b.primary.delta_e
