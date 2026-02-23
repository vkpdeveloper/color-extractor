from __future__ import annotations

from pathlib import Path

import numpy as np
from skimage.morphology import closing, disk, opening, remove_small_holes, remove_small_objects

from .extract import extract_dominant_colors
from .io import read_image_rgb, save_masked_image
from .localize import CombinedMaskProvider, MaskProvider
from .models import ExtractedColor, ExtractionResult
from .naming import assign_palette_names
from .palette import load_palette


class ColorExtractionPipeline:
    def __init__(
        self,
        mask_provider: MaskProvider | None = None,
        fallback_palette_path: str | Path | None = None,
        random_state: int = 42,
    ) -> None:
        self.mask_provider = mask_provider
        self.random_state = random_state

        self.default_fallback_palette_path = (
            Path(__file__).resolve().parents[2] / "data" / "open_palette.csv"
        )
        if fallback_palette_path is None:
            fallback_palette_path = self.default_fallback_palette_path
        self.fallback_palette_path = Path(fallback_palette_path)

    def run(
        self,
        image_path: str,
        title: str,
        palette_path: str | None,
        top_k: int = 3,
        debug_mask_out: str | None = None,
        solid_garment_mode: bool = False,
    ) -> ExtractionResult:
        image_rgb = read_image_rgb(image_path)
        mask_provider = self.mask_provider or CombinedMaskProvider(title=title)
        mask = mask_provider.get_mask(image_rgb)
        if solid_garment_mode:
            mask = self._tighten_mask(mask)

        warnings: list[str] = []
        mask_coverage = float(mask.mean())
        if mask_coverage < 0.05:
            warnings.append("low_mask_coverage")
            if top_k > 1:
                top_k = 1
        if mask_coverage > 0.95:
            warnings.append("high_mask_coverage")

        extracted_colors, extract_warnings = extract_dominant_colors(
            image_rgb=image_rgb,
            mask=mask,
            top_k=top_k,
            random_state=self.random_state,
            prefer_high_lightness=mask_coverage < 0.05,
        )
        warnings.extend(extract_warnings)

        palette_entries, palette_source = load_palette(
            palette_path=palette_path,
            fallback_palette_path=self.fallback_palette_path,
        )

        named_colors = assign_palette_names(
            extracted_colors,
            palette_entries,
            humanize=(
                palette_source == "open_fallback"
                and self.fallback_palette_path.resolve()
                == self.default_fallback_palette_path.resolve()
            ),
        )
        if not named_colors:
            raise RuntimeError("no colors were extracted from the image")
        if solid_garment_mode:
            pruned = self._prune_secondary_shades(named_colors)
            if len(pruned) < len(named_colors):
                warnings.append("solid_mode_pruned_secondary_shades")
            named_colors = pruned

        if debug_mask_out:
            save_masked_image(image_rgb, mask, debug_mask_out)

        return ExtractionResult(
            primary=named_colors[0],
            dominant_colors=named_colors,
            palette_source=palette_source,
            mask_coverage=mask_coverage,
            warnings=warnings,
        )

    def _tighten_mask(self, mask: np.ndarray) -> np.ndarray:
        if mask.size == 0:
            return mask
        refined = closing(mask, disk(2))
        refined = remove_small_holes(refined, max_size=2000)
        refined = opening(refined, disk(1))
        refined = remove_small_objects(refined, min_size=400)
        return refined.astype(bool)

    def _prune_secondary_shades(
        self,
        colors: list[ExtractedColor],
        min_secondary_proportion: float = 0.08,
        primary_l_delta_threshold: float = 22.0,
    ) -> list[ExtractedColor]:
        if len(colors) <= 1:
            return colors

        primary = colors[0]
        kept = [primary]
        for color in colors[1:]:
            if color.proportion < min_secondary_proportion:
                continue
            if abs(float(color.lab[0]) - float(primary.lab[0])) > primary_l_delta_threshold:
                continue
            kept.append(color)

        if not kept:
            kept = [primary]

        total = sum(c.proportion for c in kept)
        if total <= 0:
            return [primary]

        normalized: list[ExtractedColor] = []
        for color in kept:
            normalized.append(
                ExtractedColor(
                    hex=color.hex,
                    rgb=color.rgb,
                    lab=color.lab,
                    proportion=float(color.proportion / total),
                    matched_name=color.matched_name,
                    matched_palette_name=color.matched_palette_name,
                    matched_code=color.matched_code,
                    delta_e=color.delta_e,
                )
            )
        return normalized
