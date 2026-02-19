from __future__ import annotations

from pathlib import Path

from .extract import extract_dominant_colors
from .io import read_image_rgb, save_masked_image
from .localize import CombinedMaskProvider, MaskProvider
from .models import ExtractionResult
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
    ) -> ExtractionResult:
        image_rgb = read_image_rgb(image_path)
        mask_provider = self.mask_provider or CombinedMaskProvider(title=title)
        mask = mask_provider.get_mask(image_rgb)

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

        if debug_mask_out:
            save_masked_image(image_rgb, mask, debug_mask_out)

        return ExtractionResult(
            primary=named_colors[0],
            dominant_colors=named_colors,
            palette_source=palette_source,
            mask_coverage=mask_coverage,
            warnings=warnings,
        )
