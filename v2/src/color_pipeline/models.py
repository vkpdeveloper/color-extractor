from __future__ import annotations

from dataclasses import dataclass
from typing import Any

RGB = tuple[int, int, int]
LAB = tuple[float, float, float]


@dataclass(frozen=True)
class PaletteEntry:
    name: str
    code: str | None
    hex: str | None
    lab: LAB


@dataclass(frozen=True)
class ExtractedColor:
    hex: str
    rgb: RGB
    lab: LAB
    proportion: float
    matched_name: str | None = None
    matched_palette_name: str | None = None
    matched_code: str | None = None
    delta_e: float | None = None

    def to_dict(self) -> dict[str, Any]:
        return {
            "hex": self.hex,
            "rgb": list(self.rgb),
            "lab": [float(v) for v in self.lab],
            "proportion": float(self.proportion),
            "percentage": float(self.proportion * 100.0),
            "matched_name": self.matched_name,
            "matched_palette_name": self.matched_palette_name,
            "matched_code": self.matched_code,
            "delta_e": None if self.delta_e is None else float(self.delta_e),
        }


@dataclass(frozen=True)
class ExtractionResult:
    primary: ExtractedColor
    dominant_colors: list[ExtractedColor]
    palette_source: str
    mask_coverage: float
    warnings: list[str]

    def to_dict(self) -> dict[str, Any]:
        return {
            "primary": self.primary.to_dict(),
            "dominant_colors": [color.to_dict() for color in self.dominant_colors],
            "palette_source": self.palette_source,
            "mask_coverage": float(self.mask_coverage),
            "warnings": list(self.warnings),
        }
