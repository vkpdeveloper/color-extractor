from __future__ import annotations

import csv
import json
import re
from pathlib import Path

import numpy as np
from skimage import color as skcolor

from .models import PaletteEntry

_HEX_PATTERN = re.compile(r"^#?[0-9A-Fa-f]{6}$")


class PaletteValidationError(ValueError):
    pass


def load_palette(
    palette_path: str | Path | None,
    fallback_palette_path: str | Path,
) -> tuple[list[PaletteEntry], str]:
    if palette_path is None:
        entries = _load_palette_file(fallback_palette_path)
        return entries, "open_fallback"

    entries = _load_palette_file(palette_path)
    return entries, "pantone_user"


def _load_palette_file(path_like: str | Path) -> list[PaletteEntry]:
    path = Path(path_like)
    if not path.exists():
        raise PaletteValidationError(f"palette file does not exist: {path}")

    if path.suffix.lower() == ".csv":
        entries = _load_csv(path)
    elif path.suffix.lower() == ".json":
        entries = _load_json(path)
    else:
        raise PaletteValidationError(
            f"unsupported palette format '{path.suffix}'. Use .csv or .json"
        )

    if not entries:
        raise PaletteValidationError(f"palette has no usable entries: {path}")
    return entries


def _load_csv(path: Path) -> list[PaletteEntry]:
    with path.open("r", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        if not reader.fieldnames:
            raise PaletteValidationError(f"palette csv has no header: {path}")

        entries: list[PaletteEntry] = []
        for idx, row in enumerate(reader, start=2):
            entries.append(_parse_entry(row, f"{path}:{idx}"))
        return entries


def _load_json(path: Path) -> list[PaletteEntry]:
    payload = json.loads(path.read_text(encoding="utf-8"))

    if isinstance(payload, dict):
        if "colors" not in payload or not isinstance(payload["colors"], list):
            raise PaletteValidationError(
                f"json palette at {path} must be a list or include a 'colors' list"
            )
        records = payload["colors"]
    elif isinstance(payload, list):
        records = payload
    else:
        raise PaletteValidationError(
            f"json palette at {path} must be a list or object with 'colors'"
        )

    entries: list[PaletteEntry] = []
    for idx, record in enumerate(records, start=1):
        if not isinstance(record, dict):
            raise PaletteValidationError(
                f"invalid palette entry at {path}:{idx} (expected object)"
            )
        entries.append(_parse_entry(record, f"{path}:{idx}"))
    return entries


def _parse_entry(raw_entry: dict[str, object], location: str) -> PaletteEntry:
    normalized: dict[str, object] = {
        str(key).strip().lower(): value
        for key, value in raw_entry.items()
        if key is not None
    }

    name = _as_clean_str(normalized.get("name"))
    if not name:
        raise PaletteValidationError(f"{location}: missing required field 'name'")

    code = _as_clean_str(normalized.get("code"))
    hex_value = _as_clean_str(normalized.get("hex"))

    if hex_value:
        rgb = _hex_to_rgb(hex_value, location)
        lab = _rgb_to_lab(rgb)
        canonical_hex = _rgb_to_hex(rgb)
        return PaletteEntry(name=name, code=code, hex=canonical_hex, lab=lab)

    l_raw = normalized.get("l")
    a_raw = normalized.get("a")
    b_raw = normalized.get("b")

    if l_raw is None or a_raw is None or b_raw is None:
        raise PaletteValidationError(
            f"{location}: provide either 'hex' or numeric 'l','a','b' values"
        )

    try:
        lab = (float(l_raw), float(a_raw), float(b_raw))
    except (TypeError, ValueError) as exc:
        raise PaletteValidationError(
            f"{location}: invalid Lab values, expected numeric l/a/b"
        ) from exc

    rgb = _lab_to_rgb(lab)
    return PaletteEntry(name=name, code=code, hex=_rgb_to_hex(rgb), lab=lab)


def _as_clean_str(value: object) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text if text else None


def _hex_to_rgb(value: str, location: str) -> tuple[int, int, int]:
    if not _HEX_PATTERN.match(value):
        raise PaletteValidationError(f"{location}: invalid hex color '{value}'")

    normalized = value[1:] if value.startswith("#") else value
    return (
        int(normalized[0:2], 16),
        int(normalized[2:4], 16),
        int(normalized[4:6], 16),
    )


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"


def _rgb_to_lab(rgb: tuple[int, int, int]) -> tuple[float, float, float]:
    rgb_arr = np.array(rgb, dtype=np.float64).reshape(1, 1, 3) / 255.0
    lab = skcolor.rgb2lab(rgb_arr).reshape(3)
    return float(lab[0]), float(lab[1]), float(lab[2])


def _lab_to_rgb(lab: tuple[float, float, float]) -> tuple[int, int, int]:
    lab_arr = np.array(lab, dtype=np.float64).reshape(1, 1, 3)
    rgb = skcolor.lab2rgb(lab_arr).reshape(3)
    clipped = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])
