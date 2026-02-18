from __future__ import annotations

import json

import pytest

from v2.src.color_pipeline.palette import PaletteValidationError, load_palette


def test_load_palette_from_csv_hex(tmp_path):
    palette_file = tmp_path / "palette.csv"
    palette_file.write_text(
        "name,code,hex\n"
        "Signal Red,PMS-TEST-RED,#FF0000\n"
        "Deep Blue,PMS-TEST-BLUE,#0033CC\n",
        encoding="utf-8",
    )

    entries, source = load_palette(palette_file, fallback_palette_path=palette_file)

    assert source == "pantone_user"
    assert len(entries) == 2
    assert entries[0].name == "Signal Red"
    assert entries[0].hex == "#FF0000"


def test_load_palette_from_json_lab(tmp_path):
    palette_file = tmp_path / "palette.json"
    palette_file.write_text(
        json.dumps(
            {
                "colors": [
                    {"name": "Lab Color", "code": "PMS-LAB", "l": 50, "a": 20, "b": -30}
                ]
            }
        ),
        encoding="utf-8",
    )

    entries, _ = load_palette(palette_file, fallback_palette_path=palette_file)

    assert len(entries) == 1
    assert entries[0].name == "Lab Color"
    assert entries[0].code == "PMS-LAB"
    assert entries[0].lab == (50.0, 20.0, -30.0)


def test_invalid_palette_schema_raises(tmp_path):
    palette_file = tmp_path / "bad.csv"
    palette_file.write_text("name,code\nMissing Color,PMS-000\n", encoding="utf-8")

    with pytest.raises(PaletteValidationError):
        load_palette(palette_file, fallback_palette_path=palette_file)
