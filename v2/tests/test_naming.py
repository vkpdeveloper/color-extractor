from __future__ import annotations

from v2.src.color_pipeline.models import ExtractedColor, PaletteEntry
from v2.src.color_pipeline.naming import assign_palette_names


def test_assign_palette_names_uses_lowest_delta_e():
    color = ExtractedColor(
        hex="#FF0000",
        rgb=(255, 0, 0),
        lab=(53.2408, 80.0925, 67.2032),
        proportion=1.0,
    )
    palette = [
        PaletteEntry(name="Blue", code="PMS-BLUE", hex="#0000FF", lab=(32.2970, 79.1875, -107.8602)),
        PaletteEntry(name="Red", code="PMS-RED", hex="#FF0000", lab=(53.2408, 80.0925, 67.2032)),
    ]

    named = assign_palette_names([color], palette)

    assert len(named) == 1
    assert named[0].matched_name == "Red"
    assert named[0].matched_palette_name == "Red"
    assert named[0].matched_code == "PMS-RED"
    assert named[0].delta_e is not None
    assert named[0].delta_e < 0.01


def test_assign_palette_names_returns_human_label_for_navy_family():
    color = ExtractedColor(
        hex="#2A2945",
        rgb=(42, 41, 69),
        lab=(18.0326, 7.9902, -16.9788),
        proportion=1.0,
    )
    palette = [
        PaletteEntry(name="Navy Blue", code="19-3923 TCX", hex="#282D3C", lab=(18.3803, 3.8871, -13.7969)),
        PaletteEntry(name="Marine Blue", code="19-3920 TCX", hex="#2B2E43", lab=(19.1628, 6.9176, -13.8434)),
        PaletteEntry(name="Black", code="19-3911 TCX", hex="#26262A", lab=(16.1634, 1.1590, -2.3360)),
    ]

    named = assign_palette_names([color], palette, humanize=True)

    assert named[0].matched_name == "Navy Blue"
    assert named[0].matched_palette_name == "Marine Blue"
    assert named[0].matched_code == "19-3920 TCX"


def test_humanization_does_not_map_pink_to_navy_from_substring_collision():
    color = ExtractedColor(
        hex="#B4939A",
        rgb=(180, 147, 154),
        lab=(64.0365, 13.3820, 0.8559),
        proportion=1.0,
    )
    palette = [
        PaletteEntry(name="Blush Pink", code="15-1614 TCX", hex="#D0939E", lab=(67.9792, 22.4013, 1.8728)),
        PaletteEntry(name="Marine Blue", code="19-3920 TCX", hex="#2B2E43", lab=(19.1628, 6.9176, -13.8434)),
        PaletteEntry(name="Navy Blue", code="19-3923 TCX", hex="#282D3C", lab=(18.3803, 3.8871, -13.7969)),
    ]

    named = assign_palette_names([color], palette, humanize=True)

    assert named[0].matched_name == "Pink"
    assert named[0].matched_palette_name == "Blush Pink"
