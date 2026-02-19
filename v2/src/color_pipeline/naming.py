from __future__ import annotations

from dataclasses import replace
import re

import numpy as np
from skimage.color import deltaE_ciede2000

from .models import ExtractedColor, PaletteEntry


def assign_palette_names(
    colors: list[ExtractedColor],
    palette: list[PaletteEntry],
    humanize: bool = False,
) -> list[ExtractedColor]:
    if not palette:
        raise ValueError("palette must contain at least one entry")

    palette_lab = np.asarray(
        [entry.lab for entry in palette], dtype=np.float64
    ).reshape(1, -1, 3)
    named: list[ExtractedColor] = []

    for color in colors:
        source_lab = np.asarray(color.lab, dtype=np.float64).reshape(1, 1, 3)
        distances = deltaE_ciede2000(source_lab, palette_lab).reshape(-1)
        if _is_bright_neutral(color.lab):
            distances = _apply_bright_neutral_penalty(color.lab, palette_lab, distances)
        if humanize:
            matched_name, representative_entry, representative_distance = (
                _resolve_human_label(
                    palette=palette,
                    distances=distances,
                )
            )
        else:
            best_idx = int(np.argmin(distances))
            representative_entry = palette[best_idx]
            representative_distance = float(distances[best_idx])
            matched_name = representative_entry.name

        named.append(
            replace(
                color,
                matched_name=matched_name,
                matched_palette_name=representative_entry.name,
                matched_code=representative_entry.code,
                delta_e=float(representative_distance),
            )
        )

    return named


def _resolve_human_label(
    palette: list[PaletteEntry],
    distances: np.ndarray,
    top_n: int = 7,
) -> tuple[str, PaletteEntry, float]:
    ordered_indices = np.argsort(distances)
    top_indices = ordered_indices[: max(1, top_n)]

    group_scores: dict[str, float] = {}
    group_best: dict[str, tuple[PaletteEntry, float]] = {}

    for idx in top_indices:
        entry = palette[int(idx)]
        distance = float(distances[int(idx)])
        human_name = _humanize_name(entry.name)

        # Soft vote: close swatches influence the same human label.
        score = float(np.exp(-distance / 6.0))
        group_scores[human_name] = group_scores.get(human_name, 0.0) + score

        current_best = group_best.get(human_name)
        if current_best is None or distance < current_best[1]:
            group_best[human_name] = (entry, distance)

    winner = max(
        group_scores.items(),
        key=lambda item: (item[1], -group_best[item[0]][1]),
    )[0]
    representative_entry, representative_distance = group_best[winner]
    return winner, representative_entry, representative_distance


def _is_bright_neutral(lab: tuple[float, float, float]) -> bool:
    l_star, a_star, b_star = lab
    chroma = float(np.sqrt(a_star * a_star + b_star * b_star))
    return l_star >= 88.0 and chroma <= 6.0


def _apply_bright_neutral_penalty(
    source_lab: tuple[float, float, float],
    palette_lab: np.ndarray,
    distances: np.ndarray,
) -> np.ndarray:
    l_star, _, _ = source_lab
    palette_l = palette_lab.reshape(-1, 3)[:, 0]
    palette_chroma = np.sqrt(
        np.square(palette_lab.reshape(-1, 3)[:, 1])
        + np.square(palette_lab.reshape(-1, 3)[:, 2])
    )
    penalty = np.zeros_like(distances)
    darker = palette_l < max(l_star - 2.0, 0.0)
    neutral = palette_chroma <= 8.0
    penalty[darker & neutral] = 4.0
    return distances + penalty


def _humanize_name(name: str) -> str:
    normalized = name.strip().lower()
    words = set(re.findall(r"[a-z]+", normalized))

    def has_phrase(phrase: str) -> bool:
        return phrase in normalized

    def has_any_phrases(*phrases: str) -> bool:
        return any(has_phrase(phrase) for phrase in phrases)

    def has_word(word: str) -> bool:
        return word in words

    def has_any_words(*candidates: str) -> bool:
        return any(has_word(candidate) for candidate in candidates)

    # Metals.
    if has_phrase("rose gold"):
        return "Rose Gold"
    if has_word("gold"):
        return "Gold"
    if has_word("silver"):
        return "Silver"
    if has_word("champagne"):
        return "Champagne"

    # Whites / neutrals.
    if has_any_phrases(
        "off white", "soft white", "warm white", "natural white", "milk white"
    ):
        return "Off White"
    if has_word("white"):
        return "White"
    if has_word("ivory"):
        return "Ivory"
    if has_any_words("cream", "vanilla"):
        return "Cream"
    if has_any_words("beige", "bone", "parchment", "sand", "cashew"):
        return "Beige"

    # Dark neutrals.
    if has_word("black"):
        return "Black"
    if has_any_words("grey", "gray", "anthracite", "charcoal", "steel", "metal", "ash"):
        return "Grey"

    # Blues / cyan.
    if has_any_phrases("black navy", "deep navy", "ink blue"):
        return "Navy Blue"
    if has_any_words("navy", "marine", "midnight", "snorkel"):
        return "Navy Blue"
    if has_phrase("royal blue"):
        return "Royal Blue"
    if has_word("denim"):
        return "Denim Blue"
    if has_any_phrases("ice blue", "airy blue"):
        return "Sky Blue"
    if has_any_words("sky", "powder"):
        return "Sky Blue"
    if has_phrase("petrol blue") or has_word("teal"):
        return "Teal"
    if has_any_words("turquoise", "aqua"):
        return "Turquoise"
    if has_word("blue"):
        return "Blue"

    # Greens.
    if has_word("rama"):
        return "Rama Green"
    if has_word("olive"):
        return "Olive Green"
    if has_word("mint"):
        return "Mint Green"
    if has_word("sage"):
        return "Sage Green"
    if has_word("bottle"):
        return "Bottle Green"
    if has_word("parrot"):
        return "Parrot Green"
    if has_word("mehendi"):
        return "Mehendi Green"
    if has_word("emerald"):
        return "Emerald Green"
    if has_word("green"):
        return "Green"

    # Reds / pinks.
    if has_any_words("maroon", "burgundy", "wine", "beetroot", "marsala", "cordovan"):
        return "Maroon"
    if has_word("rani"):
        return "Rani Pink"
    if has_any_words("fuchsia", "pink", "blush", "rose", "raspberry"):
        return "Pink"
    if has_word("coral"):
        return "Coral"
    if has_word("peach"):
        return "Peach"
    if has_any_words("red", "scarlet", "chilli"):
        return "Red"

    # Yellow / orange.
    if has_word("mustard"):
        return "Mustard"
    if has_word("saffron"):
        return "Saffron"
    if has_any_words("yellow", "lemon", "daffodil", "golden"):
        return "Yellow"
    if has_any_words("orange", "tangerine", "amber", "turmeric", "mango"):
        return "Orange"
    if has_word("rust"):
        return "Rust"
    if has_word("ochre"):
        return "Ochre"

    # Browns.
    if has_any_words(
        "camel",
        "khaki",
        "toffee",
        "cognac",
        "coffee",
        "tobacco",
        "macchiato",
        "brown",
        "wood",
        "earth",
        "mushroom",
        "taupe",
    ):
        return "Brown"
    if has_word("tan"):
        return "Tan"

    # Purples.
    if has_any_words(
        "violet", "purple", "plum", "lilac", "lavender", "orchid", "eggplant", "mauve"
    ):
        return "Purple"

    return name.strip().title()
