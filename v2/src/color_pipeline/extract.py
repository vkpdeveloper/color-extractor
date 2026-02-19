from __future__ import annotations

import math

import numpy as np
from skimage import color as skcolor
from skimage.color import deltaE_ciede2000
from sklearn.cluster import KMeans

from .models import ExtractedColor


def extract_dominant_colors(
    image_rgb: np.ndarray,
    mask: np.ndarray,
    top_k: int = 3,
    max_clusters: int = 4,
    random_state: int = 42,
) -> tuple[list[ExtractedColor], list[str]]:
    warnings: list[str] = []

    if mask.shape[:2] != image_rgb.shape[:2]:
        raise ValueError("mask shape must match image dimensions")

    pixels = image_rgb[mask]
    if pixels.shape[0] == 0:
        warnings.append("mask_empty_using_full_image")
        pixels = image_rgb.reshape(-1, 3)

    pixels_float = pixels.astype(np.float64) / 255.0
    filtered_pixels = _remove_specular_pixels(pixels_float)

    if filtered_pixels.shape[0] < max(32, int(0.25 * pixels_float.shape[0])):
        warnings.append("highlight_filter_removed_too_many_pixels")
        filtered_pixels = pixels_float

    lab_pixels = skcolor.rgb2lab(filtered_pixels.reshape(-1, 1, 3)).reshape(-1, 3)

    if lab_pixels.shape[0] == 0:
        raise RuntimeError("no pixels available after preprocessing")

    effective_max_clusters = max(int(max_clusters), int(top_k), 1)
    min_clusters = max(1, min(int(top_k), effective_max_clusters))

    shadow_bias = _is_shadow_biased(lab_pixels)
    if _is_low_variance(lab_pixels):
        chosen_k = 1
    else:
        chosen_k = _choose_cluster_count(
            lab_pixels,
            max_clusters=effective_max_clusters,
            min_clusters=min_clusters,
            random_state=random_state,
        )

    if chosen_k == 1 and shadow_bias:
        high_l_pixels = _select_high_lightness_pixels(lab_pixels)
        if high_l_pixels.shape[0] > 0:
            centers_lab = np.mean(high_l_pixels, axis=0).reshape(1, 3)
        else:
            centers_lab = np.mean(lab_pixels, axis=0).reshape(1, 3)
        labels = np.zeros(lab_pixels.shape[0], dtype=int)
        counts = np.array([float(lab_pixels.shape[0])], dtype=np.float64)
    else:
        kmeans = KMeans(n_clusters=chosen_k, n_init=10, random_state=random_state)
        labels = kmeans.fit_predict(lab_pixels)
        centers_lab = kmeans.cluster_centers_
        counts = np.bincount(labels, minlength=chosen_k).astype(np.float64)
    centers_lab, counts = _merge_neutral_clusters(centers_lab, counts)
    centers_lab, counts = _merge_chroma_aligned_clusters(centers_lab, counts)
    proportions = counts / max(np.sum(counts), 1.0)

    ordered = np.argsort(proportions)[::-1]
    selected = ordered[: max(1, top_k)]

    extracted: list[ExtractedColor] = []
    for idx in selected:
        lab = (
            float(centers_lab[idx][0]),
            float(centers_lab[idx][1]),
            float(centers_lab[idx][2]),
        )
        rgb = _lab_to_rgb_uint8(np.asarray(lab, dtype=np.float64))
        extracted.append(
            ExtractedColor(
                hex=_rgb_to_hex(rgb),
                rgb=rgb,
                lab=lab,
                proportion=float(proportions[idx]),
            )
        )

    total_kept = sum(color.proportion for color in extracted)
    if total_kept > 0:
        extracted = [
            ExtractedColor(
                hex=color.hex,
                rgb=color.rgb,
                lab=color.lab,
                proportion=float(color.proportion / total_kept),
            )
            for color in extracted
        ]

    return extracted, warnings


def _remove_specular_pixels(pixels_float: np.ndarray) -> np.ndarray:
    hsv = skcolor.rgb2hsv(pixels_float.reshape(-1, 1, 3)).reshape(-1, 3)
    lab = skcolor.rgb2lab(pixels_float.reshape(-1, 1, 3)).reshape(-1, 3)

    l_star = lab[:, 0]
    chroma = np.sqrt(np.square(lab[:, 1]) + np.square(lab[:, 2]))

    specular_hsv = (hsv[:, 2] > 0.96) & (hsv[:, 1] < 0.10)
    specular_lab = (l_star > 97.0) & (chroma < 6.0)

    keep = ~(specular_hsv | specular_lab)
    if np.count_nonzero(keep) == 0:
        return pixels_float
    return pixels_float[keep]


def _choose_cluster_count(
    lab_pixels: np.ndarray,
    max_clusters: int,
    min_clusters: int,
    random_state: int,
    min_improvement: float = 0.12,
) -> int:
    pixel_count = lab_pixels.shape[0]
    if pixel_count < 2:
        return 1

    max_k = max(1, min(max_clusters, int(math.sqrt(pixel_count)), pixel_count))
    if max_k == 1:
        return 1

    sample = lab_pixels
    if pixel_count > 12000:
        rng = np.random.default_rng(random_state)
        indices = rng.choice(pixel_count, 12000, replace=False)
        sample = lab_pixels[indices]

    unique_count = np.unique(np.round(sample, 4), axis=0).shape[0]
    max_k = min(max_k, unique_count)
    if max_k <= 1:
        return 1

    inertias: list[float] = []
    for k in range(1, max_k + 1):
        model = KMeans(n_clusters=k, n_init=10, random_state=random_state)
        model.fit(sample)
        inertias.append(float(model.inertia_))

    chosen = 1
    for i in range(1, len(inertias)):
        prev = inertias[i - 1]
        curr = inertias[i]
        improvement = (prev - curr) / max(prev, 1e-9)
        if improvement >= min_improvement:
            chosen = i + 1
        else:
            break

    min_k = max(1, min(int(min_clusters), max_k))
    return max(min_k, chosen)


def _is_shadow_biased(
    lab_pixels: np.ndarray,
    l_diff_threshold: float = 6.0,
    chroma_median_threshold: float = 8.0,
) -> bool:
    if lab_pixels.size == 0:
        return False
    l_values = lab_pixels[:, 0]
    l_mean = float(np.mean(l_values))
    l_p90 = float(np.percentile(l_values, 90))
    chroma = np.sqrt(np.square(lab_pixels[:, 1]) + np.square(lab_pixels[:, 2]))
    chroma_median = float(np.median(chroma))
    return (
        l_p90 - l_mean
    ) >= l_diff_threshold and chroma_median <= chroma_median_threshold


def _select_high_lightness_pixels(
    lab_pixels: np.ndarray, percentile: float = 75.0
) -> np.ndarray:
    if lab_pixels.size == 0:
        return lab_pixels
    l_values = lab_pixels[:, 0]
    threshold = float(np.percentile(l_values, percentile))
    selected = lab_pixels[l_values >= threshold]
    if selected.shape[0] < max(32, int(0.1 * lab_pixels.shape[0])):
        return lab_pixels
    return selected


def _is_low_variance(
    lab_pixels: np.ndarray,
    l_std_threshold: float = 6.0,
    chroma_std_threshold: float = 4.0,
) -> bool:
    if lab_pixels.size == 0:
        return False
    l_std = float(np.std(lab_pixels[:, 0]))
    chroma = np.sqrt(np.square(lab_pixels[:, 1]) + np.square(lab_pixels[:, 2]))
    chroma_std = float(np.std(chroma))
    return l_std <= l_std_threshold and chroma_std <= chroma_std_threshold


def _merge_neutral_clusters(
    centers_lab: np.ndarray,
    counts: np.ndarray,
    chroma_threshold: float = 10.0,
    delta_e_threshold: float = 8.0,
) -> tuple[np.ndarray, np.ndarray]:
    if centers_lab.shape[0] <= 1:
        return centers_lab, counts

    chroma = np.sqrt(np.square(centers_lab[:, 1]) + np.square(centers_lab[:, 2]))
    neutral_indices = [
        int(i) for i, value in enumerate(chroma) if float(value) <= chroma_threshold
    ]
    if len(neutral_indices) < 2:
        return centers_lab, counts

    merged = [False] * centers_lab.shape[0]
    new_centers: list[np.ndarray] = []
    new_counts: list[float] = []

    for idx in range(centers_lab.shape[0]):
        if merged[idx]:
            continue
        if idx not in neutral_indices:
            new_centers.append(centers_lab[idx])
            new_counts.append(float(counts[idx]))
            merged[idx] = True
            continue

        group = [idx]
        merged[idx] = True
        for jdx in neutral_indices:
            if merged[jdx]:
                continue
            distance = float(
                deltaE_ciede2000(
                    centers_lab[idx].reshape(1, 1, 3),
                    centers_lab[jdx].reshape(1, 1, 3),
                ).reshape(-1)[0]
            )
            if distance <= delta_e_threshold:
                group.append(jdx)
                merged[jdx] = True

        group_counts = counts[group].astype(np.float64)
        weight = float(np.sum(group_counts))
        if weight <= 0:
            continue
        weighted_center = (centers_lab[group] * group_counts[:, None]).sum(
            axis=0
        ) / weight
        new_centers.append(weighted_center)
        new_counts.append(weight)

    if not new_centers:
        return centers_lab, counts
    return np.asarray(new_centers, dtype=np.float64), np.asarray(
        new_counts, dtype=np.float64
    )


def _merge_chroma_aligned_clusters(
    centers_lab: np.ndarray,
    counts: np.ndarray,
    delta_e_threshold: float = 10.0,
    chroma_min: float = 6.0,
    hue_cosine_threshold: float = 0.985,
) -> tuple[np.ndarray, np.ndarray]:
    if centers_lab.shape[0] <= 1:
        return centers_lab, counts

    a_vals = centers_lab[:, 1]
    b_vals = centers_lab[:, 2]
    chroma = np.sqrt(np.square(a_vals) + np.square(b_vals))

    merged = [False] * centers_lab.shape[0]
    new_centers: list[np.ndarray] = []
    new_counts: list[float] = []

    for idx in range(centers_lab.shape[0]):
        if merged[idx]:
            continue
        if chroma[idx] < chroma_min:
            new_centers.append(centers_lab[idx])
            new_counts.append(float(counts[idx]))
            merged[idx] = True
            continue

        base = centers_lab[idx]
        base_vec = np.array([base[1], base[2]], dtype=np.float64)
        base_norm = float(np.linalg.norm(base_vec))
        if base_norm == 0:
            new_centers.append(centers_lab[idx])
            new_counts.append(float(counts[idx]))
            merged[idx] = True
            continue

        group = [idx]
        merged[idx] = True
        for jdx in range(centers_lab.shape[0]):
            if merged[jdx] or jdx == idx:
                continue
            if chroma[jdx] < chroma_min:
                continue
            cand = centers_lab[jdx]
            cand_vec = np.array([cand[1], cand[2]], dtype=np.float64)
            cand_norm = float(np.linalg.norm(cand_vec))
            if cand_norm == 0:
                continue
            cosine = float(np.dot(base_vec, cand_vec) / (base_norm * cand_norm))
            if cosine < hue_cosine_threshold:
                continue
            distance = float(
                deltaE_ciede2000(
                    base.reshape(1, 1, 3),
                    cand.reshape(1, 1, 3),
                ).reshape(-1)[0]
            )
            if distance <= delta_e_threshold:
                group.append(jdx)
                merged[jdx] = True

        group_counts = counts[group].astype(np.float64)
        weight = float(np.sum(group_counts))
        if weight <= 0:
            continue
        weighted_center = (centers_lab[group] * group_counts[:, None]).sum(
            axis=0
        ) / weight
        new_centers.append(weighted_center)
        new_counts.append(weight)

    if not new_centers:
        return centers_lab, counts
    return np.asarray(new_centers, dtype=np.float64), np.asarray(
        new_counts, dtype=np.float64
    )


def _lab_to_rgb_uint8(lab: np.ndarray) -> tuple[int, int, int]:
    rgb = skcolor.lab2rgb(lab.reshape(1, 1, 3)).reshape(3)
    clipped = np.clip(np.round(rgb * 255.0), 0, 255).astype(np.uint8)
    return int(clipped[0]), int(clipped[1]), int(clipped[2])


def _rgb_to_hex(rgb: tuple[int, int, int]) -> str:
    return f"#{rgb[0]:02X}{rgb[1]:02X}{rgb[2]:02X}"
