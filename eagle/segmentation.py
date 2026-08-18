"""Anatomy-aware post-processing for gallbladder masks."""

from __future__ import annotations

from typing import Any

import numpy as np
from scipy import ndimage

from eagle.spec import (
    BOUNDARY_SMOOTH_SIGMA,
    CLOSING_RADIUS_MM,
    GAUSSIAN_PRIOR_MEANS,
    GAUSSIAN_PRIOR_SCORE_MIN,
    GAUSSIAN_PRIOR_STDS,
    GAUSSIAN_PRIOR_WEIGHTS,
    NEARBY_COMPONENT_MAX_DISTANCE_MM,
    RELATIVE_VOLUME_THRESHOLD,
)


def gaussian_score(value: float, mean: float, std: float) -> float:
    if std <= 0:
        return 0.0
    return float(np.exp(-0.5 * ((value - mean) / std) ** 2))


def ellipsoid_structure(radius_mm: float, spacing: tuple[float, float, float]) -> np.ndarray:
    radii = [max(1, int(round(radius_mm / float(step)))) for step in spacing]
    grids = np.ogrid[tuple(slice(-radius, radius + 1) for radius in radii)]
    distance = sum((grid / radius) ** 2 for grid, radius in zip(grids, radii))
    return distance <= 1.0


def _relative_centroid(mask: np.ndarray) -> tuple[float, float, float]:
    centroid = ndimage.center_of_mass(mask)
    return tuple(float(coord / size) for coord, size in zip(centroid, mask.shape))


def _position_score(relative: tuple[float, float, float]) -> float:
    return float(
        sum(
            weight * gaussian_score(coord, mean, std)
            for coord, mean, std, weight in zip(
                relative,
                GAUSSIAN_PRIOR_MEANS,
                GAUSSIAN_PRIOR_STDS,
                GAUSSIAN_PRIOR_WEIGHTS,
            )
        )
    )


def postprocess_segmentation(
    segmentation: np.ndarray,
    spacing: tuple[float, float, float],
    relative_volume_threshold: float = RELATIVE_VOLUME_THRESHOLD,
    max_distance_mm: float = NEARBY_COMPONENT_MAX_DISTANCE_MM,
    smooth_sigma: float = BOUNDARY_SMOOTH_SIGMA,
    closing_radius_mm: float = CLOSING_RADIUS_MM,
) -> tuple[np.ndarray, dict[str, Any]]:
    binary = np.asarray(segmentation) > 0
    empty_info = {
        "original_regions": 0,
        "retained_regions": 0,
        "original_volume_mm3": 0.0,
        "processed_volume_mm3": 0.0,
        "removed_small": 0,
        "removed_distant": 0,
        "removed_off_anatomy": 0,
    }
    if not np.any(binary):
        return np.zeros_like(binary, dtype=np.uint8), empty_info

    voxel_volume = float(np.prod(spacing))
    labels, n_regions = ndimage.label(binary)
    sizes = ndimage.sum(binary, labels, index=list(range(1, n_regions + 1)))
    sizes = np.atleast_1d(np.asarray(sizes, dtype=np.float64))
    largest_voxels = float(np.max(sizes))
    min_voxels = largest_voxels * relative_volume_threshold

    candidates: list[tuple[int, float, float]] = []
    removed_off_anatomy = 0
    for label_id in range(1, n_regions + 1):
        region = labels == label_id
        score = _position_score(_relative_centroid(region))
        if score > GAUSSIAN_PRIOR_SCORE_MIN:
            candidates.append((label_id, float(sizes[label_id - 1]), score))
        else:
            removed_off_anatomy += 1

    if candidates:
        max_volume = max(item[1] for item in candidates)
        primary_id = max(
            candidates,
            key=lambda item: 0.6 * (item[1] / max_volume) + 0.4 * item[2],
        )[0]
    else:
        primary_id = int(np.argmax(sizes) + 1)

    kept = labels == primary_id
    distance_map = ndimage.distance_transform_edt(~kept, sampling=spacing)
    removed_small = 0
    removed_distant = 0
    retained = 1

    for label_id in range(1, n_regions + 1):
        if label_id == primary_id:
            continue
        region = labels == label_id
        if float(sizes[label_id - 1]) < min_voxels:
            removed_small += 1
            continue
        if float(np.min(distance_map[region])) > max_distance_mm:
            removed_distant += 1
            continue
        kept = np.logical_or(kept, region)
        retained += 1

    structure = ellipsoid_structure(closing_radius_mm, spacing)
    kept = ndimage.binary_closing(kept, structure=structure)
    kept = ndimage.binary_fill_holes(kept)
    smoothed = ndimage.gaussian_filter(kept.astype(np.float32), sigma=smooth_sigma, mode="nearest")
    result = (smoothed > 0.5).astype(np.uint8)

    info = {
        "original_regions": int(n_regions),
        "retained_regions": int(retained),
        "original_volume_mm3": float(np.sum(binary) * voxel_volume),
        "processed_volume_mm3": float(np.sum(result) * voxel_volume),
        "removed_small": int(removed_small),
        "removed_distant": int(removed_distant),
        "removed_off_anatomy": int(removed_off_anatomy),
    }
    return result, info
