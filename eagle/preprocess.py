"""Dual-view region preparation for the diagnostic encoder."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import distance_transform_edt, gaussian_filter
from skimage.transform import resize

from eagle.spec import (
    ENLARGED_DISTAL_MM,
    ENLARGED_DISTAL_WEIGHT,
    ENLARGED_INTERIOR_WEIGHT,
    ENLARGED_INTERMEDIATE_MM,
    ENLARGED_INTERMEDIATE_WEIGHT,
    ENLARGED_PROXIMAL_MM,
    ENLARGED_PROXIMAL_WEIGHT,
    ENLARGED_SPACING_MM,
    ENLARGED_WALL_WEIGHT,
    MIN_MASK_VOXELS,
    PATCH_SIZE,
    STANDARD_EXPANDED_BOUNDARY_MM,
    STANDARD_EXPANDED_WEIGHT,
    STANDARD_INTERIOR_WEIGHT,
    STANDARD_SPACING_MM,
    STANDARD_WALL_WEIGHT,
    WALL_THICKNESS_MM,
    WINDOW_LEVEL_HU,
    WINDOW_WIDTH_HU,
)


def apply_window(
    image: np.ndarray,
    level: int = WINDOW_LEVEL_HU,
    width: int = WINDOW_WIDTH_HU,
) -> np.ndarray:
    low = level - width / 2.0
    high = level + width / 2.0
    return np.clip(image.astype(np.float32), low, high)


def zscore(image: np.ndarray) -> np.ndarray:
    values = image.astype(np.float32)
    std = float(np.std(values))
    if std == 0:
        return np.zeros_like(values)
    return ((values - float(np.mean(values))) / std).astype(np.float32)


def resample_volume(
    data: np.ndarray,
    original_spacing: tuple[float, float, float],
    target_spacing: tuple[float, float, float],
    is_mask: bool = False,
) -> np.ndarray:
    scale = np.asarray(original_spacing, dtype=np.float64) / np.asarray(target_spacing, dtype=np.float64)
    new_shape = np.maximum(np.round(np.asarray(data.shape) * scale).astype(int), 1)
    if is_mask:
        return resize(
            data.astype(np.float32),
            new_shape,
            order=0,
            mode="constant",
            anti_aliasing=False,
            preserve_range=True,
        )

    z_scale = float(scale[0])
    source = data.astype(np.float32)
    if z_scale > 3:
        order = 3
    elif z_scale < 0.5:
        sigma = [max(0.0, 0.7 * (1.0 / value - 1.0)) for value in scale]
        source = gaussian_filter(source, sigma)
        order = 2
    else:
        order = 1
    return resize(
        source,
        new_shape,
        order=order,
        mode="edge",
        anti_aliasing=True,
        preserve_range=True,
    )


def extract_centered_patch(
    volume: np.ndarray,
    center: tuple[int, int, int],
    size: tuple[int, int, int] = PATCH_SIZE,
) -> np.ndarray:
    size_arr = np.asarray(size, dtype=int)
    center_arr = np.asarray(center, dtype=int)
    start = center_arr - size_arr // 2
    end = start + size_arr
    dst_start = np.maximum(-start, 0)
    dst_end = size_arr - np.maximum(end - np.asarray(volume.shape), 0)
    src_start = np.maximum(start, 0)
    src_end = np.minimum(end, volume.shape)
    out = np.zeros(size, dtype=volume.dtype)
    if np.any(dst_end <= dst_start) or np.any(src_end <= src_start):
        return out
    dest_slices = tuple(slice(int(s), int(e)) for s, e in zip(dst_start, dst_end))
    src_slices = tuple(slice(int(s), int(e)) for s, e in zip(src_start, src_end))
    out[dest_slices] = volume[src_slices]
    return out


def mask_centroid(mask: np.ndarray) -> tuple[int, int, int]:
    coords = np.where(mask > 0)
    if coords[0].size == 0:
        return tuple(int(dim // 2) for dim in mask.shape)
    return tuple(int(np.round(np.mean(axis))) for axis in coords)


def enhance_standard_mask(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    binary = mask > 0
    weights = np.zeros(mask.shape, dtype=np.float32)
    if not np.any(binary):
        return weights
    dist_in = distance_transform_edt(binary, sampling=spacing)
    dist_out = distance_transform_edt(~binary, sampling=spacing)
    interior = binary & (dist_in > WALL_THICKNESS_MM)
    wall = binary & (dist_in <= WALL_THICKNESS_MM)
    expanded = (~binary) & (dist_out <= STANDARD_EXPANDED_BOUNDARY_MM)
    weights[interior] = STANDARD_INTERIOR_WEIGHT
    weights[wall] = STANDARD_WALL_WEIGHT
    weights[expanded] = STANDARD_EXPANDED_WEIGHT
    return weights


def enhance_enlarged_mask(mask: np.ndarray, spacing: tuple[float, float, float]) -> np.ndarray:
    binary = mask > 0
    weights = np.zeros(mask.shape, dtype=np.float32)
    if not np.any(binary):
        return weights
    dist_in = distance_transform_edt(binary, sampling=spacing)
    dist_out = distance_transform_edt(~binary, sampling=spacing)
    interior = binary & (dist_in > WALL_THICKNESS_MM)
    wall = binary & (dist_in <= WALL_THICKNESS_MM)
    weights[(~binary) & (dist_out <= ENLARGED_DISTAL_MM)] = ENLARGED_DISTAL_WEIGHT
    weights[(~binary) & (dist_out <= ENLARGED_INTERMEDIATE_MM)] = ENLARGED_INTERMEDIATE_WEIGHT
    weights[(~binary) & (dist_out <= ENLARGED_PROXIMAL_MM)] = ENLARGED_PROXIMAL_WEIGHT
    weights[interior] = ENLARGED_INTERIOR_WEIGHT
    weights[wall] = ENLARGED_WALL_WEIGHT
    return weights


@dataclass(frozen=True)
class DualViewCase:
    standard_image: np.ndarray
    standard_mask: np.ndarray
    enlarged_image: np.ndarray
    enlarged_mask: np.ndarray


def prepare_dual_view(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> DualViewCase | None:
    windowed = apply_window(image)
    standard_image = resample_volume(windowed, spacing, STANDARD_SPACING_MM, is_mask=False)
    standard_mask = resample_volume(mask, spacing, STANDARD_SPACING_MM, is_mask=True)
    if float(np.sum(standard_mask > 0)) < MIN_MASK_VOXELS:
        return None
    standard_center = mask_centroid(standard_mask)
    standard_image_patch = zscore(extract_centered_patch(standard_image, standard_center))
    standard_mask_patch = extract_centered_patch(standard_mask, standard_center)
    standard_weights = enhance_standard_mask(standard_mask_patch, STANDARD_SPACING_MM)

    enlarged_image = resample_volume(windowed, spacing, ENLARGED_SPACING_MM, is_mask=False)
    enlarged_mask = resample_volume(mask, spacing, ENLARGED_SPACING_MM, is_mask=True)
    if float(np.sum(enlarged_mask > 0)) < MIN_MASK_VOXELS:
        return None
    enlarged_center = mask_centroid(enlarged_mask)
    enlarged_image_patch = zscore(extract_centered_patch(enlarged_image, enlarged_center))
    enlarged_mask_patch = extract_centered_patch(enlarged_mask, enlarged_center)
    enlarged_weights = enhance_enlarged_mask(enlarged_mask_patch, ENLARGED_SPACING_MM)

    return DualViewCase(
        standard_image=standard_image_patch.astype(np.float32),
        standard_mask=standard_weights.astype(np.float32),
        enlarged_image=enlarged_image_patch.astype(np.float32),
        enlarged_mask=enlarged_weights.astype(np.float32),
    )
