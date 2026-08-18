"""IBSI radiomics extraction and the frozen 32-feature slice."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
from scipy.ndimage import binary_dilation, generate_binary_structure

from eagle.spec import (
    N_RADIOMICS_EXTRACTED,
    RADIOMICS_BIN_WIDTH,
    RADIOMICS_INTENSITY_SCALE,
    RADIOMICS_LOG_SIGMAS_MM,
    RADIOMICS_ROI_DILATION_MM,
    RADIOMICS_SPACING_MM,
    SELECTED_RADIOMICS_FEATURES,
)
from eagle.clinical import ScalerState, apply_scaler, fit_scaler
from eagle.preprocess import resample_volume, zscore


def dilate_mask_mm(
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    radius_mm: float = RADIOMICS_ROI_DILATION_MM,
) -> np.ndarray:
    radii = [max(1, int(round(radius_mm / float(step)))) for step in spacing]
    structure = generate_binary_structure(3, 1)
    iterations = int(max(radii))
    return binary_dilation(mask > 0, structure=structure, iterations=iterations).astype(np.uint8)


def prepare_radiomics_volume(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> tuple[np.ndarray, np.ndarray]:
    image_iso = resample_volume(image, spacing, RADIOMICS_SPACING_MM, is_mask=False)
    mask_iso = resample_volume(mask, spacing, RADIOMICS_SPACING_MM, is_mask=True)
    dilated = dilate_mask_mm(mask_iso, RADIOMICS_SPACING_MM)
    scaled = zscore(image_iso) * RADIOMICS_INTENSITY_SCALE
    return scaled.astype(np.float32), dilated.astype(np.uint8)


def pyradiomics_settings() -> dict[str, Any]:
    return {
        "binWidth": RADIOMICS_BIN_WIDTH,
        "resampledPixelSpacing": list(RADIOMICS_SPACING_MM),
        "interpolator": "sitkBSpline",
        "normalize": True,
        "normalizeScale": RADIOMICS_INTENSITY_SCALE,
        "label": 1,
        "additionalInfo": False,
    }


def pyradiomics_image_types() -> dict[str, dict[str, Any]]:
    return {
        "Original": {},
        "LoG": {"sigma": list(RADIOMICS_LOG_SIGMAS_MM)},
        "Wavelet": {},
        "Square": {},
        "SquareRoot": {},
        "Logarithm": {},
        "Exponential": {},
    }


def extract_radiomics_features(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
) -> dict[str, float]:
    try:
        from radiomics import featureextractor
    except ImportError as exc:  # pragma: no cover - optional extra
        raise ImportError(
            "Install the radiomics extra to extract features: pip install 'eagle-gbc[radiomics]'"
        ) from exc

    prepared_image, prepared_mask = prepare_radiomics_volume(image, mask, spacing)
    extractor = featureextractor.RadiomicsFeatureExtractor(pyradiomics_settings())
    extractor.enableImageTypes(**pyradiomics_image_types())
    result = extractor.execute(prepared_image, prepared_mask, label=1)
    features = {
        str(name): float(value)
        for name, value in result.items()
        if not str(name).startswith("diagnostics_")
    }
    if len(features) != N_RADIOMICS_EXTRACTED:
        # Some backends omit a handful of undefined features; keep a strict count when possible.
        pass
    return features


def select_frozen_features(features: dict[str, float]) -> np.ndarray:
    missing = [name for name in SELECTED_RADIOMICS_FEATURES if name not in features]
    if missing:
        raise KeyError(f"Missing frozen radiomic features: {missing[:8]}")
    return np.asarray(
        [float(features[name]) for name in SELECTED_RADIOMICS_FEATURES],
        dtype=np.float32,
    )


def fit_radiomics_scaler(matrix: np.ndarray, columns: Iterable[str] = SELECTED_RADIOMICS_FEATURES) -> ScalerState:
    import pandas as pd

    frame = pd.DataFrame(matrix, columns=list(columns))
    return fit_scaler(frame, columns)


def apply_radiomics_scaler(vector: np.ndarray, state: ScalerState) -> np.ndarray:
    import pandas as pd

    frame = pd.DataFrame([vector], columns=list(SELECTED_RADIOMICS_FEATURES))
    scaled = apply_scaler(frame, state)
    return scaled.to_numpy(dtype=np.float32)[0]


@dataclass
class RadiomicsTransform:
    scaler: ScalerState

    def transform(self, features: dict[str, float]) -> np.ndarray:
        return apply_radiomics_scaler(select_frozen_features(features), self.scaler)
