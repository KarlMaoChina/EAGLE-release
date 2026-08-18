"""EAGLE v1.0: dual-view CT and clinical fusion for gallbladder lesion scores."""

from __future__ import annotations

from eagle.spec import (
    CLINICAL_FEATURE_NAMES,
    DEPLOYMENT_THRESHOLD,
    MODEL_VERSION,
    N_FOLDS,
    PATCH_SIZE,
    SELECTED_RADIOMICS_FEATURES,
)

__all__ = [
    "CLINICAL_FEATURE_NAMES",
    "DEPLOYMENT_THRESHOLD",
    "MODEL_VERSION",
    "N_FOLDS",
    "PATCH_SIZE",
    "SELECTED_RADIOMICS_FEATURES",
]

__version__ = MODEL_VERSION
