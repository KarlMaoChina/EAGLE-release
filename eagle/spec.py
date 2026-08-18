"""Frozen EAGLE v1.0 specification.

Values match the public methods: 12 clinical inputs, 32 IBSI radiomic
features, dual-view geometry, BiAMF hyperparameters, and deployment
threshold T = 0.5. No site identifiers or local paths belong here.
"""

from __future__ import annotations

from dataclasses import dataclass
from importlib.resources import files
from typing import Literal

FeatureKind = Literal["continuous", "categorical"]


@dataclass(frozen=True)
class ClinicalFeature:
    name: str
    kind: FeatureKind
    aliases: tuple[str, ...] = ()


MODEL_NAME = "EAGLE"
MODEL_VERSION = "1.0"
RANDOM_SEED = 42
N_FOLDS = 5
DEPLOYMENT_THRESHOLD = 0.5

# Dual-view diagnostic tensors (D, H, W) in array order matching NIfTI data.
PATCH_SIZE = (96, 112, 80)
STANDARD_SPACING_MM = (0.72, 0.72, 1.0)
ENLARGED_SPACING_MM = (0.9, 0.9, 2.0)
WINDOW_LEVEL_HU = 40
WINDOW_WIDTH_HU = 300
MIN_MASK_VOXELS = 100

# Anatomy-aware segmentation post-process.
GAUSSIAN_PRIOR_MEANS = (0.34, 0.38, 0.60)
GAUSSIAN_PRIOR_STDS = (0.03, 0.05, 0.15)
GAUSSIAN_PRIOR_WEIGHTS = (0.40, 0.30, 0.30)
GAUSSIAN_PRIOR_SCORE_MIN = 0.3
RELATIVE_VOLUME_THRESHOLD = 0.05
NEARBY_COMPONENT_MAX_DISTANCE_MM = 20.0
CLOSING_RADIUS_MM = 2.0
BOUNDARY_SMOOTH_SIGMA = 0.5

# Region-adaptive mask weights.
WALL_THICKNESS_MM = 1.5
STANDARD_EXPANDED_BOUNDARY_MM = 3.0
STANDARD_INTERIOR_WEIGHT = 0.7
STANDARD_WALL_WEIGHT = 0.9
STANDARD_EXPANDED_WEIGHT = 1.0
ENLARGED_WALL_WEIGHT = 1.0
ENLARGED_PROXIMAL_MM = 5.0
ENLARGED_PROXIMAL_WEIGHT = 0.8
ENLARGED_INTERIOR_WEIGHT = 0.6
ENLARGED_INTERMEDIATE_MM = 10.0
ENLARGED_INTERMEDIATE_WEIGHT = 0.4
ENLARGED_DISTAL_MM = 15.0
ENLARGED_DISTAL_WEIGHT = 0.2

# Radiomics (IBSI / PyRadiomics).
RADIOMICS_SPACING_MM = (1.0, 1.0, 1.0)
RADIOMICS_BIN_WIDTH = 5
RADIOMICS_INTENSITY_SCALE = 100.0
RADIOMICS_ROI_DILATION_MM = 5.0
RADIOMICS_LOG_SIGMAS_MM = (1.0, 2.0, 3.0)
N_RADIOMICS_EXTRACTED = 1470
N_RADIOMICS_SELECTED = 32

# DVSE / BiAMF.
CHANNEL_SCALE = 0.4  # yields a 204-d pooled vector from ResNet-18-3D
UNIFIED_FEATURE_DIM = 256
ATTENTION_HEADS = 4
REDUCTION_FACTOR = 4
HIDDEN_DROPOUT = 0.4
FINAL_DROPOUT = 0.3
TABULAR_HIDDEN = (128, 256)

DVSE_LEARNING_RATE = 5.0e-4
DVSE_WEIGHT_DECAY = 0.01
DVSE_BATCH_SIZE = 16
DVSE_SCHEDULER_PATIENCE = 8
DVSE_EARLY_STOPPING = 15

FUSION_LEARNING_RATE = 5.0e-4
FUSION_WEIGHT_DECAY = 0.15
FUSION_BATCH_SIZE = 8
FUSION_SCHEDULER_PATIENCE = 4
FUSION_EARLY_STOPPING = 12
FUSION_MAX_EPOCHS = 100
SCHEDULER_FACTOR = 0.5
MIN_LEARNING_RATE = 1.0e-6
SELECTION_AUC_WEIGHT = 0.7
MISSINGNESS_EXCLUSION_FRACTION = 0.30

CLINICAL_FEATURES: tuple[ClinicalFeature, ...] = (
    ClinicalFeature("cholelithiasis", "categorical", ("history of cholelithiasis", "gallbladder stones")),
    ClinicalFeature("age", "continuous", ("age at diagnosis",)),
    ClinicalFeature("ag_ratio", "continuous", ("a/g ratio", "a/g", "albumin/globulin ratio")),
    ClinicalFeature("alp", "continuous", ("alkaline phosphatase",)),
    ClinicalFeature("prealbumin", "continuous", ("pa",)),
    ClinicalFeature("dbil", "continuous", ("direct bilirubin",)),
    ClinicalFeature("tbil", "continuous", ("total bilirubin",)),
    ClinicalFeature("cea", "continuous", ("carcinoembryonic antigen",)),
    ClinicalFeature("afp", "continuous", ("alpha-fetoprotein",)),
    ClinicalFeature("ca19_9", "continuous", ("ca19-9", "ca199", "carbohydrate antigen 19-9")),
    ClinicalFeature("ca15_3", "continuous", ("ca15-3", "ca153", "carbohydrate antigen 15-3")),
    ClinicalFeature("ca125", "continuous", ("ca-125", "carbohydrate antigen 125")),
)

CLINICAL_FEATURE_NAMES: tuple[str, ...] = tuple(item.name for item in CLINICAL_FEATURES)
CONTINUOUS_CLINICAL_FEATURES: tuple[str, ...] = tuple(
    item.name for item in CLINICAL_FEATURES if item.kind == "continuous"
)
CATEGORICAL_CLINICAL_FEATURES: tuple[str, ...] = tuple(
    item.name for item in CLINICAL_FEATURES if item.kind == "categorical"
)


def load_selected_radiomics_names() -> tuple[str, ...]:
    text = files("eagle.assets").joinpath("radiomics_features_v1.txt").read_text(encoding="utf-8")
    names = tuple(line.strip() for line in text.splitlines() if line.strip() and not line.startswith("#"))
    if len(names) != N_RADIOMICS_SELECTED:
        raise RuntimeError(f"Expected {N_RADIOMICS_SELECTED} frozen radiomic names, found {len(names)}")
    return names


SELECTED_RADIOMICS_FEATURES: tuple[str, ...] = load_selected_radiomics_names()
