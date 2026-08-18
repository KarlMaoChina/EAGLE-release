# -*- coding: utf-8 -*-
"""Check that spec.py and eagle_v1.yaml list the same v1.0 constants."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eagle.runtime import load_yaml
from eagle.spec import (
    ATTENTION_HEADS,
    CHANNEL_SCALE,
    CLINICAL_FEATURE_NAMES,
    CLOSING_RADIUS_MM,
    DEPLOYMENT_THRESHOLD,
    DVSE_BATCH_SIZE,
    DVSE_EARLY_STOPPING,
    DVSE_LEARNING_RATE,
    DVSE_SCHEDULER_PATIENCE,
    DVSE_WEIGHT_DECAY,
    ENLARGED_SPACING_MM,
    FINAL_DROPOUT,
    FUSION_BATCH_SIZE,
    FUSION_EARLY_STOPPING,
    FUSION_LEARNING_RATE,
    FUSION_MAX_EPOCHS,
    FUSION_SCHEDULER_PATIENCE,
    FUSION_WEIGHT_DECAY,
    HIDDEN_DROPOUT,
    MISSINGNESS_EXCLUSION_FRACTION,
    N_FOLDS,
    N_RADIOMICS_EXTRACTED,
    N_RADIOMICS_SELECTED,
    NEARBY_COMPONENT_MAX_DISTANCE_MM,
    PATCH_SIZE,
    RADIOMICS_BIN_WIDTH,
    RADIOMICS_INTENSITY_SCALE,
    RADIOMICS_LOG_SIGMAS_MM,
    RADIOMICS_ROI_DILATION_MM,
    RADIOMICS_SPACING_MM,
    RANDOM_SEED,
    REDUCTION_FACTOR,
    RELATIVE_VOLUME_THRESHOLD,
    SELECTED_RADIOMICS_FEATURES,
    SELECTION_AUC_WEIGHT,
    STANDARD_SPACING_MM,
    UNIFIED_FEATURE_DIM,
    WALL_THICKNESS_MM,
    WINDOW_LEVEL_HU,
    WINDOW_WIDTH_HU,
)

SPEC_RADIOMICS = (
    "original_firstorder_InterquartileRange",
    "wavelet-HLL_firstorder_10Percentile",
    "log-sigma-2-0-mm-3D_firstorder_Entropy",
    "wavelet-HHH_firstorder_10Percentile",
    "logarithm_firstorder_Uniformity",
    "wavelet-HLH_firstorder_Entropy",
    "square_firstorder_Skewness",
    "squareroot_firstorder_Entropy",
    "original_firstorder_Entropy",
    "log-sigma-2-0-mm-3D_glcm_MaximumProbability",
    "wavelet-LLH_glcm_Contrast",
    "original_glcm_Imc1",
    "squareroot_glcm_SumSquares",
    "wavelet-HLH_glcm_Imc1",
    "wavelet-HHH_glcm_Imc2",
    "wavelet-LHH_glcm_Imc1",
    "wavelet-LLL_glcm_MaximumProbability",
    "wavelet-LHL_gldm_DependenceEntropy",
    "wavelet-LLL_gldm_DependenceEntropy",
    "logarithm_gldm_SmallDependenceEmphasis",
    "log-sigma-1-0-mm-3D_gldm_LargeDependenceEmphasis",
    "wavelet-LHH_gldm_SmallDependenceEmphasis",
    "wavelet-LHH_gldm_DependenceEntropy",
    "wavelet-LLH_glszm_ZoneEntropy",
    "wavelet-LLL_glszm_LargeAreaLowGrayLevelEmphasis",
    "wavelet-LHH_glszm_SizeZoneNonUniformityNormalized",
    "wavelet-HHH_glrlm_RunVariance",
    "squareroot_glrlm_GrayLevelNonUniformityNormalized",
    "logarithm_glrlm_GrayLevelNonUniformityNormalized",
    "wavelet-LHH_glrlm_LowGrayLevelRunEmphasis",
    "wavelet-LHH_glrlm_RunEntropy",
    "wavelet-HHH_glrlm_GrayLevelNonUniformityNormalized",
)

PAPER_CLINICAL = (
    "cholelithiasis",
    "age",
    "ag_ratio",
    "alp",
    "prealbumin",
    "dbil",
    "tbil",
    "cea",
    "afp",
    "ca19_9",
    "ca15_3",
    "ca125",
)

CHECKS = [
    ("clinical_n", 12, len(CLINICAL_FEATURE_NAMES)),
    ("clinical_names", PAPER_CLINICAL, CLINICAL_FEATURE_NAMES),
    ("radiomics_n", 32, len(SELECTED_RADIOMICS_FEATURES)),
    ("radiomics_extracted", 1470, N_RADIOMICS_EXTRACTED),
    ("radiomics_names", PAPER_RADIOMICS, SELECTED_RADIOMICS_FEATURES),
    ("threshold", 0.5, DEPLOYMENT_THRESHOLD),
    ("folds", 5, N_FOLDS),
    ("seed", 42, RANDOM_SEED),
    ("patch", (96, 112, 80), PATCH_SIZE),
    ("standard_spacing", (0.72, 0.72, 1.0), STANDARD_SPACING_MM),
    ("enlarged_spacing", (0.9, 0.9, 2.0), ENLARGED_SPACING_MM),
    ("window_level", 40, WINDOW_LEVEL_HU),
    ("window_width", 300, WINDOW_WIDTH_HU),
    ("radiomics_spacing", (1.0, 1.0, 1.0), RADIOMICS_SPACING_MM),
    ("radiomics_bin", 5, RADIOMICS_BIN_WIDTH),
    ("radiomics_scale", 100.0, RADIOMICS_INTENSITY_SCALE),
    ("radiomics_dilate_mm", 5.0, RADIOMICS_ROI_DILATION_MM),
    ("radiomics_log_sigma", (1.0, 2.0, 3.0), RADIOMICS_LOG_SIGMAS_MM),
    ("dvse_lr", 5.0e-4, DVSE_LEARNING_RATE),
    ("dvse_wd", 0.01, DVSE_WEIGHT_DECAY),
    ("dvse_bs", 16, DVSE_BATCH_SIZE),
    ("dvse_sched_patience", 8, DVSE_SCHEDULER_PATIENCE),
    ("dvse_early_stop", 15, DVSE_EARLY_STOPPING),
    ("fusion_lr", 5.0e-4, FUSION_LEARNING_RATE),
    ("fusion_wd", 0.15, FUSION_WEIGHT_DECAY),
    ("fusion_bs", 8, FUSION_BATCH_SIZE),
    ("fusion_sched_patience", 4, FUSION_SCHEDULER_PATIENCE),
    ("fusion_early_stop", 12, FUSION_EARLY_STOPPING),
    ("fusion_max_epochs", 100, FUSION_MAX_EPOCHS),
    ("hidden_dropout", 0.4, HIDDEN_DROPOUT),
    ("final_dropout", 0.3, FINAL_DROPOUT),
    ("auc_weight", 0.7, SELECTION_AUC_WEIGHT),
    ("reduction", 4, REDUCTION_FACTOR),
    ("attn_heads", 4, ATTENTION_HEADS),
    ("unified_dim", 256, UNIFIED_FEATURE_DIM),
    ("channel_scale", 0.4, CHANNEL_SCALE),
    ("dvse_feature_dim", 204, int(512 * CHANNEL_SCALE)),
    ("missing_frac", 0.30, MISSINGNESS_EXCLUSION_FRACTION),
    ("vol_thresh", 0.05, RELATIVE_VOLUME_THRESHOLD),
    ("nearby_mm", 20.0, NEARBY_COMPONENT_MAX_DISTANCE_MM),
    ("closing_mm", 2.0, CLOSING_RADIUS_MM),
    ("wall_mm", 1.5, WALL_THICKNESS_MM),
]


def _yaml_checks() -> list[tuple[str, object, object]]:
    cfg = load_yaml(ROOT / "configs" / "eagle_v1.yaml")
    return [
        ("yaml.seed", RANDOM_SEED, cfg["seed"]),
        ("yaml.num_folds", N_FOLDS, cfg["num_folds"]),
        ("yaml.threshold", DEPLOYMENT_THRESHOLD, cfg["deployment_threshold"]),
        ("yaml.dvse.lr", DVSE_LEARNING_RATE, cfg["dvse"]["learning_rate"]),
        ("yaml.dvse.wd", DVSE_WEIGHT_DECAY, cfg["dvse"]["weight_decay"]),
        ("yaml.dvse.bs", DVSE_BATCH_SIZE, cfg["dvse"]["batch_size"]),
        ("yaml.dvse.patience", DVSE_SCHEDULER_PATIENCE, cfg["dvse"]["scheduler_patience"]),
        ("yaml.dvse.early_stop", DVSE_EARLY_STOPPING, cfg["dvse"]["early_stopping"]),
        ("yaml.fusion.lr", FUSION_LEARNING_RATE, cfg["fusion"]["learning_rate"]),
        ("yaml.fusion.wd", FUSION_WEIGHT_DECAY, cfg["fusion"]["weight_decay"]),
        ("yaml.fusion.bs", FUSION_BATCH_SIZE, cfg["fusion"]["batch_size"]),
        ("yaml.fusion.patience", FUSION_SCHEDULER_PATIENCE, cfg["fusion"]["scheduler_patience"]),
        ("yaml.fusion.early_stop", FUSION_EARLY_STOPPING, cfg["fusion"]["early_stopping"]),
        ("yaml.fusion.max_epochs", FUSION_MAX_EPOCHS, cfg["fusion"]["max_epochs"]),
        ("yaml.fusion.reduction", REDUCTION_FACTOR, cfg["fusion"]["reduction_factor"]),
        ("yaml.fusion.hidden_dropout", HIDDEN_DROPOUT, cfg["fusion"]["hidden_dropout"]),
        ("yaml.fusion.final_dropout", FINAL_DROPOUT, cfg["fusion"]["final_dropout"]),
        ("yaml.fusion.auc_weight", SELECTION_AUC_WEIGHT, cfg["fusion"]["selection_auc_weight"]),
    ]


def main() -> int:
    lines = ["EAGLE v1.0 specification check", ""]
    n_ok = 0
    checks = list(CHECKS) + _yaml_checks()
    for name, expected, got in checks:
        ok = expected == got
        n_ok += int(ok)
        mark = "OK" if ok else "DIFF"
        lines.append(f"[{mark}] {name}: expected={expected!r} got={got!r}")
    lines.append("")
    lines.append(f"matched {n_ok}/{len(checks)} numeric/name checks")
    text = "\n".join(lines) + "\n"
    print(text, end="")
    out_dir = ROOT / "outputs"
    out_dir.mkdir(parents=True, exist_ok=True)
    (out_dir / "spec_check.txt").write_text(text, encoding="utf-8")
    return 0 if n_ok == len(checks) else 1


if __name__ == "__main__":
    raise SystemExit(main())
