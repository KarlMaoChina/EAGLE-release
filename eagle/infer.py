"""End-to-end frozen inference: dual-view tensors + tabular inputs -> probability."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from eagle.clinical import (
    ImputationState,
    ScalerState,
    apply_imputation,
    apply_scaler,
    clinical_vector,
    fit_imputation,
    fit_scaler,
    standardize_clinical_columns,
)
from eagle.data import DualViewSample, load_case_table, load_spacing, load_volume, to_nchw
from eagle.io import read_json, write_json
from eagle.metrics import operating_point_metrics, safe_ap, safe_auc
from eagle.models import BiAMF, load_biamf
from eagle.preprocess import prepare_dual_view
from eagle.radiomics import apply_radiomics_scaler, select_frozen_features
from eagle.segmentation import postprocess_segmentation
from eagle.spec import (
    CONTINUOUS_CLINICAL_FEATURES,
    DEPLOYMENT_THRESHOLD,
    N_FOLDS,
    PATCH_SIZE,
    SELECTED_RADIOMICS_FEATURES,
)


@dataclass
class FreezePackage:
    imputation: ImputationState
    clinical_scaler: ScalerState
    radiomics_scaler: ScalerState
    fusion_paths: list[Path]

    @classmethod
    def from_dir(cls, root: str | Path) -> "FreezePackage":
        root = Path(root)
        stats = read_json(root / "preprocess_stats.json")
        fusion_dir = root / "biamf"
        paths = [fusion_dir / f"fold_{fold}.pt" for fold in range(N_FOLDS)]
        missing = [str(path) for path in paths if not path.is_file()]
        if missing:
            raise FileNotFoundError(
                "Missing fusion checkpoints. Expected "
                f"{fusion_dir}/fold_{{0..4}}.pt. Missing: {missing}"
            )
        return cls(
            imputation=ImputationState.from_dict(stats["imputation"]),
            clinical_scaler=ScalerState.from_dict(stats["clinical_scaler"]),
            radiomics_scaler=ScalerState.from_dict(stats["radiomics_scaler"]),
            fusion_paths=paths,
        )


def write_freeze_stats(
    output_dir: str | Path,
    imputation: ImputationState,
    clinical_scaler: ScalerState,
    radiomics_scaler: ScalerState,
) -> Path:
    return write_json(
        Path(output_dir) / "preprocess_stats.json",
        {
            "imputation": imputation.as_dict(),
            "clinical_scaler": clinical_scaler.as_dict(),
            "radiomics_scaler": radiomics_scaler.as_dict(),
        },
    )


def fit_freeze_stats(table_path: str | Path, output_dir: str | Path) -> Path:
    table = load_case_table(table_path)
    imputation = fit_imputation(table)
    imputed = apply_imputation(table, imputation)
    clinical_scaler = fit_scaler(imputed, CONTINUOUS_CLINICAL_FEATURES)
    radiomics_scaler = fit_scaler(imputed, SELECTED_RADIOMICS_FEATURES)
    return write_freeze_stats(output_dir, imputation, clinical_scaler, radiomics_scaler)


def prepare_clinical_row(row, package: FreezePackage) -> np.ndarray:
    import pandas as pd

    frame = standardize_clinical_columns(pd.DataFrame([row]))
    frame = apply_imputation(frame, package.imputation)
    frame = apply_scaler(frame, package.clinical_scaler)
    return clinical_vector(frame.iloc[0])


def prepare_radiomics_row(features: dict[str, float], package: FreezePackage) -> np.ndarray:
    return apply_radiomics_scaler(select_frozen_features(features), package.radiomics_scaler)


@torch.no_grad()
def predict_sample(
    sample: DualViewSample,
    models: Sequence[BiAMF],
    device: torch.device,
) -> dict[str, Any]:
    fold_scores: list[float] = []
    for model in models:
        model.eval()
        logits = model(
            sample.standard_image.unsqueeze(0).to(device),
            sample.standard_mask.unsqueeze(0).to(device),
            sample.enlarged_image.unsqueeze(0).to(device),
            sample.enlarged_mask.unsqueeze(0).to(device),
            sample.clinical.unsqueeze(0).to(device),
            sample.radiomics.unsqueeze(0).to(device),
        )
        fold_scores.append(float(torch.sigmoid(logits.view(-1))[0].cpu()))
    probability = float(np.mean(fold_scores)) if fold_scores else float("nan")
    return {
        "case_id": sample.case_id,
        "probability": probability,
        "positive": int(probability >= DEPLOYMENT_THRESHOLD),
        "threshold": DEPLOYMENT_THRESHOLD,
        "fold_scores": fold_scores,
    }


def load_ensemble(package: FreezePackage, device: torch.device) -> list[BiAMF]:
    return [load_biamf(path, map_location=device).to(device) for path in package.fusion_paths]


def predict_volume(
    image: np.ndarray,
    mask: np.ndarray,
    spacing: tuple[float, float, float],
    clinical: np.ndarray,
    radiomics: np.ndarray,
    models: Sequence[BiAMF],
    device: torch.device,
    case_id: str = "case",
    refine_mask: bool = True,
) -> dict[str, Any]:
    if refine_mask:
        mask, seg_info = postprocess_segmentation(mask, spacing)
    else:
        seg_info = {}
    prepared = prepare_dual_view(image, mask, spacing)
    if prepared is None:
        raise ValueError("Dual-view preparation failed because the mask was empty after resampling.")
    sample = DualViewSample(
        standard_image=to_nchw(prepared.standard_image),
        standard_mask=to_nchw(prepared.standard_mask),
        enlarged_image=to_nchw(prepared.enlarged_image),
        enlarged_mask=to_nchw(prepared.enlarged_mask),
        clinical=torch.from_numpy(np.asarray(clinical, dtype=np.float32)),
        radiomics=torch.from_numpy(np.asarray(radiomics, dtype=np.float32)),
        label=torch.tensor(-1.0),
        case_id=case_id,
    )
    result = predict_sample(sample, models, device)
    result["segmentation"] = seg_info
    result["patch_size"] = list(PATCH_SIZE)
    return result


def summarize_scores(case_ids: list[str], labels: list[float], scores: list[float]) -> dict[str, Any]:
    labeled = [(label, score) for label, score in zip(labels, scores) if label in (0.0, 1.0)]
    y = [item[0] for item in labeled]
    p = [item[1] for item in labeled]
    return {
        "n": len(case_ids),
        "auc": safe_auc(y, p) if y else float("nan"),
        "ap": safe_ap(y, p) if y else float("nan"),
        **operating_point_metrics(y, p, DEPLOYMENT_THRESHOLD),
    }


def load_nifti_pair(image_path: str | Path, mask_path: str | Path) -> tuple[np.ndarray, np.ndarray, tuple[float, float, float]]:
    return load_volume(image_path), load_volume(mask_path), load_spacing(image_path)
