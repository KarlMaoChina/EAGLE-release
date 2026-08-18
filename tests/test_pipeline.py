from __future__ import annotations

import numpy as np
import pandas as pd
import pytest
import torch

from eagle.clinical import apply_imputation, clinical_vector, fit_imputation, standardize_clinical_columns
from eagle.metrics import operating_point_metrics, selection_metric, wilson_interval, youden_threshold
from eagle.models import BiAMF, CrossModalAttention, DualDynamicWeighting, DualViewSpatialEncoder, SpatialPriorModulation, TabularEncoder
from eagle.preprocess import DualViewCase, enhance_enlarged_mask, enhance_standard_mask, prepare_dual_view
from eagle.segmentation import postprocess_segmentation
from eagle.spec import (
    CLINICAL_FEATURE_NAMES,
    DEPLOYMENT_THRESHOLD,
    N_RADIOMICS_SELECTED,
    PATCH_SIZE,
    SELECTED_RADIOMICS_FEATURES,
)


def test_spec_counts() -> None:
    assert len(CLINICAL_FEATURE_NAMES) == 12
    assert len(SELECTED_RADIOMICS_FEATURES) == N_RADIOMICS_SELECTED == 32
    assert DEPLOYMENT_THRESHOLD == 0.5
    assert PATCH_SIZE == (96, 112, 80)


def test_clinical_header_aliases_are_english_only() -> None:
    frame = pd.DataFrame(
        {
            "History of cholelithiasis": [1],
            "Age at diagnosis": [60],
            "A/G Ratio": [1.4],
            "ALP": [90],
            "PA": [200],
            "DBIL": [4.0],
            "TBIL": [12.0],
            "CEA": [2.0],
            "AFP": [3.0],
            "CA19-9": [20.0],
            "CA15-3": [10.0],
            "CA125": [15.0],
        }
    )
    mapped = standardize_clinical_columns(frame)
    vector = clinical_vector(mapped.iloc[0])
    assert vector.shape == (12,)
    assert mapped.loc[0, "cholelithiasis"] == 1
    assert mapped.loc[0, "prealbumin"] == 200


def test_imputation_ignores_label() -> None:
    frame = pd.DataFrame(
        {
            "cholelithiasis": [1, 0, np.nan],
            "age": [60.0, 70.0, np.nan],
            "ag_ratio": [1.4, 1.2, 1.3],
            "alp": [80.0, 100.0, np.nan],
            "prealbumin": [220.0, 180.0, 200.0],
            "dbil": [3.0, 5.0, 4.0],
            "tbil": [10.0, 14.0, 12.0],
            "cea": [1.0, 3.0, 2.0],
            "afp": [2.0, 4.0, 3.0],
            "ca19_9": [10.0, 30.0, 20.0],
            "ca15_3": [8.0, 12.0, 10.0],
            "ca125": [11.0, 19.0, 15.0],
            "label": [0, 1, 1],
        }
    )
    state = fit_imputation(frame)
    filled = apply_imputation(frame, state)
    assert filled.loc[2, "age"] == pytest.approx(65.0)
    assert filled.loc[2, "cholelithiasis"] in (0.0, 1.0)
    assert "label" not in state.medians
    assert "label" not in state.modes


def test_postprocess_drops_off_anatomy_keeps_nearby() -> None:
    volume = np.zeros((80, 80, 80), dtype=np.uint8)
    spacing = (1.0, 1.0, 1.0)
    cz, cy, cx = int(0.34 * 80), int(0.38 * 80), int(0.60 * 80)
    volume[cz - 6 : cz + 6, cy - 6 : cy + 6, cx - 6 : cx + 6] = 1
    volume[cz + 8 : cz + 11, cy - 2 : cy + 2, cx - 2 : cx + 2] = 1
    volume[2:8, 70:78, 4:10] = 1
    cleaned, info = postprocess_segmentation(volume, spacing)
    assert info["original_regions"] >= 2
    assert cleaned[2:8, 70:78, 4:10].sum() == 0
    assert cleaned[cz - 6 : cz + 6, cy - 6 : cy + 6, cx - 6 : cx + 6].sum() > 0


def test_dual_view_shapes_and_mask_weights() -> None:
    image = np.zeros((64, 64, 48), dtype=np.float32)
    mask = np.zeros((64, 64, 48), dtype=np.uint8)
    image[20:44, 20:44, 12:36] = 80.0
    mask[22:42, 22:42, 14:34] = 1
    prepared = prepare_dual_view(image, mask, spacing=(1.0, 1.0, 2.0))
    assert isinstance(prepared, DualViewCase)
    assert prepared.standard_image.shape == PATCH_SIZE
    assert prepared.enlarged_image.shape == PATCH_SIZE
    assert prepared.standard_mask.max() <= 1.0 + 1e-6
    assert prepared.enlarged_mask.max() <= 1.0 + 1e-6
    wall = enhance_standard_mask(mask.astype(np.float32), (1.0, 1.0, 2.0))
    enlarged = enhance_enlarged_mask(mask.astype(np.float32), (1.0, 1.0, 2.0))
    assert wall.max() == pytest.approx(1.0)
    assert enlarged.max() == pytest.approx(1.0)


def test_metrics_threshold_and_wilson() -> None:
    labels = [0, 0, 1, 1]
    scores = [0.1, 0.2, 0.8, 0.9]
    metrics = operating_point_metrics(labels, scores, 0.5)
    assert metrics["tp"] == 2
    assert metrics["tn"] == 2
    low, high = wilson_interval(2, 2)
    assert low > 0.1
    assert high == pytest.approx(1.0)
    assert youden_threshold(labels, scores) >= 0.2
    assert selection_metric(0.9, 0.5) == pytest.approx(0.7 * 0.9 + 0.3 * 0.5)


def test_fusion_modules_cpu() -> None:
    spm = SpatialPriorModulation(8)
    features = torch.randn(2, 8, 4, 4, 4)
    mask = torch.ones(2, 1, 8, 8, 8)
    out = spm(features, mask)
    assert out.shape == features.shape

    tabular = TabularEncoder(44)
    hidden = tabular.features(torch.randn(3, 44))
    assert hidden.shape == (3, 256)

    attention = CrossModalAttention(8)
    fused = attention([torch.randn(2, 8), torch.randn(2, 8), torch.randn(2, 8)])
    assert fused.shape == (2, 24)

    weighting = DualDynamicWeighting(8, n_modalities=3)
    weighted = weighting([torch.randn(2, 8) for _ in range(3)])
    assert len(weighted) == 3


def test_biamf_forward_shape() -> None:
    model = BiAMF(n_clinical=12, n_radiomics=32)
    model.eval()
    with torch.no_grad():
        logits = model(
            torch.randn(1, 1, *PATCH_SIZE),
            torch.rand(1, 1, *PATCH_SIZE),
            torch.randn(1, 1, *PATCH_SIZE),
            torch.rand(1, 1, *PATCH_SIZE),
            torch.randn(1, 12),
            torch.randn(1, 32),
        )
    assert logits.shape == (1, 1)


def test_dvse_pooled_dim_is_204() -> None:
    encoder = DualViewSpatialEncoder()
    assert encoder.feature_dim == 204
    hidden = encoder.features(torch.randn(1, 1, *PATCH_SIZE), torch.rand(1, 1, *PATCH_SIZE))
    assert hidden.shape == (1, 204)


def test_radiomics_settings_do_not_renormalize() -> None:
    from eagle.radiomics import pyradiomics_settings

    settings = pyradiomics_settings()
    assert settings["normalize"] is False
    assert settings["binWidth"] == 5


def test_evaluate_cli(tmp_path) -> None:
    from eagle.cli import main as cli_main
    from eagle.io import read_json

    table = tmp_path / "scores.csv"
    table.write_text("case_id,label,probability\ncase_0001,0,0.1\ncase_0002,1,0.9\n", encoding="utf-8")
    output = tmp_path / "metrics.json"
    assert cli_main(["evaluate", "--table", str(table), "--output", str(output)]) == 0
    metrics = read_json(output)
    assert metrics["tp"] == 1
    assert metrics["tn"] == 1
    assert metrics["threshold"] == 0.5
