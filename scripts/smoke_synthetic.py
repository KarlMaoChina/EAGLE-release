# -*- coding: utf-8 -*-
"""Synthetic end-to-end smoke: mask cleanup -> dual-view patches -> fit-stats -> infer."""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from eagle.cli import main as cli_main
from eagle.data import FILE_NAMES, load_spacing, load_volume, save_nifti
from eagle.io import write_json
from eagle.metrics import selection_metric
from eagle.models import BiAMF, save_checkpoint
from eagle.runtime import seed_everything
from eagle.spec import (
    CLINICAL_FEATURE_NAMES,
    DEPLOYMENT_THRESHOLD,
    ENLARGED_SPACING_MM,
    N_FOLDS,
    PATCH_SIZE,
    SELECTED_RADIOMICS_FEATURES,
    STANDARD_SPACING_MM,
)


def _make_synthetic_volume(out_dir: Path) -> tuple[Path, Path]:
    rng = np.random.default_rng(42)
    shape = (80, 96, 72)
    spacing = (1.0, 0.8, 0.8)
    image = rng.normal(40.0, 25.0, size=shape).astype(np.float32)
    mask = np.zeros(shape, dtype=np.uint8)
    cz, cy, cx = int(0.34 * shape[0]), int(0.38 * shape[1]), int(0.60 * shape[2])
    zz, yy, xx = np.ogrid[: shape[0], : shape[1], : shape[2]]
    ellipsoid = ((zz - cz) / 10) ** 2 + ((yy - cy) / 14) ** 2 + ((xx - cx) / 12) ** 2 <= 1.0
    mask[ellipsoid] = 1
    image[ellipsoid] += 30.0
    out_dir.mkdir(parents=True, exist_ok=True)
    image_path = out_dir / "ct.nii.gz"
    mask_path = out_dir / "mask.nii.gz"
    save_nifti(image, image_path, spacing)
    save_nifti(mask, mask_path, spacing)
    return image_path, mask_path


def _make_table(path: Path) -> None:
    row = {"case_id": "case_0001", "fold": 0, "label": 1}
    for name in CLINICAL_FEATURE_NAMES:
        row[name] = 1.0 if name == "cholelithiasis" else 50.0
    for name in SELECTED_RADIOMICS_FEATURES:
        row[name] = 0.1
    path.parent.mkdir(parents=True, exist_ok=True)
    pd.DataFrame([row]).to_csv(path, index=False)


def _write_random_freeze(weights_dir: Path) -> None:
    weights_dir.mkdir(parents=True, exist_ok=True)
    extra = {
        "n_clinical": len(CLINICAL_FEATURE_NAMES),
        "n_radiomics": len(SELECTED_RADIOMICS_FEATURES),
        "reduction_factor": 4,
        "hidden_dropout": 0.4,
        "final_dropout": 0.3,
    }
    fusion_dir = weights_dir / "biamf"
    fusion_dir.mkdir(parents=True, exist_ok=True)
    seed_everything(42)
    model = BiAMF(len(CLINICAL_FEATURE_NAMES), len(SELECTED_RADIOMICS_FEATURES))
    first = fusion_dir / "fold_0.pt"
    save_checkpoint(model, first, extra=extra)
    payload = first.read_bytes()
    for fold in range(1, N_FOLDS):
        (fusion_dir / f"fold_{fold}.pt").write_bytes(payload)


def run(work: Path | None = None) -> dict:
    seed_everything(42)
    work = Path(work) if work is not None else (ROOT / "outputs" / "smoke")
    work.mkdir(parents=True, exist_ok=True)
    image_path, mask_path = _make_synthetic_volume(work / "raw")
    table_path = work / "table.csv"
    _make_table(table_path)

    assert cli_main([
        "postprocess-mask",
        "--mask", str(mask_path),
        "--output", str(work / "mask_clean.nii.gz"),
    ]) == 0
    assert cli_main([
        "preprocess",
        "--image", str(image_path),
        "--mask", str(mask_path),
        "--output-dir", str(work / "cases"),
        "--case-id", "case_0001",
    ]) == 0
    assert cli_main(["fit-stats", "--table", str(table_path), "--output-dir", str(work / "weights")]) == 0

    case_dir = work / "cases" / "case_0001"
    shapes = {name: list(load_volume(case_dir / filename).shape) for name, filename in FILE_NAMES.items()}
    assert all(tuple(shape) == PATCH_SIZE for shape in shapes.values()), shapes
    std_spacing = load_spacing(case_dir / FILE_NAMES["standard_image"])
    enl_spacing = load_spacing(case_dir / FILE_NAMES["enlarged_image"])
    assert np.allclose(std_spacing, STANDARD_SPACING_MM), std_spacing
    assert np.allclose(enl_spacing, ENLARGED_SPACING_MM), enl_spacing

    _write_random_freeze(work / "weights")
    assert cli_main([
        "infer",
        "--image", str(image_path),
        "--mask", str(mask_path),
        "--clinical-table", str(table_path),
        "--case-id", "case_0001",
        "--weights", str(work / "weights"),
        "--output", str(work / "pred.json"),
        "--device", "cpu",
    ]) == 0

    pred = json.loads((work / "pred.json").read_text(encoding="utf-8"))
    report = {
        "ok": True,
        "patch_size": list(PATCH_SIZE),
        "shapes": shapes,
        "standard_spacing": list(std_spacing),
        "enlarged_spacing": list(enl_spacing),
        "threshold": DEPLOYMENT_THRESHOLD,
        "probability": pred["probability"],
        "positive": pred["positive"],
        "n_folds": len(pred["fold_scores"]),
        "probability_in_unit_interval": 0.0 <= pred["probability"] <= 1.0,
        "binary_matches_threshold": pred["positive"] == int(pred["probability"] >= DEPLOYMENT_THRESHOLD),
        "selection_metric_example": selection_metric(0.9, 0.5),
    }
    write_json(work / "report.json", report)
    (work / "report.txt").write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


if __name__ == "__main__":
    result = run()
    print(json.dumps(result, indent=2))
