"""Classification and segmentation metrics used in the methods."""

from __future__ import annotations

from typing import Iterable

import numpy as np
from scipy import ndimage
from sklearn.metrics import average_precision_score, roc_auc_score


def wilson_interval(successes: int, n: int, z: float = 1.96) -> tuple[float, float]:
    if n <= 0:
        return (float("nan"), float("nan"))
    p = successes / n
    denom = 1.0 + z**2 / n
    centre = (p + z**2 / (2 * n)) / denom
    half = (z / denom) * np.sqrt(p * (1 - p) / n + z**2 / (4 * n**2))
    return float(max(0.0, centre - half)), float(min(1.0, centre + half))


def binary_counts(labels: Iterable[float], scores: Iterable[float], threshold: float) -> dict[str, int]:
    y_true = np.asarray(list(labels), dtype=int)
    y_pred = (np.asarray(list(scores)) >= threshold).astype(int)
    tp = int(np.sum((y_true == 1) & (y_pred == 1)))
    tn = int(np.sum((y_true == 0) & (y_pred == 0)))
    fp = int(np.sum((y_true == 0) & (y_pred == 1)))
    fn = int(np.sum((y_true == 1) & (y_pred == 0)))
    return {"tp": tp, "tn": tn, "fp": fp, "fn": fn, "n": int(len(y_true))}


def operating_point_metrics(labels: Iterable[float], scores: Iterable[float], threshold: float) -> dict[str, float]:
    counts = binary_counts(labels, scores, threshold)
    tp, tn, fp, fn = counts["tp"], counts["tn"], counts["fp"], counts["fn"]
    n = counts["n"]

    def ratio(num: int, den: int) -> float:
        return float(num / den) if den else float("nan")

    metrics = {
        "threshold": float(threshold),
        "accuracy": ratio(tp + tn, n),
        "sensitivity": ratio(tp, tp + fn),
        "specificity": ratio(tn, tn + fp),
        "ppv": ratio(tp, tp + fp),
        "npv": ratio(tn, tn + fn),
        **{key: float(value) for key, value in counts.items()},
    }
    for name, successes, total in (
        ("accuracy", tp + tn, n),
        ("sensitivity", tp, tp + fn),
        ("specificity", tn, tn + fp),
        ("ppv", tp, tp + fp),
        ("npv", tn, tn + fn),
    ):
        low, high = wilson_interval(successes, total)
        metrics[f"{name}_ci_low"] = low
        metrics[f"{name}_ci_high"] = high
    return metrics


def safe_auc(labels: Iterable[float], scores: Iterable[float]) -> float:
    y_true = np.asarray(list(labels))
    y_score = np.asarray(list(scores))
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(roc_auc_score(y_true, y_score))


def safe_ap(labels: Iterable[float], scores: Iterable[float]) -> float:
    y_true = np.asarray(list(labels))
    y_score = np.asarray(list(scores))
    if len(set(y_true.tolist())) < 2:
        return float("nan")
    return float(average_precision_score(y_true, y_score))


def youden_threshold(labels: Iterable[float], scores: Iterable[float]) -> float:
    y_true = np.asarray(list(labels), dtype=int)
    y_score = np.asarray(list(scores), dtype=float)
    thresholds = np.unique(y_score)
    if thresholds.size == 0:
        return 0.5
    best_t = float(thresholds[0])
    best_j = -np.inf
    for threshold in thresholds:
        counts = binary_counts(y_true, y_score, float(threshold))
        sensitivity = counts["tp"] / max(counts["tp"] + counts["fn"], 1)
        specificity = counts["tn"] / max(counts["tn"] + counts["fp"], 1)
        youden = sensitivity + specificity - 1.0
        if youden > best_j:
            best_j = youden
            best_t = float(threshold)
    return best_t


def selection_metric(auc: float, ap: float, auc_weight: float = 0.7) -> float:
    auc = 0.0 if np.isnan(auc) else float(auc)
    ap = 0.0 if np.isnan(ap) else float(ap)
    return auc_weight * auc + (1.0 - auc_weight) * ap


def pos_weight(labels: Iterable[float]) -> float:
    values = np.asarray(list(labels), dtype=float)
    n_pos = float(np.sum(values == 1))
    n_neg = float(np.sum(values == 0))
    if n_pos <= 0:
        return 1.0
    return n_neg / n_pos


def dice_coefficient(pred: np.ndarray, truth: np.ndarray) -> float:
    pred_b = pred > 0
    truth_b = truth > 0
    intersection = float(np.sum(pred_b & truth_b))
    denom = float(np.sum(pred_b) + np.sum(truth_b))
    return 0.0 if denom == 0 else 2.0 * intersection / denom


def volumetric_similarity(pred: np.ndarray, truth: np.ndarray) -> float:
    vp = float(np.sum(pred > 0))
    vt = float(np.sum(truth > 0))
    if vp + vt == 0:
        return 1.0
    return 1.0 - abs(vp - vt) / (vp + vt)


def _surface_voxels(mask: np.ndarray) -> np.ndarray:
    binary = mask > 0
    eroded = ndimage.binary_erosion(binary)
    return binary & ~eroded


def average_surface_distance(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred_s = _surface_voxels(pred)
    truth_s = _surface_voxels(truth)
    if not np.any(pred_s) or not np.any(truth_s):
        return float("nan")
    pred_dt = ndimage.distance_transform_edt(~truth_s, sampling=spacing)
    truth_dt = ndimage.distance_transform_edt(~pred_s, sampling=spacing)
    d1 = float(np.mean(pred_dt[pred_s]))
    d2 = float(np.mean(truth_dt[truth_s]))
    return 0.5 * (d1 + d2)


def hausdorff95(pred: np.ndarray, truth: np.ndarray, spacing: tuple[float, float, float]) -> float:
    pred_s = _surface_voxels(pred)
    truth_s = _surface_voxels(truth)
    if not np.any(pred_s) or not np.any(truth_s):
        return float("nan")
    pred_dt = ndimage.distance_transform_edt(~truth_s, sampling=spacing)
    truth_dt = ndimage.distance_transform_edt(~pred_s, sampling=spacing)
    d1 = pred_dt[pred_s]
    d2 = truth_dt[truth_s]
    return float(max(np.percentile(d1, 95), np.percentile(d2, 95)))
