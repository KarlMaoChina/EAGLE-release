"""Clinical schema, missing-value handling, and fold-wise scaling."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable

import numpy as np
import pandas as pd

from eagle.spec import (
    CATEGORICAL_CLINICAL_FEATURES,
    CLINICAL_FEATURE_NAMES,
    CLINICAL_FEATURES,
    CONTINUOUS_CLINICAL_FEATURES,
    MISSINGNESS_EXCLUSION_FRACTION,
)


def _normalize_header(value: str) -> str:
    return (
        str(value)
        .strip()
        .lower()
        .replace("−", "-")
        .replace("_", " ")
        .replace("/", " ")
        .replace("-", " ")
    )


def _alias_map() -> dict[str, str]:
    mapping: dict[str, str] = {}
    for feature in CLINICAL_FEATURES:
        mapping[_normalize_header(feature.name)] = feature.name
        for alias in feature.aliases:
            mapping[_normalize_header(alias)] = feature.name
    return mapping


HEADER_MAP = _alias_map()


def standardize_clinical_columns(frame: pd.DataFrame) -> pd.DataFrame:
    renamed = {}
    for column in frame.columns:
        key = _normalize_header(column)
        if key in HEADER_MAP:
            renamed[column] = HEADER_MAP[key]
    return frame.rename(columns=renamed)


def missing_fraction(row: pd.Series, columns: Iterable[str] = CLINICAL_FEATURE_NAMES) -> float:
    values = row[list(columns)]
    return float(values.isna().mean())


def exclude_high_missingness(
    frame: pd.DataFrame,
    threshold: float = MISSINGNESS_EXCLUSION_FRACTION,
) -> pd.DataFrame:
    keep = frame.apply(lambda row: missing_fraction(row) <= threshold, axis=1)
    return frame.loc[keep].copy()


@dataclass
class ImputationState:
    medians: dict[str, float]
    modes: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"medians": self.medians, "modes": self.modes}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ImputationState":
        return cls(
            medians={key: float(value) for key, value in payload["medians"].items()},
            modes={key: float(value) for key, value in payload["modes"].items()},
        )


def fit_imputation(frame: pd.DataFrame) -> ImputationState:
    """Fit medians/modes on a training table. Never uses the outcome column."""
    medians: dict[str, float] = {}
    modes: dict[str, float] = {}
    for name in CONTINUOUS_CLINICAL_FEATURES:
        series = pd.to_numeric(frame[name], errors="coerce")
        medians[name] = float(series.median()) if series.notna().any() else 0.0
    for name in CATEGORICAL_CLINICAL_FEATURES:
        series = pd.to_numeric(frame[name], errors="coerce")
        mode = series.mode(dropna=True)
        modes[name] = float(mode.iloc[0]) if not mode.empty else 0.0
    return ImputationState(medians=medians, modes=modes)


def apply_imputation(frame: pd.DataFrame, state: ImputationState) -> pd.DataFrame:
    out = frame.copy()
    for name, value in state.medians.items():
        if name in out.columns:
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(value)
    for name, value in state.modes.items():
        if name in out.columns:
            out[name] = pd.to_numeric(out[name], errors="coerce").fillna(value)
    return out


@dataclass
class ScalerState:
    means: dict[str, float]
    scales: dict[str, float]

    def as_dict(self) -> dict[str, Any]:
        return {"means": self.means, "scales": self.scales}

    @classmethod
    def from_dict(cls, payload: dict[str, Any]) -> "ScalerState":
        return cls(
            means={key: float(value) for key, value in payload["means"].items()},
            scales={key: float(value) for key, value in payload["scales"].items()},
        )


def fit_scaler(frame: pd.DataFrame, columns: Iterable[str]) -> ScalerState:
    means: dict[str, float] = {}
    scales: dict[str, float] = {}
    for name in columns:
        values = pd.to_numeric(frame[name], errors="coerce").to_numpy(dtype=np.float64)
        finite = values[np.isfinite(values)]
        means[name] = float(np.mean(finite)) if finite.size else 0.0
        std = float(np.std(finite)) if finite.size else 1.0
        scales[name] = std if std > 0 else 1.0
    return ScalerState(means=means, scales=scales)


def apply_scaler(frame: pd.DataFrame, state: ScalerState) -> pd.DataFrame:
    out = frame.copy()
    for name, mean in state.means.items():
        if name not in out.columns:
            continue
        scale = state.scales[name]
        values = pd.to_numeric(out[name], errors="coerce").to_numpy(dtype=np.float32)
        out[name] = ((values - mean) / scale).astype(np.float32)
    return out


def clinical_vector(row: pd.Series) -> np.ndarray:
    return np.asarray([float(row[name]) for name in CLINICAL_FEATURE_NAMES], dtype=np.float32)
