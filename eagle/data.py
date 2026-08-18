"""Case-level dataset with a public layout and DVSE augmentations."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset

from eagle.clinical import clinical_vector, standardize_clinical_columns
from eagle.io import read_table
from eagle.spec import CLINICAL_FEATURE_NAMES, PATCH_SIZE, SELECTED_RADIOMICS_FEATURES

CASE_ID_CANDIDATES = ("case_id", "id")
LABEL_CANDIDATES = ("label", "y")
SPLIT_CANDIDATES = ("fold", "split")
FILE_NAMES = {
    "standard_image": "standard_image.nii.gz",
    "standard_mask": "standard_mask.nii.gz",
    "enlarged_image": "enlarged_image.nii.gz",
    "enlarged_mask": "enlarged_mask.nii.gz",
}


def _first_present(columns: Sequence[str], candidates: Sequence[str]) -> str:
    lookup = {str(column).strip().lower(): column for column in columns}
    for name in candidates:
        if name in lookup:
            return lookup[name]
    raise KeyError(f"Expected one of {candidates} in columns {list(columns)}")


def load_case_table(path: str | Path) -> pd.DataFrame:
    frame = standardize_clinical_columns(read_table(path))
    case_col = _first_present(frame.columns, CASE_ID_CANDIDATES)
    frame = frame.rename(columns={case_col: "case_id"})
    frame["case_id"] = frame["case_id"].astype("string").str.strip()
    if any(name in {str(col).lower() for col in frame.columns} for name in LABEL_CANDIDATES):
        label_col = _first_present(frame.columns, LABEL_CANDIDATES)
        frame = frame.rename(columns={label_col: "label"})
    else:
        frame["label"] = -1
    if any(name in {str(col).lower() for col in frame.columns} for name in SPLIT_CANDIDATES):
        split_col = _first_present(frame.columns, SPLIT_CANDIDATES)
        frame = frame.rename(columns={split_col: "fold"})
    return frame


def resolve_case_dir(image_root: str | Path, case_id: str) -> Path:
    return Path(image_root) / str(case_id)


def case_paths(image_root: str | Path, case_id: str) -> dict[str, Path]:
    folder = resolve_case_dir(image_root, case_id)
    return {key: folder / filename for key, filename in FILE_NAMES.items()}


def load_volume(path: str | Path) -> np.ndarray:
    import nibabel as nib

    return np.asarray(nib.load(str(path)).get_fdata(), dtype=np.float32)


def load_spacing(path: str | Path) -> tuple[float, float, float]:
    import nibabel as nib

    zooms = nib.load(str(path)).header.get_zooms()[:3]
    return tuple(float(value) for value in zooms)


def save_nifti(
    data: np.ndarray,
    path: str | Path,
    spacing: tuple[float, float, float],
    affine: np.ndarray | None = None,
) -> Path:
    import nibabel as nib

    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    if affine is None:
        affine = np.diag([spacing[0], spacing[1], spacing[2], 1.0])
    image = nib.Nifti1Image(np.asarray(data, dtype=np.float32), affine)
    image.header.set_zooms(spacing)
    nib.save(image, str(target))
    return target


def to_nchw(volume: np.ndarray) -> torch.Tensor:
    tensor = torch.from_numpy(np.asarray(volume, dtype=np.float32))
    if tensor.ndim != 3:
        raise ValueError(f"Expected a 3D volume, got shape {tuple(tensor.shape)}")
    return tensor.unsqueeze(0)


class CoarseDropout:
    def __init__(self, probability: float = 0.2, n_holes: tuple[int, int] = (1, 3), hole_size: int = 3):
        self.probability = probability
        self.n_holes = n_holes
        self.hole_size = hole_size

    def __call__(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if np.random.rand() > self.probability:
            return image, mask
        image = image.copy()
        mask = mask.copy()
        n_holes = int(np.random.randint(self.n_holes[0], self.n_holes[1] + 1))
        for _ in range(n_holes):
            z = int(np.random.randint(0, image.shape[0]))
            y = int(np.random.randint(0, image.shape[1]))
            x = int(np.random.randint(0, image.shape[2]))
            z0, z1 = max(0, z - self.hole_size), min(image.shape[0], z + self.hole_size)
            y0, y1 = max(0, y - self.hole_size), min(image.shape[1], y + self.hole_size)
            x0, x1 = max(0, x - self.hole_size), min(image.shape[2], x + self.hole_size)
            image[z0:z1, y0:y1, x0:x1] = 0
            mask[z0:z1, y0:y1, x0:x1] = 0
        return image, mask


class IntensityAugment:
    def __init__(self, strength: float = 0.8):
        self.strength = strength

    def __call__(self, image: np.ndarray) -> np.ndarray:
        out = image.astype(np.float32, copy=True)
        if np.random.rand() < 0.5:
            factor = 1.0 + (np.random.uniform(-0.10, 0.10) * self.strength)
            mean = float(out.mean())
            out = (out - mean) * factor + mean
        if np.random.rand() < 0.5:
            out = out + np.random.normal(0.0, 0.05 * self.strength, size=out.shape).astype(np.float32)
        if np.random.rand() < 0.5:
            from scipy.ndimage import gaussian_filter

            sigma = float(np.random.uniform(0.25, 1.0) * self.strength)
            out = gaussian_filter(out, sigma=sigma)
        if np.random.rand() < 0.5:
            out = out * (1.0 + np.random.uniform(-0.15, 0.15) * self.strength)
        if np.random.rand() < 0.2:
            out = out + float(np.random.uniform(-0.05, 0.05) * self.strength)
        return out.astype(np.float32)


@dataclass
class DualViewSample:
    standard_image: torch.Tensor
    standard_mask: torch.Tensor
    enlarged_image: torch.Tensor
    enlarged_mask: torch.Tensor
    clinical: torch.Tensor
    radiomics: torch.Tensor
    label: torch.Tensor
    case_id: str


class DualViewDataset(Dataset):
    def __init__(
        self,
        table: pd.DataFrame,
        image_root: str | Path,
        augment: bool = False,
        radiomics_columns: Sequence[str] = SELECTED_RADIOMICS_FEATURES,
    ):
        self.image_root = Path(image_root)
        self.augment = augment
        self.radiomics_columns = list(radiomics_columns)
        self.dropout = CoarseDropout()
        self.intensity = IntensityAugment()
        self.table = table.reset_index(drop=True)
        missing = [name for name in CLINICAL_FEATURE_NAMES if name not in self.table.columns]
        missing += [name for name in self.radiomics_columns if name not in self.table.columns]
        if missing:
            raise KeyError(f"Case table is missing required columns: {missing}")
        needed = list(CLINICAL_FEATURE_NAMES) + list(self.radiomics_columns)
        self.table = table.dropna(subset=needed).reset_index(drop=True)
        self.table = self._keep_complete_cases(self.table)
        if self.table.empty:
            raise ValueError("No complete cases were found under the configured image root.")

    def _keep_complete_cases(self, table: pd.DataFrame) -> pd.DataFrame:
        keep: list[int] = []
        for index, row in table.iterrows():
            paths = case_paths(self.image_root, str(row["case_id"]))
            if all(path.is_file() for path in paths.values()):
                keep.append(index)
        return table.loc[keep].reset_index(drop=True)

    def __len__(self) -> int:
        return len(self.table)

    def _maybe_augment(self, image: np.ndarray, mask: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
        if not self.augment:
            return image, mask
        image, mask = self.dropout(image, mask)
        return self.intensity(image), mask

    def __getitem__(self, index: int) -> DualViewSample:
        row = self.table.iloc[index]
        paths = case_paths(self.image_root, str(row["case_id"]))
        standard_image = load_volume(paths["standard_image"])
        standard_mask = load_volume(paths["standard_mask"])
        enlarged_image = load_volume(paths["enlarged_image"])
        enlarged_mask = load_volume(paths["enlarged_mask"])
        if tuple(standard_image.shape) != PATCH_SIZE:
            raise ValueError(
                f"Case {row['case_id']} standard image has shape {standard_image.shape}, expected {PATCH_SIZE}"
            )
        standard_image, standard_mask = self._maybe_augment(standard_image, standard_mask)
        enlarged_image, enlarged_mask = self._maybe_augment(enlarged_image, enlarged_mask)
        return DualViewSample(
            standard_image=to_nchw(standard_image),
            standard_mask=to_nchw(standard_mask),
            enlarged_image=to_nchw(enlarged_image),
            enlarged_mask=to_nchw(enlarged_mask),
            clinical=torch.from_numpy(clinical_vector(row)),
            radiomics=torch.from_numpy(
                np.asarray([float(row[name]) for name in self.radiomics_columns], dtype=np.float32)
            ),
            label=torch.tensor(float(row["label"]), dtype=torch.float32),
            case_id=str(row["case_id"]),
        )


def collate_samples(batch: list[DualViewSample]) -> dict[str, torch.Tensor | list[str]]:
    return {
        "standard_image": torch.stack([item.standard_image for item in batch], dim=0),
        "standard_mask": torch.stack([item.standard_mask for item in batch], dim=0),
        "enlarged_image": torch.stack([item.enlarged_image for item in batch], dim=0),
        "enlarged_mask": torch.stack([item.enlarged_mask for item in batch], dim=0),
        "clinical": torch.stack([item.clinical for item in batch], dim=0),
        "radiomics": torch.stack([item.radiomics for item in batch], dim=0),
        "label": torch.stack([item.label for item in batch], dim=0),
        "case_id": [item.case_id for item in batch],
    }
