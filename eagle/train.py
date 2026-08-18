"""Training loops for DVSE streams and BiAMF fusion."""

from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np
import torch
import torch.nn.functional as F
from torch.optim import AdamW
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from tqdm import tqdm

from eagle.data import DualViewDataset, collate_samples, load_case_table
from eagle.io import ensure_dir, write_json
from eagle.metrics import pos_weight, safe_ap, safe_auc, selection_metric
from eagle.models import BiAMF, DualViewSpatialEncoder, TabularEncoder, save_checkpoint
from eagle.runtime import resolve_device, seed_everything
from eagle.spec import (
    CLINICAL_FEATURE_NAMES,
    DVSE_BATCH_SIZE,
    DVSE_EARLY_STOPPING,
    DVSE_LEARNING_RATE,
    DVSE_SCHEDULER_PATIENCE,
    DVSE_WEIGHT_DECAY,
    FUSION_BATCH_SIZE,
    FUSION_EARLY_STOPPING,
    FUSION_LEARNING_RATE,
    FUSION_MAX_EPOCHS,
    FUSION_SCHEDULER_PATIENCE,
    FUSION_WEIGHT_DECAY,
    MIN_LEARNING_RATE,
    N_FOLDS,
    SCHEDULER_FACTOR,
    SELECTED_RADIOMICS_FEATURES,
    SELECTION_AUC_WEIGHT,
)


@dataclass
class TrainConfig:
    table_path: Path
    image_root: Path
    output_dir: Path
    device: str | None = None
    num_workers: int = 0
    seed: int = 42


def _loader(dataset: DualViewDataset, batch_size: int, shuffle: bool, num_workers: int, device: torch.device) -> DataLoader:
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=shuffle,
        num_workers=num_workers,
        pin_memory=device.type == "cuda",
        collate_fn=collate_samples,
    )


def _weighted_bce(logits: torch.Tensor, labels: torch.Tensor, weight: float) -> torch.Tensor:
    pos = torch.tensor(weight, device=logits.device, dtype=logits.dtype)
    return F.binary_cross_entropy_with_logits(logits.view(-1), labels.view(-1), pos_weight=pos)


@torch.no_grad()
def _collect_predictions(model: torch.nn.Module, loader: DataLoader, device: torch.device, kind: str) -> dict[str, Any]:
    model.eval()
    scores: list[float] = []
    labels: list[float] = []
    case_ids: list[str] = []
    for batch in loader:
        labels.extend(float(value) for value in batch["label"].tolist())
        case_ids.extend(batch["case_id"])
        if kind == "standard":
            logits = model(batch["standard_image"].to(device), batch["standard_mask"].to(device))
        elif kind == "enlarged":
            logits = model(batch["enlarged_image"].to(device), batch["enlarged_mask"].to(device))
        elif kind == "tabular":
            features = torch.cat([batch["clinical"], batch["radiomics"]], dim=1).to(device)
            logits = model(features)[:, 1]
        else:
            logits = model(
                batch["standard_image"].to(device),
                batch["standard_mask"].to(device),
                batch["enlarged_image"].to(device),
                batch["enlarged_mask"].to(device),
                batch["clinical"].to(device),
                batch["radiomics"].to(device),
            )
        scores.extend(torch.sigmoid(logits.view(-1)).cpu().tolist())
    return {
        "case_id": case_ids,
        "label": labels,
        "score": scores,
        "auc": safe_auc(labels, scores),
        "ap": safe_ap(labels, scores),
    }


def _split_fold(table, fold: int):
    if "fold" not in table.columns:
        raise KeyError("Training tables must include a `fold` column with values 0..4.")
    train = table[table["fold"] != fold].copy()
    val = table[table["fold"] == fold].copy()
    return train, val


def train_dvse_stream(config: TrainConfig, stream: str = "standard") -> dict[str, Any]:
    if stream not in {"standard", "enlarged"}:
        raise ValueError("stream must be 'standard' or 'enlarged'")
    seed_everything(config.seed)
    device = resolve_device(config.device)
    table = load_case_table(config.table_path)
    output = ensure_dir(config.output_dir / f"dvse_{stream}")
    summaries: dict[str, Any] = {}

    for fold in range(N_FOLDS):
        train_df, val_df = _split_fold(table, fold)
        train_loader = _loader(
            DualViewDataset(train_df, config.image_root, augment=True),
            DVSE_BATCH_SIZE,
            True,
            config.num_workers,
            device,
        )
        val_loader = _loader(
            DualViewDataset(val_df, config.image_root, augment=False),
            DVSE_BATCH_SIZE,
            False,
            config.num_workers,
            device,
        )
        model = DualViewSpatialEncoder().to(device)
        optimizer = AdamW(model.parameters(), lr=DVSE_LEARNING_RATE, weight_decay=DVSE_WEIGHT_DECAY)
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=DVSE_SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
        weight = pos_weight(train_df["label"].tolist())
        best = -np.inf
        stale = 0
        best_path = output / f"fold_{fold}.pt"
        for epoch in range(1, FUSION_MAX_EPOCHS + 1):
            model.train()
            losses: list[float] = []
            for batch in tqdm(train_loader, desc=f"DVSE {stream} fold {fold} epoch {epoch}", leave=False):
                if stream == "standard":
                    image, mask = batch["standard_image"].to(device), batch["standard_mask"].to(device)
                else:
                    image, mask = batch["enlarged_image"].to(device), batch["enlarged_mask"].to(device)
                logits = model(image, mask)
                loss = _weighted_bce(logits, batch["label"].to(device), weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                losses.append(float(loss.item()))
            val = _collect_predictions(model, val_loader, device, stream)
            scheduler.step(0.0 if np.isnan(val["auc"]) else val["auc"])
            score = selection_metric(val["auc"], val["ap"], 1.0)
            if score > best:
                best = score
                stale = 0
                save_checkpoint(
                    model,
                    best_path,
                    extra={"fold": fold, "stream": stream, "epoch": epoch, "auc": val["auc"]},
                )
            else:
                stale += 1
            if stale >= DVSE_EARLY_STOPPING:
                break
        summaries[str(fold)] = {"best_score": float(best), "checkpoint": str(best_path)}
    write_json(output / "summary.json", summaries)
    return summaries


def train_tabular(config: TrainConfig) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    table = load_case_table(config.table_path)
    output = ensure_dir(config.output_dir / "tabular")
    n_features = len(CLINICAL_FEATURE_NAMES) + len(SELECTED_RADIOMICS_FEATURES)
    summaries: dict[str, Any] = {}
    for fold in range(N_FOLDS):
        train_df, val_df = _split_fold(table, fold)
        train_loader = _loader(
            DualViewDataset(train_df, config.image_root, augment=False),
            FUSION_BATCH_SIZE,
            True,
            config.num_workers,
            device,
        )
        val_loader = _loader(
            DualViewDataset(val_df, config.image_root, augment=False),
            FUSION_BATCH_SIZE,
            False,
            config.num_workers,
            device,
        )
        model = TabularEncoder(n_features).to(device)
        optimizer = AdamW(model.parameters(), lr=FUSION_LEARNING_RATE, weight_decay=FUSION_WEIGHT_DECAY)
        weight = pos_weight(train_df["label"].tolist())
        best = -np.inf
        stale = 0
        best_path = output / f"fold_{fold}.pt"
        for epoch in range(1, FUSION_MAX_EPOCHS + 1):
            model.train()
            for batch in train_loader:
                features = torch.cat([batch["clinical"], batch["radiomics"]], dim=1).to(device)
                logits = model(features)[:, 1]
                loss = _weighted_bce(logits, batch["label"].to(device), weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val = _collect_predictions(model, val_loader, device, "tabular")
            score = selection_metric(val["auc"], val["ap"], SELECTION_AUC_WEIGHT)
            if score > best:
                best = score
                stale = 0
                save_checkpoint(model, best_path, extra={"fold": fold, "epoch": epoch, "auc": val["auc"]})
            else:
                stale += 1
            if stale >= FUSION_EARLY_STOPPING:
                break
        summaries[str(fold)] = {"best_score": float(best), "checkpoint": str(best_path)}
    write_json(output / "summary.json", summaries)
    return summaries


def train_fusion(config: TrainConfig, extractor_root: str | Path | None = None) -> dict[str, Any]:
    seed_everything(config.seed)
    device = resolve_device(config.device)
    table = load_case_table(config.table_path)
    output = ensure_dir(config.output_dir / "biamf")
    extractor_root = Path(extractor_root) if extractor_root is not None else config.output_dir
    summaries: dict[str, Any] = {}

    for fold in range(N_FOLDS):
        train_df, val_df = _split_fold(table, fold)
        train_loader = _loader(
            DualViewDataset(train_df, config.image_root, augment=False),
            FUSION_BATCH_SIZE,
            True,
            config.num_workers,
            device,
        )
        val_loader = _loader(
            DualViewDataset(val_df, config.image_root, augment=False),
            FUSION_BATCH_SIZE,
            False,
            config.num_workers,
            device,
        )
        model = BiAMF(len(CLINICAL_FEATURE_NAMES), len(SELECTED_RADIOMICS_FEATURES)).to(device)
        model.load_frozen_extractors(
            tabular_path=_optional_ckpt(extractor_root / "tabular" / f"fold_{fold}.pt"),
            enlarged_path=_optional_ckpt(extractor_root / "dvse_enlarged" / f"fold_{fold}.pt"),
            standard_path=_optional_ckpt(extractor_root / "dvse_standard" / f"fold_{fold}.pt"),
            map_location=device,
        )
        optimizer = AdamW(
            [parameter for parameter in model.parameters() if parameter.requires_grad],
            lr=FUSION_LEARNING_RATE,
            weight_decay=FUSION_WEIGHT_DECAY,
        )
        scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=SCHEDULER_FACTOR,
            patience=FUSION_SCHEDULER_PATIENCE,
            min_lr=MIN_LEARNING_RATE,
        )
        weight = pos_weight(train_df["label"].tolist())
        best = -np.inf
        stale = 0
        best_path = output / f"fold_{fold}.pt"
        for epoch in range(1, FUSION_MAX_EPOCHS + 1):
            model.train()
            for batch in tqdm(train_loader, desc=f"BiAMF fold {fold} epoch {epoch}", leave=False):
                logits = model(
                    batch["standard_image"].to(device),
                    batch["standard_mask"].to(device),
                    batch["enlarged_image"].to(device),
                    batch["enlarged_mask"].to(device),
                    batch["clinical"].to(device),
                    batch["radiomics"].to(device),
                )
                loss = _weighted_bce(logits, batch["label"].to(device), weight)
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
            val = _collect_predictions(model, val_loader, device, "fusion")
            score = selection_metric(val["auc"], val["ap"], SELECTION_AUC_WEIGHT)
            scheduler.step(score)
            if score > best:
                best = score
                stale = 0
                save_checkpoint(
                    model,
                    best_path,
                    extra={
                        "fold": fold,
                        "epoch": epoch,
                        "auc": val["auc"],
                        "ap": val["ap"],
                        "n_clinical": len(CLINICAL_FEATURE_NAMES),
                        "n_radiomics": len(SELECTED_RADIOMICS_FEATURES),
                    },
                )
            else:
                stale += 1
            if stale >= FUSION_EARLY_STOPPING:
                break
        summaries[str(fold)] = {"best_score": float(best), "checkpoint": str(best_path)}
    write_json(output / "summary.json", summaries)
    return summaries


def _optional_ckpt(path: Path) -> Path | None:
    return path if path.is_file() else None
