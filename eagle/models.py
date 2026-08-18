"""DVSE, tabular encoder, and BiAMF fusion."""

from __future__ import annotations

from pathlib import Path
from typing import Any

import torch
from torch import nn
import torch.nn.functional as F

from eagle.spec import (
    ATTENTION_HEADS,
    CHANNEL_SCALE,
    FINAL_DROPOUT,
    HIDDEN_DROPOUT,
    PATCH_SIZE,
    REDUCTION_FACTOR,
    TABULAR_HIDDEN,
    UNIFIED_FEATURE_DIM,
)


class SpatialPriorModulation(nn.Module):
    """Mask-guided attention inserted after residual stages 1–3."""

    def __init__(self, in_channels: int):
        super().__init__()
        reduced = max(1, in_channels // 4)
        self.attention = nn.Sequential(
            nn.Conv3d(in_channels + 1, reduced, kernel_size=1),
            nn.ReLU(inplace=True),
            nn.Conv3d(reduced, 1, kernel_size=1),
            nn.Sigmoid(),
        )

    def forward(self, features: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        resized_mask = F.interpolate(mask, size=features.shape[2:], mode="trilinear", align_corners=False)
        attention_map = self.attention(torch.cat([features, resized_mask], dim=1))
        return features * attention_map


class _BasicBlock3d(nn.Module):
    def __init__(self, in_planes: int, planes: int, stride: int = 1):
        super().__init__()
        self.conv1 = nn.Conv3d(in_planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm3d(planes)
        self.conv2 = nn.Conv3d(planes, planes, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm3d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = None
        if stride != 1 or in_planes != planes:
            self.downsample = nn.Sequential(
                nn.Conv3d(in_planes, planes, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm3d(planes),
            )

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        residual = inputs
        out = self.relu(self.bn1(self.conv1(inputs)))
        out = self.bn2(self.conv2(out))
        if self.downsample is not None:
            residual = self.downsample(inputs)
        return self.relu(out + residual)


def _make_layer(in_planes: int, planes: int, n_blocks: int, stride: int) -> nn.Sequential:
    blocks = [_BasicBlock3d(in_planes, planes, stride=stride)]
    for _ in range(1, n_blocks):
        blocks.append(_BasicBlock3d(planes, planes, stride=1))
    return nn.Sequential(*blocks)


class DualViewSpatialEncoder(nn.Module):
    """ResNet-18-3D with spatial prior modulation (one stream)."""

    def __init__(self, channel_scale: float = CHANNEL_SCALE):
        super().__init__()
        planes = [max(1, int(value * channel_scale)) for value in (64, 128, 256, 512)]
        self.layer0 = nn.Sequential(
            nn.Conv3d(1, planes[0], kernel_size=7, stride=1, padding=3, bias=False),
            nn.BatchNorm3d(planes[0]),
            nn.ReLU(inplace=True),
        )
        self.layer1 = _make_layer(planes[0], planes[0], 2, stride=1)
        self.layer2 = _make_layer(planes[0], planes[1], 2, stride=2)
        self.layer3 = _make_layer(planes[1], planes[2], 2, stride=2)
        self.layer4 = _make_layer(planes[2], planes[3], 2, stride=2)
        self.avgpool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Linear(planes[3], 1)
        self.spm1 = SpatialPriorModulation(planes[0])
        self.spm2 = SpatialPriorModulation(planes[1])
        self.spm3 = SpatialPriorModulation(planes[2])
        self.feature_dim = planes[3]

    def features(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        hidden = self.layer0(image)
        hidden = self.spm1(self.layer1(hidden), mask)
        hidden = self.spm2(self.layer2(hidden), mask)
        hidden = self.spm3(self.layer3(hidden), mask)
        hidden = self.layer4(hidden)
        hidden = self.avgpool(hidden)
        return torch.flatten(hidden, 1)

    def forward(self, image: torch.Tensor, mask: torch.Tensor) -> torch.Tensor:
        return self.fc(self.features(image, mask))

    def attention_map(self, image: torch.Tensor, mask: torch.Tensor, stage: int = 3) -> torch.Tensor:
        hidden = self.layer0(image)
        hidden = self.spm1(self.layer1(hidden), mask)
        if stage == 1:
            resized = F.interpolate(mask, size=hidden.shape[2:], mode="trilinear", align_corners=False)
            return self.spm1.attention(torch.cat([hidden, resized], dim=1))
        hidden = self.spm2(self.layer2(hidden), mask)
        if stage == 2:
            resized = F.interpolate(mask, size=hidden.shape[2:], mode="trilinear", align_corners=False)
            return self.spm2.attention(torch.cat([hidden, resized], dim=1))
        hidden = self.layer3(hidden)
        resized = F.interpolate(mask, size=hidden.shape[2:], mode="trilinear", align_corners=False)
        return self.spm3.attention(torch.cat([hidden, resized], dim=1))


class TabularEncoder(nn.Module):
    """Frozen clinical + radiomics encoder. Intermediate width is 256."""

    def __init__(self, n_features: int):
        super().__init__()
        hidden0, hidden1 = TABULAR_HIDDEN
        self.fc0 = nn.Linear(n_features, hidden0)
        self.fc = nn.Linear(hidden0, hidden1)
        self.fc2 = nn.Linear(hidden1, 2)
        self.dropout1 = nn.Dropout(0.5)
        self.dropout2 = nn.Dropout(0.5)

    def features(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.fc0(values))
        hidden = self.dropout1(hidden)
        return self.fc(hidden)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        hidden = F.relu(self.features(values))
        hidden = self.dropout2(hidden)
        return self.fc2(hidden)


class ResidualBlock(nn.Module):
    def __init__(self, dim: int, dropout: float):
        super().__init__()
        self.block = nn.Sequential(
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(dim, dim),
            nn.LayerNorm(dim),
        )
        self.residual_weight = nn.Parameter(torch.ones(1))

    def forward(self, inputs: torch.Tensor) -> torch.Tensor:
        return inputs + self.residual_weight * self.block(inputs)


class CrossModalAttention(nn.Module):
    def __init__(self, dim: int, n_heads: int = ATTENTION_HEADS):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=n_heads)

    def forward(self, features_list: list[torch.Tensor]) -> torch.Tensor:
        tokens = [features.unsqueeze(0) for features in features_list]
        attended: list[torch.Tensor] = []
        for index, query in enumerate(tokens):
            key_value = torch.cat(
                [token for inner, token in enumerate(tokens) if inner != index],
                dim=0,
            )
            updated, _ = self.attention(query, key_value, key_value)
            attended.append(updated.squeeze(0))
        return torch.cat(attended, dim=1)


class DualDynamicWeighting(nn.Module):
    def __init__(self, dim: int, n_modalities: int = 3):
        super().__init__()
        self.global_weights = nn.ParameterList([nn.Parameter(torch.ones(1, dim)) for _ in range(n_modalities)])
        self.gates = nn.ModuleList(
            [nn.Sequential(nn.Linear(dim, 1), nn.Sigmoid()) for _ in range(n_modalities)]
        )

    def forward(self, features_list: list[torch.Tensor]) -> list[torch.Tensor]:
        weighted: list[torch.Tensor] = []
        for features, weight, gate in zip(features_list, self.global_weights, self.gates):
            updated = features * weight * gate(features)
            updated = F.layer_norm(updated, (features.shape[-1],))
            weighted.append(updated)
        return weighted


def _projection(in_dim: int, out_dim: int, dropout: float) -> nn.Sequential:
    return nn.Sequential(
        nn.Linear(in_dim, out_dim),
        nn.LayerNorm(out_dim),
        nn.GELU(),
        nn.Dropout(dropout),
    )


class BiAMF(nn.Module):
    """Bidirectional adaptive modal fusion of tabular + dual-view image features."""

    expected_shape = PATCH_SIZE

    def __init__(
        self,
        n_clinical: int,
        n_radiomics: int,
        reduction_factor: int = REDUCTION_FACTOR,
        hidden_dropout: float = HIDDEN_DROPOUT,
        final_dropout: float = FINAL_DROPOUT,
        channel_scale: float = CHANNEL_SCALE,
    ):
        super().__init__()
        self.n_clinical = int(n_clinical)
        self.n_radiomics = int(n_radiomics)
        self.hparams = {
            "reduction_factor": reduction_factor,
            "hidden_dropout": hidden_dropout,
            "final_dropout": final_dropout,
            "channel_scale": channel_scale,
        }
        self.tabular = TabularEncoder(n_clinical + n_radiomics)
        self.enlarged_encoder = DualViewSpatialEncoder(channel_scale)
        self.standard_encoder = DualViewSpatialEncoder(channel_scale)
        self._freeze_extractors()

        image_dim = self.enlarged_encoder.feature_dim
        self.tabular_dim = UNIFIED_FEATURE_DIM
        self.enlarged_proj = _projection(image_dim, self.tabular_dim, hidden_dropout)
        self.standard_proj = _projection(image_dim, self.tabular_dim, hidden_dropout)
        self.cross_modal = CrossModalAttention(self.tabular_dim)
        self.weighting = DualDynamicWeighting(self.tabular_dim, n_modalities=3)

        fused_dim = self.tabular_dim * 3
        reduced_dim = fused_dim // reduction_factor
        self.feat_norm = nn.LayerNorm(fused_dim)
        self.dim_reduction = _projection(fused_dim, reduced_dim, hidden_dropout)
        self.head = nn.Sequential(
            ResidualBlock(reduced_dim, dropout=hidden_dropout),
            ResidualBlock(reduced_dim, dropout=hidden_dropout),
            nn.LayerNorm(reduced_dim),
            nn.Dropout(final_dropout),
            nn.Linear(reduced_dim, 1),
        )

    def _freeze_extractors(self) -> None:
        for module in (self.tabular, self.enlarged_encoder, self.standard_encoder):
            for parameter in module.parameters():
                parameter.requires_grad = False
            module.eval()

    def train(self, mode: bool = True):
        super().train(mode)
        self._freeze_extractors()
        return self

    def _check_shape(self, tensor: torch.Tensor, name: str) -> None:
        expected = (tensor.size(0), 1, *self.expected_shape)
        if tuple(tensor.shape) != expected:
            raise ValueError(f"{name} expected {expected}, got {tuple(tensor.shape)}")

    def forward(
        self,
        standard_image: torch.Tensor,
        standard_mask: torch.Tensor,
        enlarged_image: torch.Tensor,
        enlarged_mask: torch.Tensor,
        clinical: torch.Tensor,
        radiomics: torch.Tensor,
    ) -> torch.Tensor:
        self._check_shape(standard_image, "standard_image")
        self._check_shape(enlarged_image, "enlarged_image")
        with torch.no_grad():
            tabular_features = self.tabular.features(torch.cat([clinical, radiomics], dim=1))
            enlarged_features = self.enlarged_encoder.features(enlarged_image, enlarged_mask)
            standard_features = self.standard_encoder.features(standard_image, standard_mask)
        projected = [
            tabular_features,
            self.enlarged_proj(enlarged_features),
            self.standard_proj(standard_features),
        ]
        attended = self.cross_modal(projected)
        chunks = torch.split(attended, self.tabular_dim, dim=1)
        weighted = self.weighting(list(chunks))
        fused = self.feat_norm(torch.cat(weighted, dim=1))
        return self.head(self.dim_reduction(fused))

    def load_frozen_extractors(
        self,
        tabular_path: str | Path | None = None,
        enlarged_path: str | Path | None = None,
        standard_path: str | Path | None = None,
        map_location: str | torch.device = "cpu",
    ) -> None:
        if tabular_path is not None:
            self.tabular.load_state_dict(_load_state_dict(tabular_path, map_location))
        if enlarged_path is not None:
            self.enlarged_encoder.load_state_dict(_load_state_dict(enlarged_path, map_location))
        if standard_path is not None:
            self.standard_encoder.load_state_dict(_load_state_dict(standard_path, map_location))
        self._freeze_extractors()


def _extract_state_dict(payload: Any) -> dict[str, torch.Tensor]:
    if isinstance(payload, nn.Module):
        return payload.state_dict()
    if isinstance(payload, dict):
        for key in ("model_state_dict", "state_dict"):
            if key in payload and isinstance(payload[key], dict):
                return payload[key]
        if all(isinstance(value, torch.Tensor) for value in payload.values()):
            return payload
    raise TypeError(f"Unsupported checkpoint payload: {type(payload)!r}")


def _load_state_dict(path: str | Path, map_location: str | torch.device) -> dict[str, torch.Tensor]:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    return _extract_state_dict(payload)


def save_checkpoint(model: nn.Module, path: str | Path, extra: dict[str, Any] | None = None) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "format": "eagle-v1",
        "model_state_dict": model.state_dict(),
        "extra": extra or {},
    }
    torch.save(payload, path)
    return path


def load_biamf(path: str | Path, map_location: str | torch.device = "cpu") -> BiAMF:
    payload = torch.load(path, map_location=map_location, weights_only=False)
    extra = payload.get("extra", {}) if isinstance(payload, dict) else {}
    model = BiAMF(
        n_clinical=int(extra.get("n_clinical", 12)),
        n_radiomics=int(extra.get("n_radiomics", 32)),
        reduction_factor=int(extra.get("reduction_factor", REDUCTION_FACTOR)),
        hidden_dropout=float(extra.get("hidden_dropout", HIDDEN_DROPOUT)),
        final_dropout=float(extra.get("final_dropout", FINAL_DROPOUT)),
    )
    model.load_state_dict(_extract_state_dict(payload))
    model.eval()
    return model
