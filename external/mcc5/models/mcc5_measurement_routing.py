"""Generic measurement-location routing used only by explicit external configs."""
from __future__ import annotations

import torch
from torch import nn


class MeasurementLocationFusion(nn.Module):
    def __init__(self, local_group_mode="measurement_correct", context_feature_dim=4,
                 local_hidden_dim=64, unified_hidden_dim=73, fusion_dim=128,
                 system_classes=20, fine_labels=10, coarse_labels=2):
        super().__init__()
        if local_group_mode not in {"measurement_correct", "unified"}:
            raise ValueError(local_group_mode)
        self.local_group_mode = local_group_mode
        self.context_feature_dim = int(context_feature_dim)
        if local_group_mode == "measurement_correct":
            self.motor_encoder = nn.Sequential(nn.Linear(3, local_hidden_dim), nn.GELU())
            self.gearbox_encoder = nn.Sequential(nn.Linear(3, local_hidden_dim), nn.GELU())
            local_dim = 2 * local_hidden_dim
        else:
            self.unified_encoder = nn.Sequential(nn.Linear(6, unified_hidden_dim), nn.GELU())
            local_dim = unified_hidden_dim
        self.fusion = nn.Sequential(nn.Linear(local_dim + self.context_feature_dim, fusion_dim), nn.GELU())
        self.system_head = nn.Linear(fusion_dim, system_classes)
        self.fine_head = nn.Linear(fusion_dim, fine_labels)
        self.coarse_head = nn.Linear(fusion_dim, coarse_labels)

    def forward(self, signal, context=None):
        if signal.ndim != 3 or signal.shape[-1] != 6:
            raise ValueError(f"Expected [B,T,6], got {tuple(signal.shape)}")
        pooled = signal.mean(dim=1)
        if self.local_group_mode == "measurement_correct":
            local = torch.cat([self.motor_encoder(pooled[:, :3]), self.gearbox_encoder(pooled[:, 3:])], dim=-1)
        else:
            local = self.unified_encoder(pooled)
        if self.context_feature_dim:
            if context is None or context.shape[-1] != self.context_feature_dim:
                raise ValueError("Missing or invalid MCC5 context features")
            local = torch.cat([local, context], dim=-1)
        fused = self.fusion(local)
        return {
            "system_logits": self.system_head(fused),
            "fine_logits": self.fine_head(fused),
            "coarse_logits": self.coarse_head(fused),
            "fused": fused,
        }
