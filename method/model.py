"""Auroral Multi-Task (AMT) neural-network architecture.

The architecture matches the model used in the revised manuscript:

* a shared encoder receives the 116-dimensional solar-wind driver vector;
* nine spatial/temporal geometry features bypass the shared encoder;
* four structurally identical regression heads with independent parameters
  predict diffuse-electron, monoenergetic-electron, broadband-electron,
  and ion energy flux in log10 space.
"""

from __future__ import annotations

import torch
from torch import nn


class AMT(nn.Module):
    """Shared solar-wind encoder with four channel-specific regression heads."""

    def __init__(
        self,
        sw_dim: int = 116,
        skip_dim: int = 9,
        hidden_wide: int = 1024,
        hidden_mid: int = 512,
        latent_dim: int = 256,
        head_hidden: int = 128,
        dropout: float = 0.2,
        out_clamp: tuple[float, float] | None = (-6.5, 4.0),
    ) -> None:
        super().__init__()
        self.sw_dim = int(sw_dim)
        self.skip_dim = int(skip_dim)
        self.latent_dim = int(latent_dim)
        self.out_clamp = out_clamp

        self.backbone = nn.Sequential(
            nn.Linear(self.sw_dim, hidden_wide),
            nn.BatchNorm1d(hidden_wide),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_wide, hidden_mid),
            nn.BatchNorm1d(hidden_mid),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_mid, self.latent_dim),
            nn.BatchNorm1d(self.latent_dim),
            nn.GELU(),
        )

        def build_head() -> nn.Sequential:
            return nn.Sequential(
                nn.Linear(self.latent_dim + self.skip_dim, head_hidden),
                nn.BatchNorm1d(head_hidden),
                nn.GELU(),
                nn.Dropout(dropout),
                nn.Linear(head_hidden, head_hidden // 2),
                nn.BatchNorm1d(head_hidden // 2),
                nn.GELU(),
                nn.Linear(head_hidden // 2, 1),
            )

        self.head_diffuse = build_head()
        self.head_mono = build_head()
        self.head_broadband = build_head()
        self.head_ion = build_head()

    def forward(self, x_sw: torch.Tensor, x_skip: torch.Tensor) -> torch.Tensor:
        """Return four log10 energy-flux predictions with shape ``(B, 4)``."""
        if x_sw.ndim != 2 or x_sw.shape[1] != self.sw_dim:
            raise ValueError(
                f"x_sw must have shape (B, {self.sw_dim}); got {tuple(x_sw.shape)}"
            )
        if x_skip.ndim != 2 or x_skip.shape[1] != self.skip_dim:
            raise ValueError(
                f"x_skip must have shape (B, {self.skip_dim}); got {tuple(x_skip.shape)}"
            )
        if x_sw.shape[0] != x_skip.shape[0]:
            raise ValueError("x_sw and x_skip must contain the same batch size")

        sw_latent = self.backbone(x_sw)
        fused = torch.cat([sw_latent, x_skip], dim=1)
        pred = torch.cat(
            [
                self.head_diffuse(fused),
                self.head_mono(fused),
                self.head_broadband(fused),
                self.head_ion(fused),
            ],
            dim=1,
        )
        if self.out_clamp is not None:
            pred = torch.clamp(pred, self.out_clamp[0], self.out_clamp[1])
        return pred
