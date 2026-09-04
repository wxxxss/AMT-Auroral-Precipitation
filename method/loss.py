"""Loss functions used for the AMT manuscript model."""

from __future__ import annotations

from collections.abc import Sequence

import torch
from torch import nn


class MultiTaskAsymmetricLoss(nn.Module):
    """Channel-specific asymmetric MSE used for the final AMT model.

    For an active target pixel (log10 flux > ``active_threshold``), an
    underprediction receives an additional channel-specific penalty. Background
    pixels and overpredicted active pixels retain unit weight.

    The manuscript configuration is ``penalties=(5, 50, 50, 10)`` for
    diffuse, monoenergetic, broadband, and ion precipitation, respectively,
    with ``active_threshold=-5``.
    """

    def __init__(
        self,
        penalties: Sequence[float] = (5.0, 50.0, 50.0, 10.0),
        active_threshold: float = -5.0,
    ) -> None:
        super().__init__()
        penalties = tuple(float(v) for v in penalties)
        if len(penalties) != 4:
            raise ValueError("penalties must contain four channel weights")
        if any(v < 0 for v in penalties):
            raise ValueError("penalties must be nonnegative")
        self.register_buffer("penalties", torch.tensor(penalties, dtype=torch.float32))
        self.active_threshold = float(active_threshold)

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        if pred.shape != target.shape:
            raise ValueError(f"pred and target must have identical shape; got {pred.shape} and {target.shape}")
        if pred.ndim != 2 or pred.shape[1] != 4:
            raise ValueError("pred and target must have shape (B, 4)")

        mse = (pred - target) ** 2
        active = (target > self.active_threshold).to(pred.dtype)
        under = (pred < target).to(pred.dtype)
        weights = 1.0 + active * under * self.penalties.to(dtype=pred.dtype)
        return (weights * mse).mean()
