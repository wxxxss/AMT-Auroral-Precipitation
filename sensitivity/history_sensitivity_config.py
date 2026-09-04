"""Shared configuration for the controlled solar-wind history study."""

from data.dataset_v4 import (
    CADENCE_MINUTES,
    SUPPORTED_HISTORY_MINUTES,
    lag_minutes_for_history,
    sw_dim_for_history,
)

__all__ = [
    "CADENCE_MINUTES",
    "SUPPORTED_HISTORY_MINUTES",
    "lag_minutes_for_history",
    "sw_dim_for_history",
]
