"""Shared inference utilities for the final AMT model."""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
import torch

from data.dataset_v4 import AuroraMultiTaskDataset_V4
from method.model import AMT


DEFAULT_MODEL_CONFIG = {
    "sw_dim": 116,
    "skip_dim": 9,
    "hidden_wide": 1024,
    "hidden_mid": 512,
    "latent_dim": 256,
    "head_hidden": 128,
    "dropout": 0.2,
}


def load_amt_model(model_path, device="cpu", config=None):
    """Load a manuscript-compatible AMT checkpoint and return ``eval()`` model."""
    cfg = dict(DEFAULT_MODEL_CONFIG)
    if config:
        cfg.update(config)
    model = AMT(**cfg).to(device)
    state = torch.load(Path(model_path), map_location=device)
    if isinstance(state, dict) and "model_state_dict" in state:
        state = state["model_state_dict"]
    model.load_state_dict(state)
    model.eval()
    return model


def build_inference_dataset(df, scaler_path):
    """Create the final AMT dataset while tolerating absent target columns."""
    work = df.copy()
    for col, default in (
        ("aurora_type", 0),
        ("ele_energy_flux", 0.0),
        ("ion_energy_flux", 0.0),
    ):
        if col not in work.columns:
            work[col] = default
    return AuroraMultiTaskDataset_V4(work, is_train=False, scaler_path=scaler_path)


def predict_batched(model, dataset, device="cpu", batch_size=32768, return_log=False):
    """Predict four AMT channels for an already constructed dataset."""
    n = len(dataset)
    out = np.empty((n, 4), dtype=np.float32)
    model.eval()
    with torch.no_grad():
        for start in range(0, n, batch_size):
            stop = min(start + batch_size, n)
            x_sw = dataset.X_sw_tensor[start:stop].to(device)
            x_skip = dataset.X_skip_tensor[start:stop].to(device)
            log_pred = model(x_sw, x_skip).cpu().numpy()
            if return_log:
                out[start:stop] = log_pred
            else:
                linear = np.power(10.0, log_pred) - 1e-6
                out[start:stop] = np.clip(linear, 0.0, None)
    return out


def run_amt_4heads(
    df,
    model,
    scaler_path,
    device="cpu",
    batch_size=32768,
    pred_prefix="pred_mlp",
):
    """Append four linear-flux AMT predictions to a dataframe copy."""
    work = df.copy()
    dataset = build_inference_dataset(work, scaler_path)
    pred = predict_batched(model, dataset, device=device, batch_size=batch_size)
    for suffix, values in zip(("d", "m", "b", "i"), pred.T):
        work[f"{pred_prefix}_{suffix}"] = values
    return work


def _grid_frame(sw_row, mlat, mlt):
    mlat = np.asarray(mlat, dtype=float)
    mlt = np.asarray(mlt, dtype=float)
    mlt_grid, mlat_grid = np.meshgrid(mlt, mlat)
    n = mlt_grid.size
    base = sw_row.to_dict() if hasattr(sw_row, "to_dict") else dict(sw_row)
    for key in ("mlat", "mlt", "aurora_type", "ele_energy_flux", "ion_energy_flux", "_dt"):
        base.pop(key, None)
    frame = pd.DataFrame({key: np.repeat(value, n) for key, value in base.items()})
    frame["mlat"] = mlat_grid.ravel().astype(np.float32)
    frame["mlt"] = (mlt_grid.ravel() % 24.0).astype(np.float32)
    frame["utc"] = pd.to_datetime(frame["utc"])
    frame["aurora_type"] = np.zeros(n, dtype=np.int8)
    frame["ele_energy_flux"] = np.zeros(n, dtype=np.float32)
    frame["ion_energy_flux"] = np.zeros(n, dtype=np.float32)
    return frame, mlat_grid.shape


def predict_grid(
    model,
    scaler_path,
    sw_row,
    mlat,
    mlt,
    device="cpu",
    batch_size=65536,
):
    """Predict four AMT channels on an MLAT x MLT grid for one driver time."""
    frame, shape = _grid_frame(sw_row, mlat, mlt)
    dataset = build_inference_dataset(frame, scaler_path)
    pred = predict_batched(model, dataset, device=device, batch_size=batch_size)
    return pred.T.reshape(4, *shape)


def predict_grid_multi(
    model,
    scaler_path,
    sw_rows,
    mlat,
    mlt,
    device="cpu",
    batch_size=131072,
):
    """Predict ``K`` driver times on one shared MLAT x MLT grid.

    Returns an array with shape ``(K, 4, n_mlat, n_mlt)``.
    """
    if isinstance(sw_rows, pd.DataFrame):
        rows = [sw_rows.iloc[i] for i in range(len(sw_rows))]
    else:
        rows = list(sw_rows)
    if not rows:
        raise ValueError("sw_rows must not be empty")

    frames = []
    shape = None
    for row in rows:
        frame, this_shape = _grid_frame(row, mlat, mlt)
        if shape is None:
            shape = this_shape
        elif shape != this_shape:
            raise RuntimeError("Grid shape changed while building multi-time inference frame")
        frames.append(frame)

    all_rows = pd.concat(frames, ignore_index=True)
    dataset = build_inference_dataset(all_rows, scaler_path)
    pred = predict_batched(model, dataset, device=device, batch_size=batch_size)
    n_points = int(np.prod(shape))
    return pred.reshape(len(rows), n_points, 4).transpose(0, 2, 1).reshape(len(rows), 4, *shape)
