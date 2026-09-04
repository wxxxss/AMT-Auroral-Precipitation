"""Helpers for the archived standalone OVATION-Prime/auroramaps snapshot."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
import pandas as pd
from scipy.interpolate import RegularGridInterpolator

from evaluation.ovation_driver import build_ovation_weighted_ec


def load_ovation_module(snapshot_root=None):
    """Import ``auroramaps.ovation`` from the archived third-party snapshot."""
    if snapshot_root is None:
        snapshot_root = Path(__file__).resolve().parents[1] / "third_party" / "auroramaps_op10"
    snapshot_root = Path(snapshot_root).resolve()
    if not (snapshot_root / "auroramaps" / "ovation.py").exists():
        raise FileNotFoundError(
            f"Archived OVATION snapshot not found under {snapshot_root}. "
            "See third_party/auroramaps_op10/README.md."
        )
    if str(snapshot_root) not in sys.path:
        sys.path.insert(0, str(snapshot_root))
    return importlib.import_module("auroramaps.ovation")


def make_flux_estimators(ovation_module=None):
    """Create the four OP10 energy-flux estimators used in the manuscript."""
    ao = ovation_module or load_ovation_module()
    return (
        ao.FluxEstimator("diff", "electron energy flux"),
        ao.FluxEstimator("mono", "electron energy flux"),
        ao.FluxEstimator("wave", "electron energy flux"),
        ao.FluxEstimator("ions", "ion energy flux"),
    )


def _native_total_flux(estimators, ref_utc, ec_value):
    total = None
    native_mlat = None
    for estimator in estimators:
        mlat_2d, _, flux = estimator.get_flux_for_time(ref_utc, float(ec_value))
        this_mlat = np.asarray(mlat_2d[:, 0], dtype=float)
        flux = np.asarray(flux, dtype=float)
        if not np.all(np.diff(this_mlat) > 0):
            order = np.argsort(this_mlat)
            this_mlat = this_mlat[order]
            flux = flux[order, :]
        if total is None:
            total = flux.copy()
            native_mlat = this_mlat
        else:
            total += flux
    return native_mlat, total


def predict_total_flux(estimators, ref_utc, ec_value, mlat, mlt):
    """Evaluate and interpolate the summed four-channel OP10 energy flux."""
    mlat = np.asarray(mlat, dtype=float)
    mlt = np.asarray(mlt, dtype=float)
    native_mlt = np.linspace(0.0, 24.0, 96)
    mlt_grid, mlat_grid = np.meshgrid(mlt, mlat)
    points = np.stack([mlat_grid.ravel(), mlt_grid.ravel() % 24.0], axis=1)
    native_mlat, total = _native_total_flux(estimators, ref_utc, ec_value)
    interp = RegularGridInterpolator(
        (native_mlat, native_mlt), total, bounds_error=False, fill_value=0.0
    )
    return interp(points).reshape(len(mlat), len(mlt))


def run_ovation_on_points(df, ovation_sw_df, estimators=None):
    """Append corrected four-hour OVATION total flux to paired SSJ locations.

    The four-hour solar-wind driver is always evaluated at the real UTC. For
    Southern-Hemisphere samples that were folded into the Northern-Hemisphere
    representation, ``utc + 182 days`` is used only for OP10's seasonal phase;
    it does not shift the solar-wind history.
    """
    required = {"utc", "mlat", "mlt"}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"Evaluation dataframe missing columns: {missing}")
    estimators = estimators or make_flux_estimators()

    out = df.copy()
    out["pred_ovation"] = np.nan
    out["_utc_real"] = pd.to_datetime(out["utc"])
    unique_real = pd.DatetimeIndex(out["_utc_real"].drop_duplicates().sort_values())
    weighted_ec = build_ovation_weighted_ec(ovation_sw_df, unique_real)
    out["_ec_ov"] = out["_utc_real"].map(pd.Series(weighted_ec, index=unique_real))

    out["_utc_eq"] = out["_utc_real"]
    if "src_hemi" in out.columns:
        south = out["src_hemi"].astype(str).str.upper().eq("S")
        out.loc[south, "_utc_eq"] = (
            out.loc[south, "_utc_real"] + pd.Timedelta(days=182)
        ).to_numpy()

    native_mlt = np.linspace(0.0, 24.0, 96)
    for (_, utc_eq), group in out.groupby(["_utc_real", "_utc_eq"], sort=False):
        ec_value = float(group["_ec_ov"].iloc[0])
        if not np.isfinite(ec_value):
            continue
        native_mlat, total = _native_total_flux(estimators, pd.Timestamp(utc_eq).to_pydatetime(), ec_value)
        interp = RegularGridInterpolator(
            (native_mlat, native_mlt), total, bounds_error=False, fill_value=0.0
        )
        points = group[["mlat", "mlt"]].to_numpy(dtype=float)
        points[:, 0] = np.abs(points[:, 0])
        points[:, 1] = np.mod(points[:, 1], 24.0)
        out.loc[group.index, "pred_ovation"] = interp(points)

    return out.drop(columns=["_utc_real", "_utc_eq", "_ec_ov"])
