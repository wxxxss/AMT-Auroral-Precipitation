"""Helpers for the archived standalone OVATION-Prime/auroramaps snapshot."""

from __future__ import annotations

import importlib
import sys
from pathlib import Path

import numpy as np
from scipy.interpolate import RegularGridInterpolator


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


def predict_total_flux(estimators, ref_utc, ec_value, mlat, mlt):
    """Evaluate and interpolate the summed four-channel OP10 energy flux."""
    mlat = np.asarray(mlat, dtype=float)
    mlt = np.asarray(mlt, dtype=float)
    native_mlt = np.linspace(0.0, 24.0, 96)
    mlt_grid, mlat_grid = np.meshgrid(mlt, mlat)
    points = np.stack([mlat_grid.ravel(), mlt_grid.ravel() % 24.0], axis=1)

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

    interp = RegularGridInterpolator(
        (native_mlat, native_mlt), total, bounds_error=False, fill_value=0.0
    )
    return interp(points).reshape(len(mlat), len(mlt))
