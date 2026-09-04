#!/usr/bin/env python3
"""Reproduce the manuscript event-scale hemispheric-power timing comparison.

The default window is the 17 March 2015 St. Patrick's Day storm from 04:00 to
18:00 UT at 5-min cadence. AMT uses its explicit 120-min driver history;
OVATION-Prime uses the standard four-hour weighted Newell coupling driver.
Both models are evaluated on the same MLAT--MLT grid and integrated using the
same spherical area elements. The resulting curves characterize model-response
timing; they are not an independent observational validation of HP accuracy.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.infer_v4_utils import load_amt_model, predict_grid
from evaluation.ovation_driver import build_ovation_weighted_ec, calculate_newell_coupling
from evaluation.ovation_model import (
    load_ovation_module,
    make_flux_estimators,
    predict_total_flux,
)

EARTH_RADIUS_CM = 6.3712e8
ERG_PER_SECOND_TO_GW = 1.0e-16
DEFAULT_START = "2015-03-17 04:00:00"
DEFAULT_END = "2015-03-17 18:00:00"


def spherical_area_elements_cm2(mlat_1d, mlt_1d) -> np.ndarray:
    """Return the area represented by each regular MLAT--MLT grid point."""
    mlat = np.asarray(mlat_1d, dtype=float)
    mlt = np.asarray(mlt_1d, dtype=float)
    if mlat.ndim != 1 or mlt.ndim != 1 or len(mlat) < 2 or len(mlt) < 2:
        raise ValueError("mlat_1d and mlt_1d must be 1-D arrays with at least two points")
    d_mlat = np.deg2rad(float(mlat[1] - mlat[0]))
    d_mlt = float(mlt[1] - mlt[0]) * (2.0 * np.pi / 24.0)
    area_lat = (EARTH_RADIUS_CM**2) * np.cos(np.deg2rad(mlat)) * abs(d_mlat) * abs(d_mlt)
    return np.repeat(area_lat[:, None], len(mlt), axis=1)


def hemispheric_power_gw(total_flux, area_cm2) -> float:
    """Integrate energy flux (erg cm^-2 s^-1) to hemispheric power in GW."""
    flux = np.asarray(total_flux, dtype=float)
    area = np.asarray(area_cm2, dtype=float)
    if flux.shape != area.shape:
        raise ValueError(f"flux shape {flux.shape} does not match area shape {area.shape}")
    return float(np.nansum(flux * area) * ERG_PER_SECOND_TO_GW)


def _instantaneous_newell(row) -> float:
    speed = float(np.sqrt(row["Vx"] ** 2 + row["Vy"] ** 2 + row["Vz"] ** 2))
    return calculate_newell_coupling(float(row["By"]), float(row["Bz"]), speed)


def evaluate_hp_series(
    event_rows: pd.DataFrame,
    all_omni: pd.DataFrame,
    *,
    model,
    scaler_path,
    estimators,
    mlat,
    mlt,
    device="cpu",
) -> pd.DataFrame:
    """Evaluate AMT and OVATION hemispheric power at each event time."""
    rows = event_rows.copy().sort_values("utc").reset_index(drop=True)
    rows["utc"] = pd.to_datetime(rows["utc"])
    ec_ov = build_ovation_weighted_ec(all_omni, rows["utc"])
    area = spherical_area_elements_cm2(mlat, mlt)

    records = []
    for i, row in rows.iterrows():
        utc = pd.Timestamp(row["utc"])
        amt_four = predict_grid(
            model,
            scaler_path,
            row,
            mlat,
            mlt,
            device=device,
        )
        amt_total = np.sum(amt_four, axis=0)
        hp_amt = hemispheric_power_gw(amt_total, area)

        weighted_ec = float(ec_ov[i])
        if np.isfinite(weighted_ec):
            ov_total = predict_total_flux(
                estimators,
                utc.to_pydatetime(),
                weighted_ec,
                mlat,
                mlt,
            )
            hp_ov = hemispheric_power_gw(ov_total, area)
        else:
            hp_ov = float("nan")

        speed = float(np.sqrt(row["Vx"] ** 2 + row["Vy"] ** 2 + row["Vz"] ** 2))
        records.append(
            {
                "utc": utc,
                "Bx": float(row["Bx"]),
                "By": float(row["By"]),
                "Bz": float(row["Bz"]),
                "P_dyn": float(row["P_dyn"]),
                "speed": speed,
                "newell_instantaneous": _instantaneous_newell(row),
                "newell_ovation_4h": weighted_ec,
                "HP_AMT_GW": hp_amt,
                "HP_OVATION_GW": hp_ov,
            }
        )
    return pd.DataFrame.from_records(records)


def plot_hp_series(series: pd.DataFrame, output_path: str | Path) -> None:
    """Render the four-panel event diagnostic used by the manuscript workflow."""
    import matplotlib.pyplot as plt

    time = pd.to_datetime(series["utc"])
    fig, axes = plt.subplots(4, 1, figsize=(10, 10), sharex=True)
    axes[0].plot(time, series["Bz"], label="IMF Bz")
    axes[0].plot(time, series["By"], label="IMF By")
    axes[0].set_ylabel("IMF (nT)")
    axes[0].legend()

    axes[1].plot(time, series["P_dyn"], label="Pdyn")
    axes[1].set_ylabel("Pdyn (nPa)")
    ax_speed = axes[1].twinx()
    ax_speed.plot(time, series["speed"], label="Speed", linestyle="--")
    ax_speed.set_ylabel("Speed (km/s)")

    axes[2].plot(time, series["newell_instantaneous"])
    axes[2].set_ylabel("Newell coupling")

    axes[3].plot(time, series["HP_AMT_GW"], label="AMT")
    axes[3].plot(time, series["HP_OVATION_GW"], label="OVATION-Prime")
    axes[3].set_ylabel("HP (GW)")
    axes[3].set_xlabel("UTC")
    axes[3].legend()

    fig.tight_layout()
    output = Path(output_path)
    output.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--omni-history", required=True, help="5-min OMNI parquet including the AMT 120-min lag columns")
    p.add_argument("--model-path", required=True)
    p.add_argument("--scaler-path", required=True)
    p.add_argument("--snapshot-root", default=None)
    p.add_argument("--start", default=DEFAULT_START)
    p.add_argument("--end", default=DEFAULT_END)
    p.add_argument("--mlat-points", type=int, default=80)
    p.add_argument("--mlt-points", type=int, default=144)
    p.add_argument("--device", default="cpu")
    p.add_argument("--output-dir", default="outputs/hp_timing")
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    all_omni = pd.read_parquet(args.omni_history)
    all_omni["utc"] = pd.to_datetime(all_omni["utc"])
    start = pd.Timestamp(args.start)
    end = pd.Timestamp(args.end)
    event = all_omni.loc[(all_omni["utc"] >= start) & (all_omni["utc"] <= end)].copy()
    if event.empty:
        raise ValueError(f"no OMNI rows found in {start} -- {end}")

    model = load_amt_model(args.model_path, device=args.device)
    ao = load_ovation_module(args.snapshot_root)
    estimators = make_flux_estimators(ao)
    mlat = np.linspace(50.0, 90.0, args.mlat_points)
    mlt = np.linspace(0.0, 24.0, args.mlt_points)

    series = evaluate_hp_series(
        event,
        all_omni,
        model=model,
        scaler_path=args.scaler_path,
        estimators=estimators,
        mlat=mlat,
        mlt=mlt,
        device=args.device,
    )
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    series.to_csv(outdir / "hp_timing_series.csv", index=False)
    plot_hp_series(series, outdir / "hp_timing_storm.png")
    print(f"wrote {len(series)} event rows to {outdir}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
