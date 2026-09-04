#!/usr/bin/env python3
"""Post-process saved spatial predictions at multiple MLT--MLAT resolutions.

This utility does not rerun AMT or OVATION-Prime inference. It reuses the
paired predictions written by ``evaluate_spatial_mlt_mlat.py`` and checks
whether the spatial conclusions are stable as the grid is refined.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.spatial_diagnostic_utils import (
    compute_spatial_metrics,
    count_spatial_bins,
    summarize_bin_counts,
    summarize_spatial_metrics,
)

DEFAULT_RESOLUTIONS = ("24x20", "48x40", "80x96")
DEFAULT_MIN_COUNTS = (10, 20, 30, 50)


def resolution_to_bin_sizes(n_mlt, n_mlat, mlat_min=50.0, mlat_max=90.0):
    n_mlt = int(n_mlt)
    n_mlat = int(n_mlat)
    if n_mlt <= 0 or n_mlat <= 0:
        raise ValueError("Grid dimensions must be positive integers")
    if mlat_max <= mlat_min:
        raise ValueError("mlat_max must be greater than mlat_min")
    return 24.0 / n_mlt, (float(mlat_max) - float(mlat_min)) / n_mlat


def parse_resolution(token):
    try:
        left, right = str(token).lower().split("x", 1)
        n_mlt, n_mlat = int(left), int(right)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid resolution {token!r}; expected e.g. 48x40") from exc
    if n_mlt <= 0 or n_mlat <= 0:
        raise ValueError("Resolution dimensions must be positive")
    return n_mlt, n_mlat


def paired_finite_predictions(predictions):
    work = predictions.copy()
    required = ["true_flux", "pred_mlp_total", "pred_ovation"]
    for col in required:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = np.ones(len(work), dtype=bool)
    for col in required:
        mask &= np.isfinite(work[col].to_numpy(dtype=float))
    return work.loc[mask].copy()


def evaluate_resolution(
    predictions,
    n_mlt,
    n_mlat,
    min_counts=DEFAULT_MIN_COUNTS,
    mlat_min=50.0,
    mlat_max=90.0,
    epsilon=1e-6,
):
    mlt_bin_hours, mlat_bin_deg = resolution_to_bin_sizes(
        n_mlt, n_mlat, mlat_min=mlat_min, mlat_max=mlat_max
    )
    paired = paired_finite_predictions(predictions)
    counts = count_spatial_bins(
        paired,
        mlt_bin_hours=mlt_bin_hours,
        mlat_bin_deg=mlat_bin_deg,
        mlat_min=mlat_min,
        mlat_max=mlat_max,
    )
    count_summary = summarize_bin_counts(counts, thresholds=tuple(int(v) for v in min_counts))
    total_points = int(count_summary["total_binned_points"])
    occupied_bins = int(count_summary["occupied_bins"])

    rows = []
    for min_count in min_counts:
        min_count = int(min_count)
        metrics = compute_spatial_metrics(
            paired,
            mlt_bin_hours=mlt_bin_hours,
            mlat_bin_deg=mlat_bin_deg,
            mlat_min=mlat_min,
            mlat_max=mlat_max,
            min_count=min_count,
            epsilon=epsilon,
        )
        summary = summarize_spatial_metrics(metrics)
        valid_bins = int(summary["n_valid_bins"])
        points_in_valid = int(summary["paired_points_in_valid_bins"])
        rows.append(
            {
                "resolution": f"{int(n_mlt)}x{int(n_mlat)}",
                "n_mlt": int(n_mlt),
                "n_mlat": int(n_mlat),
                "mlt_bin_hours": float(mlt_bin_hours),
                "mlat_bin_deg": float(mlat_bin_deg),
                "total_bins": int(n_mlt) * int(n_mlat),
                "occupied_bins": occupied_bins,
                "occupied_fraction": occupied_bins / (int(n_mlt) * int(n_mlat)),
                "count_median": float(count_summary["count_median"]),
                "count_q25": float(count_summary["count_q25"]),
                "count_q75": float(count_summary["count_q75"]),
                "count_min": int(count_summary["count_min"]),
                "count_max": int(count_summary["count_max"]),
                "min_count": min_count,
                "valid_bins": valid_bins,
                "valid_fraction_of_occupied": valid_bins / occupied_bins if occupied_bins else np.nan,
                "points_in_valid_bins": points_in_valid,
                "point_retention_fraction": points_in_valid / total_points if total_points else np.nan,
                "amt_better_bin_fraction": summary["amt_better_bin_fraction"],
                "median_delta_medae": summary["median_delta_medae"],
            }
        )
    return pd.DataFrame(rows)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="outputs/spatial_diagnostic/spatial_predictions.parquet")
    p.add_argument("--output-dir", default="outputs/spatial_resolution_sensitivity")
    p.add_argument("--resolutions", nargs="+", default=list(DEFAULT_RESOLUTIONS))
    p.add_argument("--min-counts", nargs="+", type=int, default=list(DEFAULT_MIN_COUNTS))
    p.add_argument("--mlat-min", type=float, default=50.0)
    p.add_argument("--mlat-max", type=float, default=90.0)
    p.add_argument("--epsilon", type=float, default=1e-6)
    return p


def main(argv=None):
    args = build_parser().parse_args(argv)
    if any(v < 1 for v in args.min_counts):
        raise ValueError("--min-counts values must all be >= 1")
    predictions = pd.read_parquet(args.predictions)
    required = {"mlt", "mlat", "true_flux", "pred_mlp_total", "pred_ovation"}
    missing = sorted(required.difference(predictions.columns))
    if missing:
        raise ValueError(f"Predictions file is missing required columns: {missing}")

    paired = paired_finite_predictions(predictions)
    print(f"Loaded paired predictions: {args.predictions}")
    print(f"Paired finite rows used for all grid comparisons: {len(paired):,}")

    frames = []
    for token in args.resolutions:
        n_mlt, n_mlat = parse_resolution(token)
        mlt_h, mlat_deg = resolution_to_bin_sizes(
            n_mlt, n_mlat, mlat_min=args.mlat_min, mlat_max=args.mlat_max
        )
        print(f"Evaluating {n_mlt}x{n_mlat}: {mlt_h:g} h MLT x {mlat_deg:.6g} deg MLAT")
        frames.append(
            evaluate_resolution(
                paired,
                n_mlt=n_mlt,
                n_mlat=n_mlat,
                min_counts=args.min_counts,
                mlat_min=args.mlat_min,
                mlat_max=args.mlat_max,
                epsilon=args.epsilon,
            )
        )

    results = pd.concat(frames, ignore_index=True)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    output_path = outdir / "spatial_resolution_sensitivity.csv"
    results.to_csv(output_path, index=False)

    display_cols = [
        "resolution",
        "min_count",
        "occupied_bins",
        "valid_bins",
        "valid_fraction_of_occupied",
        "point_retention_fraction",
        "amt_better_bin_fraction",
        "median_delta_medae",
    ]
    print(results[display_cols].to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    print(f"Saved sensitivity table: {output_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
