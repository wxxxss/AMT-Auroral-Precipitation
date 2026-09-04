#!/usr/bin/env python3
"""Render the manuscript MLT--MLAT diagnostic in auroral polar coordinates."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.spatial_diagnostic_utils import (
    compute_spatial_metrics,
    metrics_to_grid,
    summarize_spatial_metrics,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--predictions", default="outputs/spatial_diagnostic/spatial_predictions.parquet")
    p.add_argument("--output-dir", default="outputs/spatial_diagnostic_polar")
    p.add_argument("--mlt-bins", type=int, default=48)
    p.add_argument("--mlat-bins", type=int, default=40)
    p.add_argument("--mlat-min", type=float, default=50.0)
    p.add_argument("--mlat-max", type=float, default=90.0)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--dpi", type=int, default=300)
    return p


def build_polar_edges(n_mlt, n_mlat, mlat_min=50.0, mlat_max=90.0):
    if n_mlt <= 0 or n_mlat <= 0:
        raise ValueError("Grid dimensions must be positive")
    theta_edges = np.linspace(0.0, 2.0 * np.pi, int(n_mlt) + 1)
    radius_edges = np.linspace(0.0, float(mlat_max) - float(mlat_min), int(n_mlat) + 1)
    return theta_edges, radius_edges


def polarize_grid(grid):
    arr = np.asarray(grid, dtype=float)
    if arr.ndim != 2:
        raise ValueError("grid must be two-dimensional")
    return arr[::-1, :]


def radial_tick_spec(mlat_min=50.0, mlat_max=90.0, step_deg=10.0):
    latitudes = np.arange(mlat_max - step_deg, mlat_min - 1e-9, -step_deg)
    return mlat_max - latitudes, [f"{lat:g}°" for lat in latitudes]


def robust_range(values, symmetric=False):
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return (-1.0, 1.0) if symmetric else (0.0, 1.0)
    if symmetric:
        vmax = max(float(np.nanpercentile(np.abs(finite), 98)), 1e-6)
        return -vmax, vmax
    lo, hi = np.nanpercentile(finite, [2, 98])
    if np.isclose(lo, hi):
        hi = lo + 1e-6
    return float(lo), float(hi)


def configure_polar_axis(ax, mlat_min, mlat_max):
    # 0 MLT bottom, 6 right, 12 top, 18 left; high latitude is central.
    ax.set_theta_zero_location("S")
    ax.set_theta_direction(1)
    mlt_ticks = np.array([0.0, 6.0, 12.0, 18.0])
    ax.set_xticks(2.0 * np.pi * mlt_ticks / 24.0)
    ax.set_xticklabels(["0", "6", "12", "18"])
    ax.set_ylim(0.0, float(mlat_max) - float(mlat_min))
    radii, labels = radial_tick_spec(mlat_min, mlat_max)
    ax.set_yticks(radii)
    ax.set_yticklabels(labels)
    ax.set_rlabel_position(22.5)
    ax.grid(True, linewidth=0.6, alpha=0.55)


def plot(metrics, args, save_path):
    import matplotlib.pyplot as plt

    n_mlt, n_mlat = args.mlt_bins, args.mlat_bins
    theta_edges, radius_edges = build_polar_edges(n_mlt, n_mlat, args.mlat_min, args.mlat_max)
    grids = {
        "obs": polarize_grid(metrics_to_grid(metrics, "obs_median_log", n_mlt, n_mlat)),
        "amt": polarize_grid(metrics_to_grid(metrics, "amt_bias_median", n_mlt, n_mlat)),
        "ov": polarize_grid(metrics_to_grid(metrics, "ov_bias_median", n_mlt, n_mlat)),
        "delta": polarize_grid(metrics_to_grid(metrics, "delta_medae", n_mlt, n_mlat)),
    }
    obs_lim = robust_range(grids["obs"])
    bias_lim = robust_range(np.concatenate([grids["amt"].ravel(), grids["ov"].ravel()]), symmetric=True)
    delta_lim = robust_range(grids["delta"], symmetric=True)

    fig, axes = plt.subplots(
        2,
        2,
        figsize=(13.5, 11.5),
        subplot_kw={"projection": "polar"},
        constrained_layout=True,
    )
    panels = [
        (axes[0, 0], grids["obs"], "(a) DMSP/SSJ observed median log flux", "viridis", obs_lim, r"$\log_{10}(E_{\rm tot})$"),
        (axes[0, 1], grids["amt"], "(b) AMT median signed bias", "RdBu_r", bias_lim, "Bias (dex)"),
        (axes[1, 0], grids["ov"], "(c) OVATION-Prime median signed bias", "RdBu_r", bias_lim, "Bias (dex)"),
        (axes[1, 1], grids["delta"], r"(d) Error improvement: MedAE$_{OV}$ - MedAE$_{AMT}$", "RdBu_r", delta_lim, r"$\Delta$MedAE (dex)"),
    ]
    for ax, grid, title, cmap, limits, cbar_label in panels:
        mesh = ax.pcolormesh(
            theta_edges,
            radius_edges,
            grid,
            shading="auto",
            cmap=cmap,
            vmin=limits[0],
            vmax=limits[1],
        )
        configure_polar_axis(ax, args.mlat_min, args.mlat_max)
        ax.set_title(title, pad=22, fontsize=12)
        cbar = fig.colorbar(mesh, ax=ax, pad=0.08, shrink=0.78)
        cbar.set_label(cbar_label)
    fig.savefig(save_path, dpi=args.dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def main(argv=None):
    args = build_parser().parse_args(argv)
    predictions = pd.read_parquet(args.predictions)
    mlt_bin_hours = 24.0 / args.mlt_bins
    mlat_bin_deg = (args.mlat_max - args.mlat_min) / args.mlat_bins
    metrics = compute_spatial_metrics(
        predictions,
        mlt_bin_hours=mlt_bin_hours,
        mlat_bin_deg=mlat_bin_deg,
        mlat_min=args.mlat_min,
        mlat_max=args.mlat_max,
        min_count=args.min_count,
        epsilon=args.epsilon,
    )
    summary = summarize_spatial_metrics(metrics)
    finite = (
        np.isfinite(pd.to_numeric(predictions["true_flux"], errors="coerce"))
        & np.isfinite(pd.to_numeric(predictions["pred_mlp_total"], errors="coerce"))
        & np.isfinite(pd.to_numeric(predictions["pred_ovation"], errors="coerce"))
    )
    summary.update(
        {
            "n_mlt_bins": args.mlt_bins,
            "n_mlat_bins": args.mlat_bins,
            "mlt_bin_hours": mlt_bin_hours,
            "mlat_bin_deg": mlat_bin_deg,
            "min_count": args.min_count,
            "paired_finite_prediction_points": int(finite.sum()),
        }
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    stem = f"polar_spatial_diagnostic_{args.mlt_bins}x{args.mlat_bins}_n{args.min_count}"
    metrics.to_csv(outdir / f"{stem}_metrics.csv", index=False)
    (outdir / f"{stem}_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    plot(metrics, args, outdir / f"{stem}.png")

    print(f"Valid bins              : {summary['n_valid_bins']}")
    print(f"Points in valid bins    : {summary['paired_points_in_valid_bins']:,}")
    print(f"AMT-better bin fraction : {summary['amt_better_bin_fraction']:.3f}")
    print(f"Median (OV-AMT) MedAE   : {summary['median_delta_medae']:.3f} dex")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
