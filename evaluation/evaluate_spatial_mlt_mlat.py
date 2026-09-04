#!/usr/bin/env python3
"""Generate the paired AMT--OVATION MLT--MLAT diagnostic used in Figure 9."""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.infer_v4_utils import load_amt_model, run_amt_4heads
from evaluation.ovation_model import run_ovation_on_points
from evaluation.spatial_diagnostic_utils import (
    compute_spatial_metrics,
    count_spatial_bins,
    summarize_bin_counts,
    summarize_spatial_metrics,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-data", required=True)
    p.add_argument("--ovation-omni", required=True)
    p.add_argument("--model-path")
    p.add_argument("--scaler-path")
    p.add_argument("--output-dir", default="outputs/spatial_diagnostic")
    p.add_argument("--device", default="cpu")
    p.add_argument("--sample-max-utcs", type=int, default=60000)
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--mlt-bin-hours", type=float, default=0.5)
    p.add_argument("--mlat-bin-deg", type=float, default=1.0)
    p.add_argument("--mlat-min", type=float, default=50.0)
    p.add_argument("--mlat-max", type=float, default=90.0)
    p.add_argument("--min-count", type=int, default=20)
    p.add_argument("--epsilon", type=float, default=1e-6)
    p.add_argument("--inventory-only", action="store_true")
    p.add_argument("--reuse-predictions", action="store_true")
    return p


def sample_by_utc(df, max_utcs, seed=42):
    unique = pd.Index(pd.to_datetime(df["utc"]).dropna().unique())
    if len(unique) <= int(max_utcs):
        return df.copy()
    rng = np.random.RandomState(int(seed))
    sampled = rng.choice(unique.to_numpy(), int(max_utcs), replace=False)
    return df[df["utc"].isin(sampled)].copy()


def load_sample(path, max_utcs, seed):
    df = pd.read_parquet(path)
    df["utc"] = pd.to_datetime(df["utc"])
    df["true_flux"] = (
        pd.to_numeric(df["ele_energy_flux"], errors="coerce").fillna(0.0)
        + pd.to_numeric(df["ion_energy_flux"], errors="coerce").fillna(0.0)
    )
    return sample_by_utc(df, max_utcs, seed)


def infer_predictions(sampled, args):
    if not args.model_path or not args.scaler_path:
        raise ValueError("--model-path and --scaler-path are required for inference")
    model = load_amt_model(args.model_path, device=args.device)
    eval_df = run_amt_4heads(
        sampled,
        model,
        args.scaler_path,
        device=args.device,
        pred_prefix="pred_mlp",
    )
    eval_df["pred_mlp_total"] = (
        eval_df["pred_mlp_d"]
        + eval_df["pred_mlp_m"]
        + eval_df["pred_mlp_b"]
        + eval_df["pred_mlp_i"]
    )

    ovation_sw = pd.read_parquet(
        args.ovation_omni, columns=["utc", "Bx", "By", "Bz", "Vx", "Vy", "Vz"]
    )
    ovation_sw["utc"] = pd.to_datetime(ovation_sw["utc"])
    eval_df = run_ovation_on_points(eval_df, ovation_sw)
    keep = [
        c
        for c in [
            "utc",
            "mlat",
            "mlt",
            "src_hemi",
            "true_flux",
            "pred_mlp_total",
            "pred_ovation",
        ]
        if c in eval_df.columns
    ]
    return eval_df[keep].copy()


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    sampled = load_sample(args.test_data, args.sample_max_utcs, args.seed)
    counts = count_spatial_bins(
        sampled,
        mlt_bin_hours=args.mlt_bin_hours,
        mlat_bin_deg=args.mlat_bin_deg,
        mlat_min=args.mlat_min,
        mlat_max=args.mlat_max,
    )
    inventory = summarize_bin_counts(counts, thresholds=(10, 20, 30, 50))
    counts.to_csv(outdir / "spatial_bin_inventory.csv", index=False)
    (outdir / "spatial_inventory_summary.json").write_text(
        json.dumps(inventory, indent=2), encoding="utf-8"
    )

    print(f"Sampled {len(sampled):,} rows from {sampled['utc'].nunique():,} UTCs (seed={args.seed})")
    print(f"Occupied bins: {inventory['occupied_bins']:,}")
    for threshold in (10, 20, 30, 50):
        print(f"Bins with N >= {threshold}: {inventory[f'bins_ge_{threshold}']:,}")
    if args.inventory_only:
        return 0

    predictions_path = outdir / "spatial_predictions.parquet"
    if args.reuse_predictions:
        if not predictions_path.exists():
            raise FileNotFoundError(predictions_path)
        predictions = pd.read_parquet(predictions_path)
    else:
        predictions = infer_predictions(sampled, args)
        predictions.to_parquet(predictions_path, index=False)

    finite = (
        np.isfinite(pd.to_numeric(predictions["true_flux"], errors="coerce"))
        & np.isfinite(pd.to_numeric(predictions["pred_mlp_total"], errors="coerce"))
        & np.isfinite(pd.to_numeric(predictions["pred_ovation"], errors="coerce"))
    )
    metrics = compute_spatial_metrics(
        predictions,
        mlt_bin_hours=args.mlt_bin_hours,
        mlat_bin_deg=args.mlat_bin_deg,
        mlat_min=args.mlat_min,
        mlat_max=args.mlat_max,
        min_count=args.min_count,
        epsilon=args.epsilon,
    )
    metrics.to_csv(outdir / "spatial_metrics.csv", index=False)
    summary = summarize_spatial_metrics(metrics)
    summary.update(
        {
            "sample_max_utcs": int(args.sample_max_utcs),
            "seed": int(args.seed),
            "mlt_bin_hours": float(args.mlt_bin_hours),
            "mlat_bin_deg": float(args.mlat_bin_deg),
            "min_count": int(args.min_count),
            "paired_finite_prediction_points": int(finite.sum()),
        }
    )
    pd.DataFrame([summary]).to_csv(outdir / "spatial_summary.csv", index=False)
    print("=" * 72)
    print(f"Valid bins              : {summary['n_valid_bins']}")
    print(f"Points in valid bins    : {summary['paired_points_in_valid_bins']:,}")
    print(f"AMT-better bin fraction : {summary['amt_better_bin_fraction']:.3f}")
    print(f"Median (OV-AMT) MedAE   : {summary['median_delta_medae']:.3f} dex")
    print("=" * 72)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
