#!/usr/bin/env python3
"""Reproduce the manuscript pixel-wise AMT--OVATION-Prime comparison.

The comparison starts from the full held-out 2014 folded DMSP/SSJ test set.
Activity groups follow Section 4.2.1 exactly: Storm (Bz < -10 nT or Pdyn >
5 nPa), Quiet (Bz >= -2 nT and Pdyn <= 3 nPa), and All (no activity filter).
Independent UTCs are sampled within each group and all SSJ records sharing a
selected UTC are retained. The default limits are 30,000 unique UTCs for Storm
and Quiet and 60,000 for All.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.infer_v4_utils import load_amt_model, run_amt_4heads
from evaluation.ovation_model import (
    load_ovation_module,
    make_flux_estimators,
    run_ovation_on_points,
)

FLUX_FLOOR = 1.0e-6


def activity_mask(df: pd.DataFrame, group: str) -> pd.Series:
    """Return the instantaneous-solar-wind activity mask from Section 4.2.1."""
    if not {"Bz", "P_dyn"}.issubset(df.columns):
        raise ValueError("evaluation dataframe must contain Bz and P_dyn")
    name = str(group).strip().lower()
    if name == "storm":
        return (df["Bz"] < -10.0) | (df["P_dyn"] > 5.0)
    if name == "quiet":
        return (df["Bz"] >= -2.0) & (df["P_dyn"] <= 3.0)
    if name == "all":
        return pd.Series(True, index=df.index, dtype=bool)
    raise ValueError("group must be Storm, Quiet, or All")


def sample_group_by_unique_utc(
    df: pd.DataFrame,
    group: str,
    *,
    max_unique_times: int,
    seed: int = 42,
) -> pd.DataFrame:
    """Sample UTCs without replacement, then retain every SSJ row at those UTCs."""
    work = df.loc[activity_mask(df, group)].copy()
    work["utc"] = pd.to_datetime(work["utc"])
    unique = np.asarray(sorted(work["utc"].dropna().unique()))
    if max_unique_times <= 0:
        raise ValueError("max_unique_times must be positive")
    if len(unique) > int(max_unique_times):
        rng = np.random.default_rng(int(seed))
        selected = rng.choice(unique, size=int(max_unique_times), replace=False)
    else:
        selected = unique
    return work.loc[work["utc"].isin(selected)].sort_values("utc").reset_index(drop=True)


def regression_metrics_log10(observed, predicted, *, floor: float = FLUX_FLOOR) -> dict:
    """Pearson R, RMSE, and prediction efficiency in log10 total-flux space."""
    obs = np.asarray(observed, dtype=float)
    pred = np.asarray(predicted, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred) & (obs >= 0.0) & (pred >= 0.0)
    obs = np.log10(obs[valid] + float(floor))
    pred = np.log10(pred[valid] + float(floor))
    n = len(obs)
    if n < 2:
        return {"N": n, "R": float("nan"), "RMSE": float("nan"), "PE": float("nan")}
    residual_mse = float(np.mean((pred - obs) ** 2))
    baseline_mse = float(np.mean((obs - obs.mean()) ** 2))
    r = float(np.corrcoef(obs, pred)[0, 1]) if np.std(obs) > 0 and np.std(pred) > 0 else float("nan")
    rmse = float(np.sqrt(residual_mse))
    pe = float(1.0 - residual_mse / baseline_mse) if baseline_mse > 0 else float("nan")
    return {"N": int(n), "R": r, "RMSE": rmse, "PE": pe}


def _observed_total_flux(df: pd.DataFrame) -> np.ndarray:
    if "total_energy_flux" in df.columns:
        return df["total_energy_flux"].to_numpy(dtype=float)
    required = {"ele_energy_flux", "ion_energy_flux"}
    if required.issubset(df.columns):
        return (
            df["ele_energy_flux"].to_numpy(dtype=float)
            + df["ion_energy_flux"].to_numpy(dtype=float)
        )
    raise ValueError("test data must contain total_energy_flux or electron/ion total flux columns")


def evaluate_group(
    sampled: pd.DataFrame,
    *,
    model,
    scaler_path,
    ovation_sw: pd.DataFrame,
    estimators,
    device: str,
    batch_size: int,
) -> tuple[pd.DataFrame, dict]:
    paired = run_amt_4heads(
        sampled,
        model,
        scaler_path,
        device=device,
        batch_size=batch_size,
        pred_prefix="pred_amt",
    )
    paired["pred_amt_total"] = paired[
        ["pred_amt_d", "pred_amt_m", "pred_amt_b", "pred_amt_i"]
    ].sum(axis=1)
    paired = run_ovation_on_points(paired, ovation_sw, estimators=estimators)
    paired["obs_total"] = _observed_total_flux(paired)

    finite = np.isfinite(paired["pred_amt_total"]) & np.isfinite(paired["pred_ovation"]) & np.isfinite(paired["obs_total"])
    paired = paired.loc[finite].reset_index(drop=True)
    return paired, {
        "AMT": regression_metrics_log10(paired["obs_total"], paired["pred_amt_total"]),
        "OVATION-Prime": regression_metrics_log10(paired["obs_total"], paired["pred_ovation"]),
    }


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-data", required=True, help="Full held-out 2014 folded DMSP/SSJ test parquet")
    p.add_argument("--ovation-omni", required=True, help="OMNI table covering the four-hour OVATION history")
    p.add_argument("--model-path", required=True)
    p.add_argument("--scaler-path", required=True)
    p.add_argument("--snapshot-root", default=None, help="Directory containing archived auroramaps/ source and premodel bundle")
    p.add_argument("--output-dir", default="outputs/pixelwise_ovation")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--storm-utc-limit", type=int, default=30000)
    p.add_argument("--quiet-utc-limit", type=int, default=30000)
    p.add_argument("--all-utc-limit", type=int, default=60000)
    p.add_argument("--device", default="cpu")
    p.add_argument("--batch-size", type=int, default=32768)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    test = pd.read_parquet(args.test_data)
    test["utc"] = pd.to_datetime(test["utc"])
    ovation_sw = pd.read_parquet(args.ovation_omni)
    ovation_sw["utc"] = pd.to_datetime(ovation_sw["utc"])

    model = load_amt_model(args.model_path, device=args.device)
    ao = load_ovation_module(args.snapshot_root)
    estimators = make_flux_estimators(ao)

    limits = {
        "Storm": args.storm_utc_limit,
        "Quiet": args.quiet_utc_limit,
        "All": args.all_utc_limit,
    }
    summary = {}
    for group, limit in limits.items():
        sampled = sample_group_by_unique_utc(
            test, group, max_unique_times=limit, seed=args.seed
        )
        paired, metrics = evaluate_group(
            sampled,
            model=model,
            scaler_path=args.scaler_path,
            ovation_sw=ovation_sw,
            estimators=estimators,
            device=args.device,
            batch_size=args.batch_size,
        )
        paired.to_parquet(outdir / f"{group.lower()}_paired.parquet", index=False)
        summary[group] = {
            "sampled_unique_utc": int(sampled["utc"].nunique()),
            "paired_samples": int(len(paired)),
            **metrics,
        }

    (outdir / "metrics.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps(summary, indent=2))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
