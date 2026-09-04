#!/usr/bin/env python3
"""Reproduce the manuscript Table 4 global AMT--OVATION skill summary.

Input is the ``all_paired.parquet`` file written by
``evaluation/evaluate_pixelwise_ovation.py``. This guarantees that the global
metrics use the same sampled 2014 All evaluation set as Section 4.2.1.

Following the manuscript, rows are retained only when the observation is
finite and both model total-flux predictions are finite and greater than
1e-4 erg cm^-2 s^-1. Continuous metrics are evaluated in log10 total-flux
space, using an epsilon of 1e-6. Binary metrics use the physical-flux activity
threshold E_tot >= 0.5 erg cm^-2 s^-1.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score


REQUIRED_COLUMNS = ("obs_total", "pred_amt_total", "pred_ovation")


def filter_global_skill_sample(
    paired: pd.DataFrame,
    *,
    prediction_min: float = 1e-4,
) -> pd.DataFrame:
    """Apply the paired prediction filter stated in manuscript Section 4.2.4."""
    missing = sorted(set(REQUIRED_COLUMNS).difference(paired.columns))
    if missing:
        raise ValueError(f"Paired evaluation file is missing columns: {missing}")

    work = paired.copy()
    for col in REQUIRED_COLUMNS:
        work[col] = pd.to_numeric(work[col], errors="coerce")
    mask = np.ones(len(work), dtype=bool)
    for col in REQUIRED_COLUMNS:
        mask &= np.isfinite(work[col].to_numpy(dtype=float))
    mask &= work["pred_amt_total"].to_numpy(dtype=float) > float(prediction_min)
    mask &= work["pred_ovation"].to_numpy(dtype=float) > float(prediction_min)
    return work.loc[mask].reset_index(drop=True)


def _log_flux(values, epsilon: float) -> np.ndarray:
    flux = np.asarray(values, dtype=float)
    return np.log10(np.maximum(flux, 0.0) + float(epsilon))


def continuous_skill_metrics(
    observed_flux,
    predicted_flux,
    *,
    epsilon: float = 1e-6,
) -> dict[str, float]:
    """Return Pearson R, KGE, and NMedAE in log10 total-flux space."""
    obs = np.asarray(observed_flux, dtype=float)
    pred = np.asarray(predicted_flux, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) < 2:
        return {"R": float("nan"), "KGE": float("nan"), "NMedAE": float("nan")}

    y_true = _log_flux(obs, epsilon)
    y_pred = _log_flux(pred, epsilon)

    std_true = float(np.std(y_true))
    std_pred = float(np.std(y_pred))
    mean_true = float(np.mean(y_true))
    mean_pred = float(np.mean(y_pred))

    if std_true == 0.0 or std_pred == 0.0:
        r = float("nan")
    else:
        r = float(np.corrcoef(y_true, y_pred)[0, 1])

    if std_true == 0.0 or mean_true == 0.0 or not np.isfinite(r):
        kge = float("nan")
    else:
        alpha = std_pred / std_true
        beta = mean_pred / mean_true
        kge = float(1.0 - np.sqrt((r - 1.0) ** 2 + (alpha - 1.0) ** 2 + (beta - 1.0) ** 2))

    true_range = float(np.max(y_true) - np.min(y_true))
    medae = float(np.median(np.abs(y_pred - y_true)))
    nmedae = medae / true_range if true_range > 0.0 else float("nan")
    return {"R": r, "KGE": kge, "NMedAE": float(nmedae)}


def binary_skill_metrics(
    observed_flux,
    predicted_flux,
    *,
    activity_threshold: float = 0.5,
) -> dict[str, float]:
    """Return ROC AUC, CSI and accuracy at the manuscript activity threshold."""
    obs = np.asarray(observed_flux, dtype=float)
    pred = np.asarray(predicted_flux, dtype=float)
    valid = np.isfinite(obs) & np.isfinite(pred)
    obs = obs[valid]
    pred = pred[valid]
    if len(obs) == 0:
        return {"ROC_AUC": float("nan"), "CSI": float("nan"), "Accuracy": float("nan")}

    truth = obs >= float(activity_threshold)
    forecast = pred >= float(activity_threshold)
    tp = int(np.sum(truth & forecast))
    fn = int(np.sum(truth & ~forecast))
    fp = int(np.sum(~truth & forecast))
    denom = tp + fn + fp
    csi = float(tp / denom) if denom else float("nan")
    accuracy = float(np.mean(truth == forecast))

    if np.unique(truth).size < 2:
        auc = float("nan")
    else:
        auc = float(roc_auc_score(truth.astype(int), pred))
    return {"ROC_AUC": auc, "CSI": csi, "Accuracy": accuracy}


def evaluate_global_skill_table(
    paired: pd.DataFrame,
    *,
    prediction_min: float = 1e-4,
    activity_threshold: float = 0.5,
    epsilon: float = 1e-6,
) -> pd.DataFrame:
    """Compute the six Table 4 metrics for AMT and OVATION-Prime."""
    sample = filter_global_skill_sample(paired, prediction_min=prediction_min)
    if sample.empty:
        raise ValueError("No paired samples remain after the manuscript prediction filter")

    rows = []
    for model, column in (("AMT", "pred_amt_total"), ("OVATION-Prime", "pred_ovation")):
        continuous = continuous_skill_metrics(
            sample["obs_total"], sample[column], epsilon=epsilon
        )
        binary = binary_skill_metrics(
            sample["obs_total"], sample[column], activity_threshold=activity_threshold
        )
        rows.append(
            {
                "Model": model,
                "N": int(len(sample)),
                **binary,
                **continuous,
            }
        )
    return pd.DataFrame(rows)


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument(
        "--paired-data",
        default="outputs/pixelwise_ovation/all_paired.parquet",
        help="All paired evaluation parquet from evaluate_pixelwise_ovation.py",
    )
    p.add_argument("--output-dir", default="outputs/global_skill")
    p.add_argument("--prediction-min", type=float, default=1e-4)
    p.add_argument("--activity-threshold", type=float, default=0.5)
    p.add_argument("--epsilon", type=float, default=1e-6)
    return p


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    paired = pd.read_parquet(args.paired_data)
    sample = filter_global_skill_sample(paired, prediction_min=args.prediction_min)
    table = evaluate_global_skill_table(
        paired,
        prediction_min=args.prediction_min,
        activity_threshold=args.activity_threshold,
        epsilon=args.epsilon,
    )

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    sample.to_parquet(outdir / "global_skill_filtered_sample.parquet", index=False)
    table.to_csv(outdir / "global_skill_table.csv", index=False)

    print(f"Input paired rows       : {len(paired):,}")
    print(f"Rows after > {args.prediction_min:g} paired-model filter: {len(sample):,}")
    print(table.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
