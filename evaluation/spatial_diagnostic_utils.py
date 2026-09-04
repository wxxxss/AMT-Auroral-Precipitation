"""MLT--MLAT spatial diagnostic utilities used for the revised Figure 9."""

from __future__ import annotations

import numpy as np
import pandas as pd


def _validate_grid(mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max):
    if mlt_bin_hours <= 0 or mlat_bin_deg <= 0:
        raise ValueError("Spatial bin sizes must be positive")
    if mlat_max <= mlat_min:
        raise ValueError("mlat_max must be greater than mlat_min")
    n_mlt = 24.0 / float(mlt_bin_hours)
    n_mlat = (float(mlat_max) - float(mlat_min)) / float(mlat_bin_deg)
    if not np.isclose(n_mlt, round(n_mlt)) or not np.isclose(n_mlat, round(n_mlat)):
        raise ValueError("Bin sizes must evenly divide the requested MLT/MLAT ranges")
    return int(round(n_mlt)), int(round(n_mlat))


def assign_spatial_bins(
    df,
    mlt_bin_hours=1.0,
    mlat_bin_deg=2.0,
    mlat_min=50.0,
    mlat_max=90.0,
):
    n_mlt, n_mlat = _validate_grid(mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max)
    out = df.copy()
    mlt = np.mod(pd.to_numeric(out["mlt"], errors="coerce").to_numpy(dtype=float), 24.0)
    mlat = np.abs(pd.to_numeric(out["mlat"], errors="coerce").to_numpy(dtype=float))
    mlt_idx = np.floor(mlt / float(mlt_bin_hours))
    mlat_idx = np.floor((mlat - float(mlat_min)) / float(mlat_bin_deg))
    finite = np.isfinite(mlt) & np.isfinite(mlat)
    inside = finite & (mlat >= float(mlat_min)) & (mlat <= float(mlat_max))
    mlt_idx = np.where(inside, np.clip(mlt_idx, 0, n_mlt - 1), np.nan)
    mlat_idx = np.where(inside, np.clip(mlat_idx, 0, n_mlat - 1), np.nan)
    out["mlt_folded"] = mlt
    out["mlat_abs"] = mlat
    out["mlt_bin"] = pd.array(mlt_idx, dtype="Int64")
    out["mlat_bin"] = pd.array(mlat_idx, dtype="Int64")
    return out


def compute_spatial_metrics(
    df,
    mlt_bin_hours=1.0,
    mlat_bin_deg=2.0,
    mlat_min=50.0,
    mlat_max=90.0,
    min_count=1,
    epsilon=1e-6,
):
    if min_count < 1:
        raise ValueError("min_count must be at least 1")
    if epsilon <= 0:
        raise ValueError("epsilon must be positive")

    n_mlt, n_mlat = _validate_grid(mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max)
    work = assign_spatial_bins(df, mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max)
    for col in ["true_flux", "pred_mlp_total", "pred_ovation"]:
        work[col] = pd.to_numeric(work[col], errors="coerce")

    paired = work[
        work["mlt_bin"].notna()
        & work["mlat_bin"].notna()
        & np.isfinite(work["true_flux"])
        & np.isfinite(work["pred_mlp_total"])
        & np.isfinite(work["pred_ovation"])
    ].copy()

    paired["obs_log"] = np.log10(np.clip(paired["true_flux"].to_numpy(dtype=float), epsilon, None))
    paired["amt_log"] = np.log10(np.clip(paired["pred_mlp_total"].to_numpy(dtype=float), epsilon, None))
    paired["ov_log"] = np.log10(np.clip(paired["pred_ovation"].to_numpy(dtype=float), epsilon, None))
    paired["amt_err"] = paired["amt_log"] - paired["obs_log"]
    paired["ov_err"] = paired["ov_log"] - paired["obs_log"]
    paired["amt_abs_err"] = np.abs(paired["amt_err"])
    paired["ov_abs_err"] = np.abs(paired["ov_err"])

    grouped = paired.groupby(["mlt_bin", "mlat_bin"], observed=True)
    agg = grouped.agg(
        n=("obs_log", "size"),
        obs_median_log=("obs_log", "median"),
        amt_bias_median=("amt_err", "median"),
        ov_bias_median=("ov_err", "median"),
        amt_medae=("amt_abs_err", "median"),
        ov_medae=("ov_abs_err", "median"),
    )
    agg["delta_medae"] = agg["ov_medae"] - agg["amt_medae"]

    full_index = pd.MultiIndex.from_product(
        [range(n_mlt), range(n_mlat)], names=["mlt_bin", "mlat_bin"]
    )
    agg = agg.reindex(full_index).reset_index()
    agg["n"] = agg["n"].fillna(0).astype(int)
    agg["valid"] = agg["n"] >= int(min_count)
    metrics = [
        "obs_median_log",
        "amt_bias_median",
        "ov_bias_median",
        "amt_medae",
        "ov_medae",
        "delta_medae",
    ]
    agg.loc[~agg["valid"], metrics] = np.nan
    agg["mlt_lo"] = agg["mlt_bin"] * float(mlt_bin_hours)
    agg["mlt_hi"] = agg["mlt_lo"] + float(mlt_bin_hours)
    agg["mlat_lo"] = float(mlat_min) + agg["mlat_bin"] * float(mlat_bin_deg)
    agg["mlat_hi"] = agg["mlat_lo"] + float(mlat_bin_deg)
    return agg


def summarize_spatial_metrics(metrics):
    valid = metrics[metrics["valid"].astype(bool)].copy()
    if valid.empty:
        return {
            "n_valid_bins": 0,
            "amt_better_bin_fraction": np.nan,
            "median_delta_medae": np.nan,
            "paired_points_in_valid_bins": 0,
        }
    delta = pd.to_numeric(valid["delta_medae"], errors="coerce")
    finite = np.isfinite(delta)
    values = delta[finite]
    return {
        "n_valid_bins": int(finite.sum()),
        "amt_better_bin_fraction": float((values > 0).mean()) if len(values) else np.nan,
        "median_delta_medae": float(values.median()) if len(values) else np.nan,
        "paired_points_in_valid_bins": int(valid.loc[finite, "n"].sum()),
    }


def count_spatial_bins(
    df,
    mlt_bin_hours=1.0,
    mlat_bin_deg=2.0,
    mlat_min=50.0,
    mlat_max=90.0,
):
    n_mlt, n_mlat = _validate_grid(mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max)
    work = assign_spatial_bins(df, mlt_bin_hours, mlat_bin_deg, mlat_min, mlat_max)
    valid = work[work["mlt_bin"].notna() & work["mlat_bin"].notna()].copy()
    counts = valid.groupby(["mlt_bin", "mlat_bin"], observed=True).size().rename("n")
    full_index = pd.MultiIndex.from_product(
        [range(n_mlt), range(n_mlat)], names=["mlt_bin", "mlat_bin"]
    )
    counts = counts.reindex(full_index, fill_value=0).reset_index()
    counts["n"] = counts["n"].astype(int)
    return counts


def summarize_bin_counts(counts, thresholds=(10, 20, 30, 50)):
    occupied = counts.loc[counts["n"] > 0, "n"].astype(float)
    summary = {
        "total_binned_points": int(counts["n"].sum()),
        "occupied_bins": int((counts["n"] > 0).sum()),
        "count_median": float(occupied.median()) if len(occupied) else np.nan,
        "count_q25": float(occupied.quantile(0.25)) if len(occupied) else np.nan,
        "count_q75": float(occupied.quantile(0.75)) if len(occupied) else np.nan,
        "count_min": int(occupied.min()) if len(occupied) else 0,
        "count_max": int(occupied.max()) if len(occupied) else 0,
    }
    for threshold in thresholds:
        summary[f"bins_ge_{int(threshold)}"] = int((counts["n"] >= int(threshold)).sum())
    return summary


def metrics_to_grid(metrics, column, n_mlt, n_mlat):
    grid = np.full((int(n_mlat), int(n_mlt)), np.nan, dtype=float)
    for row in metrics[["mlt_bin", "mlat_bin", column]].itertuples(index=False):
        grid[int(row.mlat_bin), int(row.mlt_bin)] = float(getattr(row, column))
    return grid
