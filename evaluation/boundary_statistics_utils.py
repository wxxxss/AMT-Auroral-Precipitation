"""Utilities for the multi-time IMAGE auroral-boundary evaluation."""

from __future__ import annotations

import numpy as np
import pandas as pd

from evaluation.ovation_driver import build_ovation_weighted_ec


def classify_activity(bz: float, p_dyn: float) -> str:
    if bz < -10.0 or p_dyn > 5.0:
        return "Strong"
    if bz >= -2.0 and p_dyn <= 3.0:
        return "Quiet"
    return "Moderate"


def find_backward_omni_row(
    df_omni: pd.DataFrame,
    target_utc,
    tolerance: pd.Timedelta = pd.Timedelta("10min"),
):
    """Return the latest OMNI row at or before ``target_utc`` within tolerance."""
    if df_omni.empty:
        return None
    target = pd.Timestamp(target_utc)
    utc = pd.DatetimeIndex(pd.to_datetime(df_omni["utc"]))
    pos = utc.searchsorted(target, side="right") - 1
    if pos < 0:
        return None
    dt = target - utc[pos]
    if dt < pd.Timedelta(0) or dt > tolerance:
        return None
    row = df_omni.iloc[pos].copy()
    row["_dt"] = dt
    return row


def build_image_boundary_table(ealb_df: pd.DataFrame, palb_df: pd.DataFrame) -> pd.DataFrame:
    """Join IMAGE EALB/PALB products on their exact Year/SOY timestamps."""
    mlt_cols = [f"MLT_{i}" for i in range(24)]

    def prep(df: pd.DataFrame, prefix: str) -> pd.DataFrame:
        out = df[["Year", "SOY"] + mlt_cols].copy()
        out = out.drop_duplicates().reset_index(drop=True)
        conflicting = out.duplicated(["Year", "SOY"], keep=False)
        if conflicting.any():
            keys = (
                out.loc[conflicting, ["Year", "SOY"]]
                .drop_duplicates()
                .head(5)
                .to_dict("records")
            )
            raise ValueError(
                "IMAGE boundary file contains conflicting duplicate Year/SOY rows; "
                f"example keys: {keys}"
            )
        out["utc"] = (
            pd.to_datetime(out["Year"].astype(int).astype(str), format="%Y")
            + pd.to_timedelta(out["SOY"], unit="s")
        )
        vals = out[mlt_cols].replace(0.0, np.nan)
        renamed = vals.rename(columns={c: f"{prefix}_{i}" for i, c in enumerate(mlt_cols)})
        return pd.concat([out[["utc"]], renamed], axis=1)

    e = prep(ealb_df, "ealb")
    p = prep(palb_df, "palb")
    out = e.merge(p, on="utc", how="inner", validate="one_to_one").sort_values("utc").reset_index(drop=True)
    e_cols = [f"ealb_{i}" for i in range(24)]
    p_cols = [f"palb_{i}" for i in range(24)]
    e_vals = out[e_cols].to_numpy(dtype=float)
    p_vals = out[p_cols].to_numpy(dtype=float)
    paired = np.isfinite(e_vals) & np.isfinite(p_vals)
    out["paired_valid_count"] = paired.sum(axis=1)
    out["mean_eq_lat"] = np.nanmean(e_vals, axis=1)
    return out


def paired_boundary_metrics(gt, amt, ov) -> dict:
    """Score AMT and OVATION on the same valid IMAGE MLT sectors."""
    gt = np.asarray(gt, dtype=float)
    amt = np.asarray(amt, dtype=float)
    ov = np.asarray(ov, dtype=float)
    gt_valid = np.isfinite(gt)
    amt_valid = gt_valid & np.isfinite(amt)
    ov_valid = gt_valid & np.isfinite(ov)
    common = amt_valid & ov_valid

    n_image = int(gt_valid.sum())
    n_common = int(common.sum())
    if n_common:
        amt_err = amt[common] - gt[common]
        ov_err = ov[common] - gt[common]
        amt_mae = float(np.mean(np.abs(amt_err)))
        amt_rmse = float(np.sqrt(np.mean(amt_err**2)))
        ov_mae = float(np.mean(np.abs(ov_err)))
        ov_rmse = float(np.sqrt(np.mean(ov_err**2)))
        amt_win = bool(amt_mae < ov_mae)
    else:
        amt_mae = amt_rmse = ov_mae = ov_rmse = float("nan")
        amt_win = False

    return {
        "n_image_valid": n_image,
        "n_amt_valid": int(amt_valid.sum()),
        "n_ov_valid": int(ov_valid.sum()),
        "n_common_valid": n_common,
        "amt_mae": amt_mae,
        "amt_rmse": amt_rmse,
        "ov_mae": ov_mae,
        "ov_rmse": ov_rmse,
        "amt_coverage": float(amt_valid.sum() / n_image) if n_image else float("nan"),
        "ov_coverage": float(ov_valid.sum() / n_image) if n_image else float("nan"),
        "amt_win_mae": amt_win,
    }


def thin_by_time(df: pd.DataFrame, min_separation: pd.Timedelta) -> pd.DataFrame:
    """Chronologically thin rows using a greedy minimum-separation rule."""
    if df.empty:
        return df.copy()
    ordered = df.sort_values("utc").reset_index(drop=True)
    keep = [0]
    last = pd.Timestamp(ordered.loc[0, "utc"])
    for i in range(1, len(ordered)):
        now = pd.Timestamp(ordered.loc[i, "utc"])
        if now - last >= min_separation:
            keep.append(i)
            last = now
    return ordered.iloc[keep].reset_index(drop=True)


def required_amt_omni_columns():
    base = ["Bx", "By", "Bz", "Vx", "Vy", "Vz", "P_dyn"]
    return base + [f"{v}_lag_{m}" for v in base for m in range(5, 121, 5)]


def has_complete_amt_history(row) -> bool:
    try:
        vals = np.asarray([row[c] for c in required_amt_omni_columns()], dtype=float)
    except KeyError:
        return False
    return bool(np.isfinite(vals).all())


def extract_boundaries(flux_grid, mlat_array, threshold=0.5, smooth_sigma=2.0):
    """Extract equatorward and poleward threshold boundaries for each MLT column."""
    flux_grid = np.asarray(flux_grid, dtype=float)
    mlat_array = np.asarray(mlat_array, dtype=float)
    eq_b = np.full(flux_grid.shape[1], np.nan, dtype=float)
    pol_b = np.full(flux_grid.shape[1], np.nan, dtype=float)
    for i in range(flux_grid.shape[1]):
        active = np.where(flux_grid[:, i] >= threshold)[0]
        if active.size:
            eq_b[i] = mlat_array[active[0]]
            pol_b[i] = mlat_array[active[-1]]
    if smooth_sigma is not None and np.isfinite(eq_b).all() and np.isfinite(pol_b).all():
        from scipy.ndimage import gaussian_filter1d

        eq_b = gaussian_filter1d(eq_b, sigma=float(smooth_sigma), mode="wrap")
        pol_b = gaussian_filter1d(pol_b, sigma=float(smooth_sigma), mode="wrap")
    return eq_b, pol_b


def boundary_at_mlt(mlt_target_arr, boundary_grid, mlt_grid):
    mlt_target_arr = np.asarray(mlt_target_arr, dtype=float)
    boundary_grid = np.asarray(boundary_grid, dtype=float)
    mlt_grid = np.asarray(mlt_grid, dtype=float)
    return np.asarray(
        [boundary_grid[np.argmin(np.abs(mlt_grid - (mt % 24.0)))] for mt in mlt_target_arr],
        dtype=float,
    )


def build_candidate_inventory(
    image_table: pd.DataFrame,
    df_omni: pd.DataFrame,
    min_pairs: int = 18,
    tolerance: pd.Timedelta = pd.Timedelta("10min"),
) -> pd.DataFrame:
    """Build the exact-timestamp IMAGE/OMNI inventory used in the paper."""
    omni = df_omni.copy()
    omni["utc"] = pd.to_datetime(omni["utc"])
    omni = omni.sort_values("utc").drop_duplicates("utc", keep="first").reset_index(drop=True)
    missing = [c for c in required_amt_omni_columns() if c not in omni.columns]
    if missing:
        raise ValueError(f"OMNI parquet missing required AMT columns: {missing[:8]}")

    rows = []
    eligible_image = image_table[image_table["paired_valid_count"] >= int(min_pairs)].copy()
    for image_idx, rec in eligible_image.iterrows():
        eval_utc = pd.Timestamp(rec["utc"])
        sw = find_backward_omni_row(omni, eval_utc, tolerance=tolerance)
        if sw is None or not has_complete_amt_history(sw):
            continue
        rows.append(
            {
                "image_index": int(image_idx),
                "utc": pd.Timestamp(rec["utc"]),
                "eval_utc": eval_utc,
                "paired_valid_count": int(rec["paired_valid_count"]),
                "mean_eq_lat": float(rec.get("mean_eq_lat", np.nan)),
                "omni_index": int(sw.name),
                "omni_utc": pd.Timestamp(sw["utc"]),
                "omni_dt_min": float(sw["_dt"] / pd.Timedelta(minutes=1)),
                "Bz": float(sw["Bz"]),
                "P_dyn": float(sw["P_dyn"]),
                "activity": classify_activity(float(sw["Bz"]), float(sw["P_dyn"])),
            }
        )

    columns = [
        "image_index",
        "utc",
        "eval_utc",
        "paired_valid_count",
        "mean_eq_lat",
        "omni_index",
        "omni_utc",
        "omni_dt_min",
        "Bz",
        "P_dyn",
        "activity",
        "ovation_ec",
    ]
    if not rows:
        return pd.DataFrame(columns=columns)

    out = pd.DataFrame(rows)
    out["ovation_ec"] = build_ovation_weighted_ec(omni, out["eval_utc"])
    return out[np.isfinite(out["ovation_ec"])].reset_index(drop=True)


def _summary_rows(df: pd.DataFrame, activity_label: str):
    rows = []
    for (threshold, boundary), group in df.groupby(["threshold", "boundary"], dropna=False):
        rows.append(
            {
                "threshold": float(threshold),
                "activity": activity_label,
                "boundary": boundary,
                "n_time": int(len(group)),
                "amt_mae_median": float(group["amt_mae"].median()),
                "amt_mae_q25": float(group["amt_mae"].quantile(0.25)),
                "amt_mae_q75": float(group["amt_mae"].quantile(0.75)),
                "ov_mae_median": float(group["ov_mae"].median()),
                "ov_mae_q25": float(group["ov_mae"].quantile(0.25)),
                "ov_mae_q75": float(group["ov_mae"].quantile(0.75)),
                "amt_rmse_median": float(group["amt_rmse"].median()),
                "ov_rmse_median": float(group["ov_rmse"].median()),
                "amt_coverage_median": float(group["amt_coverage"].median()),
                "ov_coverage_median": float(group["ov_coverage"].median()),
                "amt_win_fraction": float(group["amt_win_mae"].astype(float).mean()),
            }
        )
    return rows


def summarize_boundary_results(results: pd.DataFrame) -> pd.DataFrame:
    if results.empty:
        return pd.DataFrame()
    rows = _summary_rows(results, "All")
    for activity, group in results.groupby("activity", sort=False):
        rows.extend(_summary_rows(group, str(activity)))
    return pd.DataFrame(rows).sort_values(["threshold", "activity", "boundary"]).reset_index(drop=True)
