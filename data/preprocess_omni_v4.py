#!/usr/bin/env python3
"""Public preprocessing helpers for the manuscript V4 OMNI/SSJ pipeline.

This module implements the OMNI-side preprocessing described in the revised
manuscript:

- seven primitive OMNI variables on a regular 5-min grid;
- manuscript quality-control bounds;
- linear interpolation only for complete gaps of at most 30 min;
- 5-min history lags;
- backward-only OMNI-to-SSJ matching with a 10-min tolerance;
- chronological 2009--2012 / 2013 train-validation splitting with
  within-class downsampling while preserving the natural class proportions.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


OMNI_VARS = ["Bx", "By", "Bz", "Vx", "Vy", "Vz", "P_dyn"]
CADENCE_MINUTES = 5
DEFAULT_INTERPOLATION_LIMIT_STEPS = 6
DEFAULT_MATCH_TOLERANCE = "10min"
DEFAULT_TRAIN_FRACTION = 0.10
DEFAULT_VAL_FRACTION = 0.05
DEFAULT_VAL_YEAR = 2013
DEFAULT_RANDOM_STATE = 42


def _interpolate_short_internal_gaps(series: pd.Series, max_steps: int) -> pd.Series:
    """Linearly fill an entire internal NaN run only when its length <= max_steps."""
    max_steps = int(max_steps)
    if max_steps < 0:
        raise ValueError("max_steps must be nonnegative")
    if max_steps == 0 or not series.isna().any():
        return series.copy()

    missing = series.isna()
    candidate = series.interpolate(method="linear", limit_area="inside")
    result = series.copy()
    run_ids = missing.ne(missing.shift(fill_value=False)).cumsum()
    for _, run_mask in missing.groupby(run_ids):
        if not bool(run_mask.iloc[0]):
            continue
        idx = run_mask.index
        # groupby above preserves the original index subset through the grouped
        # Series; select only members of this run before applying its length rule.
        idx = run_mask[run_mask].index
        if len(idx) <= max_steps:
            result.loc[idx] = candidate.loc[idx]
    return result


def regularize_omni_5min(
    df: pd.DataFrame,
    *,
    interpolation_limit_steps: int = DEFAULT_INTERPOLATION_LIMIT_STEPS,
) -> pd.DataFrame:
    """Apply manuscript QC, align to 5 min, and interpolate only short gaps."""
    required = {"utc", *OMNI_VARS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"OMNI dataframe missing columns: {missing}")

    out = df[["utc", *OMNI_VARS]].copy()
    out["utc"] = pd.to_datetime(out["utc"])
    out = out.sort_values("utc").drop_duplicates("utc", keep="first")

    for col in ("Bx", "By", "Bz"):
        out.loc[(out[col] < -500.0) | (out[col] > 500.0), col] = np.nan
    for col in ("Vx", "Vy", "Vz"):
        out.loc[(out[col] < -3000.0) | (out[col] > 3000.0), col] = np.nan
    out.loc[(out["P_dyn"] < 0.0) | (out["P_dyn"] > 90.0), "P_dyn"] = np.nan

    out = out.set_index("utc").asfreq(f"{CADENCE_MINUTES}min")
    for col in OMNI_VARS:
        out[col] = _interpolate_short_internal_gaps(
            out[col], int(interpolation_limit_steps)
        )
    return out.reset_index()


def add_history_lags(
    df: pd.DataFrame,
    *,
    history_minutes: int = 120,
    cadence_minutes: int = CADENCE_MINUTES,
    drop_incomplete: bool = True,
) -> pd.DataFrame:
    """Add raw OMNI lag columns at 5-min increments through the horizon."""
    history_minutes = int(history_minutes)
    cadence_minutes = int(cadence_minutes)
    if history_minutes <= 0 or history_minutes % cadence_minutes != 0:
        raise ValueError("history_minutes must be a positive multiple of cadence_minutes")

    required = {"utc", *OMNI_VARS}
    missing = sorted(required.difference(df.columns))
    if missing:
        raise ValueError(f"OMNI dataframe missing columns: {missing}")

    out = df.copy()
    out["utc"] = pd.to_datetime(out["utc"])
    out = out.sort_values("utc").reset_index(drop=True)

    lag_cols: list[str] = []
    for var in OMNI_VARS:
        for minute in range(cadence_minutes, history_minutes + 1, cadence_minutes):
            col = f"{var}_lag_{minute}"
            out[col] = out[var].shift(minute // cadence_minutes)
            lag_cols.append(col)

    if drop_incomplete:
        out = out.dropna(subset=[*OMNI_VARS, *lag_cols]).reset_index(drop=True)
    return out


def backward_match_ssj(
    ssj_df: pd.DataFrame,
    omni_df: pd.DataFrame,
    *,
    tolerance: str = DEFAULT_MATCH_TOLERANCE,
    drop_unmatched: bool = True,
) -> pd.DataFrame:
    """Match each SSJ record to the most recent non-future OMNI row."""
    if "utc" not in ssj_df.columns or "utc" not in omni_df.columns:
        raise ValueError("Both dataframes must contain utc")

    left = ssj_df.copy()
    right = omni_df.copy()
    left["utc"] = pd.to_datetime(left["utc"])
    right["utc"] = pd.to_datetime(right["utc"])
    left = left.sort_values("utc")
    right = right.sort_values("utc")

    out = pd.merge_asof(
        left,
        right,
        on="utc",
        direction="backward",
        tolerance=pd.Timedelta(tolerance),
    )
    if drop_unmatched:
        present = [col for col in OMNI_VARS if col in out.columns]
        if present:
            out = out.dropna(subset=present)
        lag_cols = [col for col in out.columns if "_lag_" in col]
        if lag_cols:
            out = out.dropna(subset=lag_cols)
    return out.reset_index(drop=True)


def _sample_each_class(df: pd.DataFrame, fraction: float, random_state: int) -> pd.DataFrame:
    if not 0.0 < fraction <= 1.0:
        raise ValueError("sampling fraction must be in (0, 1]")
    if "aurora_type" not in df.columns:
        raise ValueError("SSJ dataframe must contain aurora_type")
    parts = [
        group.sample(frac=fraction, random_state=random_state)
        for _, group in df.groupby("aurora_type", sort=True)
    ]
    if not parts:
        return df.iloc[0:0].copy()
    return pd.concat(parts, ignore_index=True).sort_values("utc").reset_index(drop=True)


def stratified_chronological_split(
    ssj_df: pd.DataFrame,
    *,
    val_year: int = DEFAULT_VAL_YEAR,
    train_fraction: float = DEFAULT_TRAIN_FRACTION,
    val_fraction: float = DEFAULT_VAL_FRACTION,
    random_state: int = DEFAULT_RANDOM_STATE,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """Create chronologically separated, class-stratified train/validation rows."""
    if "utc" not in ssj_df.columns:
        raise ValueError("SSJ dataframe must contain utc")
    df = ssj_df.copy()
    df["utc"] = pd.to_datetime(df["utc"])
    years = df["utc"].dt.year
    train_raw = df.loc[years < int(val_year)].copy()
    val_raw = df.loc[years == int(val_year)].copy()
    if train_raw.empty or val_raw.empty:
        raise ValueError("Both pre-validation training years and validation year are required")

    train = _sample_each_class(train_raw, float(train_fraction), int(random_state))
    val = _sample_each_class(val_raw, float(val_fraction), int(random_state))
    return train, val


def read_omni_5min_cdf(path: str | Path) -> pd.DataFrame:
    """Read the seven manuscript primitive variables from one OMNI 5-min CDF."""
    try:
        import cdflib
    except ImportError as exc:  # pragma: no cover
        raise ImportError("cdflib is required to read raw OMNI CDF files") from exc

    cdf = cdflib.CDF(str(path))
    epoch = cdflib.cdfepoch.to_datetime(cdf.varget("Epoch"))
    return pd.DataFrame(
        {
            "utc": epoch,
            "Bx": cdf.varget("BX_GSE"),
            "By": cdf.varget("BY_GSM"),
            "Bz": cdf.varget("BZ_GSM"),
            "Vx": cdf.varget("Vx"),
            "Vy": cdf.varget("Vy"),
            "Vz": cdf.varget("Vz"),
            "P_dyn": cdf.varget("Pressure"),
        }
    )


def build_omni_history_from_cdf_tree(
    cdf_root: str | Path,
    *,
    years: range,
    history_minutes: int,
) -> pd.DataFrame:
    """Build one regularized/history-augmented OMNI table from yearly CDF folders."""
    root = Path(cdf_root)
    years = list(years)
    parts: list[pd.DataFrame] = []
    for year in years:
        year_dir = root / str(year)
        if not year_dir.is_dir():
            continue
        for path in sorted(year_dir.glob("*.cdf")):
            parts.append(read_omni_5min_cdf(path))
    if not parts:
        raise FileNotFoundError(f"No OMNI CDF files found under {root} for {years}")

    raw = pd.concat(parts, ignore_index=True)
    regular = regularize_omni_5min(raw)
    return add_history_lags(regular, history_minutes=history_minutes)


def _cmd_build_omni(args: argparse.Namespace) -> None:
    out = build_omni_history_from_cdf_tree(
        args.cdf_root,
        years=range(args.start_year, args.end_year + 1),
        history_minutes=args.history_minutes,
    )
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    out.to_parquet(args.output, index=False)
    print(f"wrote {len(out):,} rows to {args.output}")


def _cmd_train_val(args: argparse.Namespace) -> None:
    ssj = pd.read_parquet(args.ssj_parquet)
    omni = pd.read_parquet(args.omni_parquet)
    train_ssj, val_ssj = stratified_chronological_split(
        ssj,
        val_year=args.val_year,
        train_fraction=args.train_fraction,
        val_fraction=args.val_fraction,
        random_state=args.seed,
    )
    train = backward_match_ssj(train_ssj, omni, tolerance=args.tolerance)
    val = backward_match_ssj(val_ssj, omni, tolerance=args.tolerance)
    Path(args.train_output).parent.mkdir(parents=True, exist_ok=True)
    Path(args.val_output).parent.mkdir(parents=True, exist_ok=True)
    train.to_parquet(args.train_output, index=False)
    val.to_parquet(args.val_output, index=False)
    print(f"train={len(train):,} -> {args.train_output}")
    print(f"val={len(val):,} -> {args.val_output}")


def _cmd_test(args: argparse.Namespace) -> None:
    ssj = pd.read_parquet(args.ssj_parquet)
    omni = pd.read_parquet(args.omni_parquet)
    test = backward_match_ssj(ssj, omni, tolerance=args.tolerance)
    Path(args.output).parent.mkdir(parents=True, exist_ok=True)
    test.to_parquet(args.output, index=False)
    print(f"test={len(test):,} -> {args.output}")


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)

    p_omni = sub.add_parser("build-omni", help="Build regularized OMNI history parquet")
    p_omni.add_argument("--cdf-root", required=True)
    p_omni.add_argument("--start-year", type=int, required=True)
    p_omni.add_argument("--end-year", type=int, required=True)
    p_omni.add_argument("--history-minutes", type=int, default=120)
    p_omni.add_argument("--output", required=True)
    p_omni.set_defaults(func=_cmd_build_omni)

    p_tv = sub.add_parser("train-val", help="Split folded SSJ and backward-match OMNI")
    p_tv.add_argument("--ssj-parquet", required=True)
    p_tv.add_argument("--omni-parquet", required=True)
    p_tv.add_argument("--train-output", required=True)
    p_tv.add_argument("--val-output", required=True)
    p_tv.add_argument("--val-year", type=int, default=DEFAULT_VAL_YEAR)
    p_tv.add_argument("--train-fraction", type=float, default=DEFAULT_TRAIN_FRACTION)
    p_tv.add_argument("--val-fraction", type=float, default=DEFAULT_VAL_FRACTION)
    p_tv.add_argument("--seed", type=int, default=DEFAULT_RANDOM_STATE)
    p_tv.add_argument("--tolerance", default=DEFAULT_MATCH_TOLERANCE)
    p_tv.set_defaults(func=_cmd_train_val)

    p_test = sub.add_parser("test", help="Backward-match a held-out folded SSJ parquet")
    p_test.add_argument("--ssj-parquet", required=True)
    p_test.add_argument("--omni-parquet", required=True)
    p_test.add_argument("--output", required=True)
    p_test.add_argument("--tolerance", default=DEFAULT_MATCH_TOLERANCE)
    p_test.set_defaults(func=_cmd_test)
    return parser


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    args.func(args)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
