#!/usr/bin/env python3
"""Evaluate AMT and four-hour OVATION-Prime boundaries against IMAGE.

The manuscript run used paired EALB/PALB coverage >= 18 of 24 MLT sectors,
backward-only OMNI matching within 10 min, a complete 120-min AMT history,
one-hour chronological thinning, and boundary thresholds 0.25/0.50/1.00
erg cm^-2 s^-1.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd

from evaluation.boundary_statistics_utils import (
    boundary_at_mlt,
    build_candidate_inventory,
    build_image_boundary_table,
    extract_boundaries,
    paired_boundary_metrics,
    required_amt_omni_columns,
    summarize_boundary_results,
    thin_by_time,
)
from evaluation.infer_v4_utils import load_amt_model, predict_grid_multi
from evaluation.ovation_model import make_flux_estimators, predict_total_flux


MLAT = np.linspace(50.0, 90.0, 80)
MLT = np.linspace(0.0, 24.0, 144)
IMAGE_MLT = np.arange(0.5, 24.5, 1.0)
TIME_TOLERANCE = pd.Timedelta(minutes=10)


def _thresholds(text):
    values = tuple(float(v.strip()) for v in text.split(",") if v.strip())
    if not values or any(v <= 0 for v in values):
        raise argparse.ArgumentTypeError("thresholds must be positive comma-separated values")
    return values


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--ealb-txt", required=True)
    p.add_argument("--palb-txt", required=True)
    p.add_argument("--omni-parquet", required=True)
    p.add_argument("--model-path")
    p.add_argument("--scaler-path")
    p.add_argument("--output-dir", default="outputs/boundary_statistics")
    p.add_argument("--device", default="cpu")
    p.add_argument("--min-image-pairs", type=int, default=18)
    p.add_argument("--thin-minutes", type=float, default=60.0)
    p.add_argument("--thresholds", type=_thresholds, default=(0.25, 0.50, 1.00))
    p.add_argument("--amt-time-batch", type=int, default=8)
    p.add_argument("--max-events", type=int)
    p.add_argument("--inventory-only", action="store_true")
    return p


def load_image_tables(ealb_path, palb_path):
    cols = ["Year", "SOY"] + [f"MLT_{i}" for i in range(24)]
    ealb = pd.read_csv(ealb_path, sep=r"\s+", comment="#", names=cols)
    palb = pd.read_csv(palb_path, sep=r"\s+", comment="#", names=cols)
    return build_image_boundary_table(ealb, palb)


def load_omni(path):
    df = pd.read_parquet(path)
    df["utc"] = pd.to_datetime(df["utc"])
    df = df.sort_values("utc").drop_duplicates("utc", keep="first").reset_index(drop=True)
    missing = [c for c in required_amt_omni_columns() if c not in df.columns]
    if missing:
        raise ValueError(f"OMNI data lack complete 120-min AMT history; examples: {missing[:8]}")
    return df


def prepare_amt_sw_row(omni, candidate):
    row = omni.iloc[int(candidate["omni_index"])].copy()
    # Geometry/time encodings are evaluated at the exact IMAGE timestamp, while
    # solar-wind values come only from the backward-matched OMNI row.
    row["utc"] = pd.Timestamp(candidate["eval_utc"])
    return row


def evaluate_threshold(total_amt, total_ov, threshold, gt_ealb, gt_palb):
    eq_amt, pol_amt = extract_boundaries(total_amt, MLAT, threshold=threshold, smooth_sigma=2.0)
    eq_ov, pol_ov = extract_boundaries(total_ov, MLAT, threshold=threshold, smooth_sigma=2.0)
    return {
        "EALB": paired_boundary_metrics(
            gt_ealb,
            boundary_at_mlt(IMAGE_MLT, eq_amt, MLT),
            boundary_at_mlt(IMAGE_MLT, eq_ov, MLT),
        ),
        "PALB": paired_boundary_metrics(
            gt_palb,
            boundary_at_mlt(IMAGE_MLT, pol_amt, MLT),
            boundary_at_mlt(IMAGE_MLT, pol_ov, MLT),
        ),
    }


def print_inventory(image_table, inventory, min_pairs):
    eligible = int((image_table["paired_valid_count"] >= min_pairs).sum())
    print("=" * 76)
    print("IMAGE boundary candidate inventory")
    print("=" * 76)
    print(f"IMAGE timestamps shared by EALB/PALB files : {len(image_table):,}")
    print(f"Paired-boundary coverage >= {min_pairs}/24       : {eligible:,}")
    print(f"OMNI/history/4-h-OVATION eligible        : {len(inventory):,}")
    if not inventory.empty:
        counts = inventory["activity"].value_counts()
        for name in ("Quiet", "Moderate", "Strong"):
            print(f"{name:>10}: {int(counts.get(name, 0)):,}")
        print(f"Time span : {inventory.utc.min()} -> {inventory.utc.max()}")
        print(f"Max backward OMNI offset: {inventory.omni_dt_min.max():.1f} min")
    print("=" * 76)


def main(argv=None):
    args = build_parser().parse_args(argv)
    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)

    image_table = load_image_tables(args.ealb_txt, args.palb_txt)
    omni = load_omni(args.omni_parquet)
    inventory = build_candidate_inventory(
        image_table,
        omni,
        min_pairs=args.min_image_pairs,
        tolerance=TIME_TOLERANCE,
    )
    inventory.to_csv(outdir / "boundary_candidate_inventory.csv", index=False)
    print_inventory(image_table, inventory, args.min_image_pairs)

    if args.inventory_only:
        return 0
    if not args.model_path or not args.scaler_path:
        raise ValueError("--model-path and --scaler-path are required unless --inventory-only is used")

    selected = inventory.copy()
    if args.thin_minutes > 0:
        selected = thin_by_time(selected, pd.Timedelta(minutes=args.thin_minutes))
    if args.max_events is not None:
        selected = selected.head(args.max_events)
    selected = selected.reset_index(drop=True)
    if selected.empty:
        raise RuntimeError("No IMAGE timestamps remain after filtering")
    selected.to_csv(outdir / "boundary_selected_events.csv", index=False)

    model = load_amt_model(args.model_path, device=args.device)
    estimators = make_flux_estimators()
    e_cols = [f"ealb_{i}" for i in range(24)]
    p_cols = [f"palb_{i}" for i in range(24)]
    rows = []

    for start in range(0, len(selected), args.amt_time_batch):
        batch = selected.iloc[start : start + args.amt_time_batch]
        sw_rows = [prepare_amt_sw_row(omni, rec) for _, rec in batch.iterrows()]
        amt_flux = predict_grid_multi(
            model,
            args.scaler_path,
            sw_rows,
            MLAT,
            MLT,
            device=args.device,
        )
        amt_total = amt_flux.sum(axis=1)

        for local_i, (_, rec) in enumerate(batch.iterrows()):
            image_rec = image_table.loc[int(rec["image_index"])]
            gt_ealb = image_rec[e_cols].to_numpy(dtype=float)
            gt_palb = image_rec[p_cols].to_numpy(dtype=float)
            ov_total = predict_total_flux(
                estimators,
                pd.Timestamp(rec["eval_utc"]).to_pydatetime(),
                float(rec["ovation_ec"]),
                MLAT,
                MLT,
            )
            for threshold in args.thresholds:
                metrics = evaluate_threshold(
                    amt_total[local_i], ov_total, float(threshold), gt_ealb, gt_palb
                )
                for boundary, values in metrics.items():
                    rows.append(
                        {
                            "utc": rec["utc"],
                            "eval_utc": rec["eval_utc"],
                            "omni_utc": rec["omni_utc"],
                            "omni_dt_min": rec["omni_dt_min"],
                            "activity": rec["activity"],
                            "Bz": rec["Bz"],
                            "P_dyn": rec["P_dyn"],
                            "ovation_ec": rec["ovation_ec"],
                            "threshold": float(threshold),
                            "boundary": boundary,
                            **values,
                        }
                    )
        pd.DataFrame(rows).to_csv(outdir / "boundary_statistics_all.partial.csv", index=False)
        print(f"Completed {min(start + len(batch), len(selected)):,}/{len(selected):,} timestamps")

    results = pd.DataFrame(rows)
    results.to_csv(outdir / "boundary_statistics_all.csv", index=False)
    summary = summarize_boundary_results(results)
    summary.to_csv(outdir / "boundary_statistics_summary.csv", index=False)
    summary.to_csv(outdir / "boundary_threshold_sensitivity.csv", index=False)
    partial = outdir / "boundary_statistics_all.partial.csv"
    if partial.exists():
        partial.unlink()

    metadata = {
        "min_image_pairs": args.min_image_pairs,
        "thin_minutes": args.thin_minutes,
        "thresholds": list(args.thresholds),
        "inventory_count": len(inventory),
        "selected_count": len(selected),
        "omni_matching": "backward-only, maximum 10 min, exact IMAGE timestamp",
        "activity": {
            "Quiet": "Bz >= -2 nT and P_dyn <= 3 nPa",
            "Strong": "Bz < -10 nT or P_dyn > 5 nPa",
            "Moderate": "all remaining eligible timestamps",
        },
        "metric_pairing": "AMT and OVATION scored on identical common valid MLT bins",
    }
    (outdir / "boundary_statistics_metadata.json").write_text(
        json.dumps(metadata, indent=2), encoding="utf-8"
    )

    primary = summary[np.isclose(summary["threshold"], 0.5)]
    print("\nPrimary threshold summary (0.5 erg cm^-2 s^-1)")
    print(primary.to_string(index=False, float_format=lambda x: f"{x:.3f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
