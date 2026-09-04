#!/usr/bin/env python3
"""Evaluate all five AMT history-length models on one common 2014 subset.

The test parquet must contain rows with complete 240-min solar-wind history so
that the 60/90/120/180/240-min models are evaluated on identical samples.
Metrics are accumulated in streaming form to avoid loading the full common
subset into memory.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import pyarrow.parquet as pq
import torch

from data.dataset_v4 import AuroraMultiTaskDataset_V4
from method.model import AMT
from sensitivity.history_sensitivity_config import (
    SUPPORTED_HISTORY_MINUTES,
    lag_minutes_for_history,
    sw_dim_for_history,
)
from sensitivity.history_sensitivity_eval_utils import (
    StreamingRegressionMetrics,
    log_total_flux,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--test-parquet", required=True)
    p.add_argument("--run-root", default="outputs/history_sensitivity")
    p.add_argument("--output-dir", default="outputs/history_sensitivity_evaluation")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--chunk-rows", type=int, default=250000)
    p.add_argument("--inference-batch-size", type=int, default=32768)
    p.add_argument("--prediction-threshold", type=float, default=1e-4)
    return p


def run_paths(root, history_minutes, seed):
    run_dir = Path(root) / f"hist{history_minutes}m_seed{seed}"
    return {
        "run_dir": run_dir,
        "checkpoint": run_dir / "aurora_v4_best.pth",
        "scaler": run_dir / "sw_scaler_v4.pkl",
        "summary": run_dir / "training_summary.json",
        "history": run_dir / "training_history.csv",
    }


def load_training_metadata(history_minutes, paths):
    summary = json.loads(paths["summary"].read_text(encoding="utf-8"))
    history_df = pd.read_csv(paths["history"])
    best_row = history_df.loc[history_df["val_loss"].idxmin()]
    return {
        "history_minutes": history_minutes,
        "sw_dim": int(summary["sw_dim"]),
        "n_params": int(summary["n_params"]),
        "best_epoch": int(best_row["epoch"]),
        "best_val_loss": float(best_row["val_loss"]),
        "seed": int(summary["seed"]),
    }


def load_model(history_minutes, checkpoint_path, device):
    sw_dim = sw_dim_for_history(history_minutes)
    model = AMT(sw_dim=sw_dim, skip_dim=9).to(device)
    state = torch.load(checkpoint_path, map_location=device)
    if int(state.get("history_minutes", history_minutes)) != history_minutes:
        raise ValueError(f"Checkpoint history mismatch: {checkpoint_path}")
    if int(state.get("sw_dim", sw_dim)) != sw_dim:
        raise ValueError(f"Checkpoint sw_dim mismatch: {checkpoint_path}")
    model.load_state_dict(state["model_state_dict"])
    model.eval()
    return model


def required_columns(schema_names):
    columns = [
        "utc",
        "mlat",
        "mlt",
        "aurora_type",
        "ele_energy_flux",
        "ion_energy_flux",
        "Bx",
        "By",
        "Bz",
        "Vx",
        "Vy",
        "Vz",
        "P_dyn",
        "P_dyn_lag_5",
    ]
    if "src_hemi" in schema_names:
        columns.append("src_hemi")
    for var in ("Bx", "By", "Bz", "Vx", "Vy", "Vz"):
        for minute in lag_minutes_for_history(240):
            columns.append(f"{var}_lag_{minute}")
    missing = [c for c in columns if c not in schema_names]
    if missing:
        raise ValueError(f"Common test parquet missing columns: {missing[:20]}")
    return columns


def columns_for_history(history_minutes, has_src_hemi):
    columns = [
        "utc",
        "mlat",
        "mlt",
        "aurora_type",
        "ele_energy_flux",
        "ion_energy_flux",
        "Bx",
        "By",
        "Bz",
        "Vx",
        "Vy",
        "Vz",
        "P_dyn",
        "P_dyn_lag_5",
    ]
    if has_src_hemi:
        columns.append("src_hemi")
    for var in ("Bx", "By", "Bz", "Vx", "Vy", "Vz"):
        for minute in lag_minutes_for_history(history_minutes):
            columns.append(f"{var}_lag_{minute}")
    return columns


def predict_total(model, dataset, device, batch_size):
    output = np.empty(len(dataset), dtype=np.float32)
    with torch.no_grad():
        for start in range(0, len(dataset), batch_size):
            stop = min(start + batch_size, len(dataset))
            x_sw = dataset.X_sw_tensor[start:stop].to(device)
            x_skip = dataset.X_skip_tensor[start:stop].to(device)
            log_pred = model(x_sw, x_skip).cpu().numpy()
            linear = np.power(10.0, log_pred).astype(np.float32, copy=False) - np.float32(1e-6)
            np.maximum(linear, 0.0, out=linear)
            output[start:stop] = linear.sum(axis=1, dtype=np.float32)
    return output


def main(argv=None):
    args = build_parser().parse_args(argv)
    if args.chunk_rows <= 0 or args.inference_batch_size <= 0:
        raise ValueError("chunk sizes must be positive")
    device = torch.device(args.device)
    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    paths_by_history = {}
    metadata = {}
    models = {}
    for history in SUPPORTED_HISTORY_MINUTES:
        paths = run_paths(args.run_root, history, args.seed)
        for key in ("checkpoint", "scaler", "summary", "history"):
            if not paths[key].exists():
                raise FileNotFoundError(paths[key])
        meta = load_training_metadata(history, paths)
        if meta["sw_dim"] != sw_dim_for_history(history):
            raise ValueError(f"Training metadata sw_dim mismatch for {history} min")
        paths_by_history[history] = paths
        metadata[history] = meta
        models[history] = load_model(history, paths["checkpoint"], device)

    metrics = {
        h: {"all": StreamingRegressionMetrics(), "filtered": StreamingRegressionMetrics()}
        for h in SUPPORTED_HISTORY_MINUTES
    }

    parquet = pq.ParquetFile(args.test_parquet)
    schema_names = set(parquet.schema.names)
    read_columns = required_columns(schema_names)
    has_src_hemi = "src_hemi" in schema_names
    total_rows = parquet.metadata.num_rows
    processed = 0
    print(f"Common 2014 history-sensitivity rows: {total_rows:,}")

    for batch in parquet.iter_batches(batch_size=args.chunk_rows, columns=read_columns):
        chunk = batch.to_pandas()
        true_total = (
            chunk["ele_energy_flux"].fillna(0.0).to_numpy(dtype=np.float32)
            + chunk["ion_energy_flux"].fillna(0.0).to_numpy(dtype=np.float32)
        )
        true_log = log_total_flux(true_total)

        for history in SUPPORTED_HISTORY_MINUTES:
            cols = columns_for_history(history, has_src_hemi)
            dataset = AuroraMultiTaskDataset_V4(
                chunk[cols],
                is_train=False,
                scaler_path=paths_by_history[history]["scaler"],
                history_minutes=history,
            )
            pred_total = predict_total(
                models[history], dataset, device, args.inference_batch_size
            )
            pred_log = log_total_flux(pred_total)
            metrics[history]["all"].update(true_log, pred_log)
            mask = pred_total > args.prediction_threshold
            metrics[history]["filtered"].update(true_log[mask], pred_log[mask])

        processed += len(chunk)
        print(f"Processed {processed:,}/{total_rows:,} rows")

    rows = []
    for history in SUPPORTED_HISTORY_MINUTES:
        all_metrics = metrics[history]["all"].finalize()
        filtered = metrics[history]["filtered"].finalize()
        row = dict(metadata[history])
        row.update(
            {
                "test_rows_common": int(all_metrics["n"]),
                "all_r": float(all_metrics["r"]),
                "all_rmse": float(all_metrics["rmse"]),
                "all_pe": float(all_metrics["pe"]),
                "prediction_threshold": float(args.prediction_threshold),
                "filtered_n": int(filtered["n"]),
                "filtered_r": float(filtered["r"]),
                "filtered_rmse": float(filtered["rmse"]),
                "filtered_pe": float(filtered["pe"]),
            }
        )
        rows.append(row)

    results = pd.DataFrame(rows).sort_values("history_minutes").reset_index(drop=True)
    results.to_csv(output_dir / "history_sensitivity_results.csv", index=False)
    print(results.to_string(index=False, float_format=lambda value: f"{value:.6f}"))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
