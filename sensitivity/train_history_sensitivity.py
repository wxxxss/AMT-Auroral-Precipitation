#!/usr/bin/env python3
"""Train one controlled solar-wind history-length sensitivity model.

All history lengths must use the same train/validation row population that is
valid for the full 240-min history. The only controlled change is the number of
5-min lagged driver features exposed to the AMT dataset. The manuscript
sensitivity runs used 100 epochs and retained the minimum-validation-loss
checkpoint for each history length.
"""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset_v4 import AuroraMultiTaskDataset_V4
from method.loss import MultiTaskAsymmetricLoss
from method.model import AMT
from sensitivity.history_sensitivity_config import (
    SUPPORTED_HISTORY_MINUTES,
    sw_dim_for_history,
)


def build_parser():
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--history-minutes", type=int, required=True, choices=SUPPORTED_HISTORY_MINUTES)
    p.add_argument("--train-parquet", required=True, help="Common 240-min-valid training rows")
    p.add_argument("--val-parquet", required=True, help="Common 240-min-valid validation rows")
    p.add_argument("--output-root", default="outputs/history_sensitivity")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    return p


def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def evaluate_validation(model, loader, criterion, device):
    model.eval()
    running_loss = 0.0
    running_mae = torch.zeros(4, device=device)
    n_samples = 0
    with torch.no_grad():
        for batch in loader:
            x_sw, x_skip, target, _, _ = [tensor.to(device) for tensor in batch]
            pred = model(x_sw, x_skip)
            loss = criterion(pred, target)
            bs = x_sw.size(0)
            running_loss += float(loss) * bs
            running_mae += (pred - target).abs().mean(dim=0) * bs
            n_samples += bs
    return running_loss / n_samples, (running_mae / n_samples).cpu().tolist()


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    history = int(args.history_minutes)
    expected_dim = sw_dim_for_history(history)

    run_dir = Path(args.output_root) / f"hist{history}m_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = run_dir / "sw_scaler_v4.pkl"
    best_path = run_dir / "aurora_v4_best.pth"

    train_df = pd.read_parquet(args.train_parquet)
    val_df = pd.read_parquet(args.val_parquet)
    train_dataset = AuroraMultiTaskDataset_V4(
        train_df,
        is_train=True,
        scaler_path=scaler_path,
        history_minutes=history,
    )
    val_dataset = AuroraMultiTaskDataset_V4(
        val_df,
        is_train=False,
        scaler_path=scaler_path,
        history_minutes=history,
    )
    actual_dim = int(train_dataset.X_sw_tensor.shape[1])
    if actual_dim != expected_dim:
        raise RuntimeError(f"Unexpected sw_dim: got {actual_dim}, expected {expected_dim}")

    generator = torch.Generator().manual_seed(args.seed)
    train_loader = DataLoader(
        train_dataset,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        drop_last=True,
        generator=generator,
    )
    val_loader = DataLoader(
        val_dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    device = torch.device(args.device)
    model = AMT(sw_dim=actual_dim, skip_dim=9).to(device)
    criterion = MultiTaskAsymmetricLoss((5.0, 50.0, 50.0, 10.0)).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode="min", factor=0.5, patience=8, min_lr=1e-6
    )

    n_params = sum(p.numel() for p in model.parameters())
    best_val_loss = float("inf")
    history_rows = []
    head_names = ["diff", "mono", "bb", "ion"]

    print(f"history={history} min | sw_dim={actual_dim} | params={n_params:,}")
    print(f"common train rows={len(train_df):,} | common val rows={len(val_df):,}")

    for epoch in range(1, args.epochs + 1):
        model.train()
        running_train = 0.0
        n_train = 0
        for batch in train_loader:
            x_sw, x_skip, target, _, _ = [tensor.to(device) for tensor in batch]
            optimizer.zero_grad(set_to_none=True)
            pred = model(x_sw, x_skip)
            loss = criterion(pred, target)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=args.max_grad_norm)
            optimizer.step()
            bs = x_sw.size(0)
            running_train += float(loss.detach()) * bs
            n_train += bs

        train_loss = running_train / n_train
        val_loss, val_mae = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]
        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            **{f"val_mae_{name}": value for name, value in zip(head_names, val_mae)},
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
        print(f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} lr={lr:.2e}")

        if val_loss < best_val_loss:
            best_val_loss = val_loss
            torch.save(
                {
                    "epoch": epoch,
                    "model_state_dict": model.state_dict(),
                    "val_loss": val_loss,
                    "train_loss": train_loss,
                    "val_per_head_mae": val_mae,
                    "history_minutes": history,
                    "sw_dim": actual_dim,
                    "skip_dim": 9,
                    "seed": args.seed,
                },
                best_path,
            )

    summary = {
        "history_minutes": history,
        "sw_dim": actual_dim,
        "skip_dim": 9,
        "seed": args.seed,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "epochs": args.epochs,
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "train_rows_common": len(train_df),
        "val_rows_common": len(val_df),
        "best_checkpoint": str(best_path),
    }
    (run_dir / "training_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
