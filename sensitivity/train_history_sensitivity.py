#!/usr/bin/env python3
"""Train one controlled solar-wind history-length sensitivity model.

All 60/90/120/180/240-min configurations use the same sample population that
is valid for the complete 240-min history. Apart from the number of retained
5-min lag steps, the public sensitivity runs use the same hidden-layer setup,
loss, optimization defaults, random seed, early-stopping rule, and
minimum-2013-validation-loss checkpoint criterion as the production AMT run.
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
    p.add_argument("--scheduler-patience", type=int, default=8)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--checkpoint-interval", type=int, default=5)
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
    if n_samples == 0:
        raise RuntimeError("empty validation DataLoader")
    return running_loss / n_samples, (running_mae / n_samples).cpu().tolist()


def _checkpoint_payload(
    *,
    epoch,
    model,
    optimizer,
    scheduler,
    val_loss,
    train_loss,
    val_mae,
    history_minutes,
    sw_dim,
    seed,
    best_val_loss,
):
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "val_loss": float(val_loss),
        "train_loss": float(train_loss),
        "val_per_head_mae": list(val_mae),
        "history_minutes": int(history_minutes),
        "sw_dim": int(sw_dim),
        "skip_dim": 9,
        "seed": int(seed),
        "best_val_loss": float(best_val_loss),
    }


def main(argv=None):
    args = build_parser().parse_args(argv)
    set_seed(args.seed)
    torch.set_default_dtype(torch.float32)
    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")

    history = int(args.history_minutes)
    expected_dim = sw_dim_for_history(history)

    run_dir = Path(args.output_root) / f"hist{history}m_seed{args.seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = run_dir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
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
    model = AMT(sw_dim=actual_dim, skip_dim=9, dropout=0.2).to(device)
    criterion = MultiTaskAsymmetricLoss(
        (5.0, 50.0, 50.0, 10.0), active_threshold=-5.0
    ).to(device)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer,
        mode="min",
        factor=args.scheduler_factor,
        patience=args.scheduler_patience,
        min_lr=args.min_learning_rate,
    )

    n_params = sum(p.numel() for p in model.parameters())
    best_val_loss = float("inf")
    epochs_without_improvement = 0
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
        if n_train == 0:
            raise RuntimeError("empty training DataLoader")

        train_loss = running_train / n_train
        val_loss, val_mae = evaluate_validation(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        row = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "lr": lr,
            "epochs_without_improvement": epochs_without_improvement,
            **{f"val_mae_{name}": value for name, value in zip(head_names, val_mae)},
        }
        history_rows.append(row)
        pd.DataFrame(history_rows).to_csv(run_dir / "training_history.csv", index=False)
        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={lr:.2e} best={best_val_loss:.6f}"
        )

        payload = _checkpoint_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            val_loss=val_loss,
            train_loss=train_loss,
            val_mae=val_mae,
            history_minutes=history,
            sw_dim=actual_dim,
            seed=args.seed,
            best_val_loss=best_val_loss,
        )
        if improved:
            torch.save(payload, best_path)
        if epoch % args.checkpoint_interval == 0:
            torch.save(payload, snapshots_dir / f"epoch_{epoch:03d}.pth")

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"early stopping after epoch {epoch}: no validation improvement "
                f"for {args.early_stop_patience} epochs"
            )
            break

    summary = {
        "history_minutes": history,
        "sw_dim": actual_dim,
        "skip_dim": 9,
        "seed": args.seed,
        "n_params": n_params,
        "best_val_loss": best_val_loss,
        "epochs_requested": args.epochs,
        "epochs_completed": len(history_rows),
        "batch_size": args.batch_size,
        "learning_rate": args.learning_rate,
        "weight_decay": args.weight_decay,
        "scheduler_patience": args.scheduler_patience,
        "scheduler_factor": args.scheduler_factor,
        "min_learning_rate": args.min_learning_rate,
        "early_stop_patience": args.early_stop_patience,
        "checkpoint_interval": args.checkpoint_interval,
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
