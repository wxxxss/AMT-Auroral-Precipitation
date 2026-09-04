#!/usr/bin/env python3
"""Train AMT with the configuration reported in the revised manuscript.

The defaults mirror Table 6 of the manuscript: AdamW, learning rate 3e-4,
weight decay 1e-2, batch size 8192, ReduceLROnPlateau with factor 0.5 and
patience 8, L2 gradient clipping at 5.0, at most 100 epochs, early-stopping
patience 50 on aggregate validation loss, and a full training snapshot every
5 epochs. The checkpoint with the minimum 2013 validation loss is retained as
the selected model.

Data paths are command-line arguments; no private filesystem locations are
embedded in this public release.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader

from data.dataset_v4 import AuroraMultiTaskDataset_V4
from method.loss import MultiTaskAsymmetricLoss
from method.model import AMT


def build_parser() -> argparse.ArgumentParser:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--train-parquet", required=True)
    p.add_argument("--val-parquet", required=True)
    p.add_argument("--output-dir", default="outputs/amt_training")
    p.add_argument("--device", default="cuda" if torch.cuda.is_available() else "cpu")
    p.add_argument("--batch-size", type=int, default=8192)
    p.add_argument("--epochs", type=int, default=100)
    p.add_argument("--learning-rate", type=float, default=3e-4)
    p.add_argument("--weight-decay", type=float, default=1e-2)
    p.add_argument("--num-workers", type=int, default=0)
    p.add_argument("--early-stop-patience", type=int, default=50)
    p.add_argument("--scheduler-patience", type=int, default=8)
    p.add_argument("--scheduler-factor", type=float, default=0.5)
    p.add_argument("--min-learning-rate", type=float, default=1e-6)
    p.add_argument("--max-grad-norm", type=float, default=5.0)
    p.add_argument("--checkpoint-interval", type=int, default=5)
    p.add_argument("--seed", type=int, default=42)
    return p


def _set_seed(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def _epoch(model, loader, criterion, device, optimizer=None, max_grad_norm=5.0):
    training = optimizer is not None
    model.train(training)
    total_loss = 0.0
    total_samples = 0
    per_head_abs = torch.zeros(4, device=device)

    context = torch.enable_grad() if training else torch.no_grad()
    with context:
        for batch in loader:
            x_sw, x_skip, target, _, _ = [item.to(device) for item in batch]
            if training:
                optimizer.zero_grad(set_to_none=True)
            pred = model(x_sw, x_skip)
            loss = criterion(pred, target)
            if training:
                loss.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=max_grad_norm)
                optimizer.step()

            batch_size = x_sw.shape[0]
            total_loss += float(loss.detach()) * batch_size
            per_head_abs += (pred.detach() - target).abs().mean(dim=0) * batch_size
            total_samples += batch_size

    if total_samples == 0:
        raise RuntimeError("empty DataLoader: no samples were evaluated")
    return total_loss / total_samples, (per_head_abs / total_samples).detach().cpu().tolist()


def _snapshot_payload(
    *,
    epoch: int,
    model,
    optimizer,
    scheduler,
    train_loss: float,
    val_loss: float,
    train_mae,
    val_mae,
    best_val_loss: float,
    config: dict,
) -> dict:
    """Return the full state required to resume a manuscript training run."""
    return {
        "epoch": int(epoch),
        "model_state_dict": model.state_dict(),
        "optimizer_state_dict": optimizer.state_dict(),
        "scheduler_state_dict": scheduler.state_dict(),
        "train_loss": float(train_loss),
        "val_loss": float(val_loss),
        "train_per_head_mae": list(train_mae),
        "val_per_head_mae": list(val_mae),
        "best_val_loss": float(best_val_loss),
        "config": dict(config),
    }


def main(argv=None) -> int:
    args = build_parser().parse_args(argv)
    _set_seed(args.seed)
    torch.set_default_dtype(torch.float32)

    if args.checkpoint_interval <= 0:
        raise ValueError("--checkpoint-interval must be positive")

    outdir = Path(args.output_dir)
    outdir.mkdir(parents=True, exist_ok=True)
    snapshots_dir = outdir / "snapshots"
    snapshots_dir.mkdir(parents=True, exist_ok=True)
    scaler_path = outdir / "sw_scaler_v4.pkl"
    best_path = outdir / "aurora_v4_best.pth"

    train_df = pd.read_parquet(args.train_parquet)
    val_df = pd.read_parquet(args.val_parquet)
    train_ds = AuroraMultiTaskDataset_V4(train_df, is_train=True, scaler_path=scaler_path)
    val_ds = AuroraMultiTaskDataset_V4(val_df, is_train=False, scaler_path=scaler_path)

    train_loader = DataLoader(
        train_ds,
        batch_size=args.batch_size,
        shuffle=True,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
        drop_last=True,
    )
    val_loader = DataLoader(
        val_ds,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=args.num_workers,
        pin_memory=args.device != "cpu",
    )

    device = torch.device(args.device)
    model = AMT(
        sw_dim=train_ds.X_sw_tensor.shape[1],
        skip_dim=train_ds.X_skip_tensor.shape[1],
        dropout=0.2,
    ).to(device)
    criterion = MultiTaskAsymmetricLoss(
        penalties=(5.0, 50.0, 50.0, 10.0), active_threshold=-5.0
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

    config = vars(args).copy()
    config.update(
        {
            "sw_dim": model.sw_dim,
            "skip_dim": model.skip_dim,
            "dropout": 0.2,
            "loss_active_threshold": -5.0,
            "penalties": [5.0, 50.0, 50.0, 10.0],
            "precision": "float32",
            "model_parameters": sum(p.numel() for p in model.parameters()),
            "selection_metric": "aggregate_2013_validation_loss",
        }
    )
    (outdir / "training_config.json").write_text(
        json.dumps(config, indent=2), encoding="utf-8"
    )

    best_val = float("inf")
    epochs_without_improvement = 0
    history = []

    for epoch in range(1, args.epochs + 1):
        train_loss, train_mae = _epoch(
            model, train_loader, criterion, device, optimizer, args.max_grad_norm
        )
        val_loss, val_mae = _epoch(model, val_loader, criterion, device)
        scheduler.step(val_loss)
        lr = optimizer.param_groups[0]["lr"]

        improved = val_loss < best_val
        if improved:
            best_val = val_loss
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        rec = {
            "epoch": epoch,
            "train_loss": train_loss,
            "val_loss": val_loss,
            "learning_rate": lr,
            "train_per_head_mae": train_mae,
            "val_per_head_mae": val_mae,
            "epochs_without_improvement": epochs_without_improvement,
        }
        history.append(rec)
        (outdir / "training_history.json").write_text(
            json.dumps(history, indent=2), encoding="utf-8"
        )

        payload = _snapshot_payload(
            epoch=epoch,
            model=model,
            optimizer=optimizer,
            scheduler=scheduler,
            train_loss=train_loss,
            val_loss=val_loss,
            train_mae=train_mae,
            val_mae=val_mae,
            best_val_loss=best_val,
            config=config,
        )

        if improved:
            torch.save(payload, best_path)

        if epoch % args.checkpoint_interval == 0:
            torch.save(payload, snapshots_dir / f"epoch_{epoch:03d}.pth")

        print(
            f"epoch={epoch:03d} train={train_loss:.6f} val={val_loss:.6f} "
            f"lr={lr:.3e} best_val={best_val:.6f}"
        )

        if (
            args.early_stop_patience > 0
            and epochs_without_improvement >= args.early_stop_patience
        ):
            print(
                f"early stopping after epoch {epoch}: "
                f"no validation improvement for {args.early_stop_patience} epochs"
            )
            break

    return 0


if __name__ == "__main__":
    raise SystemExit(main())
