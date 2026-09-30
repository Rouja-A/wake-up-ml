"""Training entry point for the extension models (Linear, D), reusing every shared building block from
`src.train` (optimizer, scheduler, EarlyStopper, one train/eval epoch, checkpoint I/O, seeding) so the
protocol is IDENTICAL to Phase 1's except for which `nn.Module` gets built -- `src.train.fit` hardcodes
`src.model.build_model(model_name)` for A/B/C, so this module supplies the equivalent for Linear/D rather
than modifying `src/train.py` (an existing file) at all.
"""
from __future__ import annotations

import time
from pathlib import Path

import torch
import torch.nn as nn

from src.train import (EarlyStopper, FitResult, History, _monitor_value, build_optimizer, build_scheduler,
                       load_checkpoint, run_eval_epoch, run_train_epoch, save_checkpoint, set_all_seeds)
from src.utils import save_json

from .model_ext import build_model_ext


def fit_ext(model_name: str, train_loader, val_loader, class_names, cfg: dict, seed: int, run_name: str,
           checkpoint_dir: Path, metrics_dir: Path, device: torch.device, max_epochs: int | None = None,
           normalization_stats: dict | None = None, channels=None, quiet: bool = False,
           model_factory=None) -> FitResult:
    """Same protocol as `src.train.fit` (checkpoint_mode="best" only -- every extension experiment is a
    Phase-1-style validation run, never a Phase-2-style final-only run), but builds the model from
    `model_ext.build_model_ext` instead of `src.model.build_model`. Pass `model_factory` (a zero-arg
    callable returning an unbuilt nn.Module) to use a different builder, e.g.
    `model_ext.build_sensor_ablation_model(in_channels=3)` for the RQ4 sensor-ablation arms, whose input
    width differs from Linear/D's fixed 6 channels.
    """
    set_all_seeds(seed)
    model = (model_factory() if model_factory is not None else build_model_ext(model_name)).to(device)
    optimizer = build_optimizer(model, cfg)
    scheduler = build_scheduler(optimizer, cfg)
    criterion = nn.CrossEntropyLoss()
    es_cfg = cfg["early_stopping"]
    stopper = EarlyStopper(mode=es_cfg.get("mode", "max"), patience=es_cfg["patience"])
    history = History()

    max_epochs = max_epochs if max_epochs is not None else cfg["max_epochs"]
    ckpt_path = Path(checkpoint_dir) / f"{run_name}.pt"
    t0 = time.time()
    epoch = 0
    for epoch in range(1, max_epochs + 1):
        tr = run_train_epoch(model, train_loader, optimizer, criterion, device)
        va, _, _ = run_eval_epoch(model, val_loader, criterion, device, class_names)
        lr_now = optimizer.param_groups[0]["lr"]
        history.train_loss.append(tr.loss); history.train_acc.append(tr.accuracy)
        history.val_loss.append(va.loss); history.val_acc.append(va.accuracy)
        history.val_macro_f1.append(va.macro_f1); history.lr.append(lr_now)

        monitor_value = _monitor_value(es_cfg["monitor"], tr, va)
        is_best = stopper.step(monitor_value, epoch)
        if is_best:
            save_checkpoint(ckpt_path, model, {
                "model_name": model_name, "seed": seed, "epoch": epoch, "run_name": run_name,
                "val_loss": va.loss, "val_acc": va.accuracy, "val_macro_f1": va.macro_f1,
                "class_names": list(class_names), "channels": list(channels) if channels else None,
                "normalization_stats": normalization_stats, "checkpoint_mode": "best",
            })
        if scheduler is not None:
            sched_monitor_name = cfg.get("scheduler", {}).get("monitor", es_cfg["monitor"])
            scheduler.step(_monitor_value(sched_monitor_name, tr, va))
        if not quiet:
            tag = "*best*" if is_best else ""
            print(f"  [{run_name}] epoch {epoch:3d}/{max_epochs}  train_loss={tr.loss:.4f} train_acc={tr.accuracy:.4f}  "
                 f"eval_loss={va.loss:.4f} eval_acc={va.accuracy:.4f} eval_macroF1={va.macro_f1:.4f}  lr={lr_now:.2e}"
                 + (f"  {tag}" if tag else ""))
        if stopper.should_stop:
            if not quiet:
                print(f"  [{run_name}] early stopping at epoch {epoch} (best epoch {stopper.best_epoch}, "
                     f"{es_cfg['patience']} epochs without improvement in {es_cfg['monitor']})")
            break

    elapsed = time.time() - t0
    curves_path = Path(metrics_dir) / f"{run_name}_curves.json"
    save_json({"run_name": run_name, "model_name": model_name, "seed": seed, "history": history.to_dict(),
              "best_epoch": stopper.best_epoch, "epochs_ran": epoch, "train_seconds": elapsed,
              "checkpoint_mode": "best"}, curves_path)

    ckpt = load_checkpoint(ckpt_path)
    return FitResult(run_name, model_name, seed, str(device), stopper.best_epoch, epoch,
                     epoch < max_epochs, elapsed, ckpt["val_loss"], ckpt["val_acc"], ckpt["val_macro_f1"],
                     str(ckpt_path), str(curves_path))
