"""Training loop shared by Phase 1 (architecture comparison) and Phase 2 (final model).

Kept deliberately simple and identical across A/B/C and both phases: same optimizer, loss, scheduler,
and early-stopping rule everywhere; only the model, the data (Phase 1 vs Phase 2), and the seed change.
"""
from __future__ import annotations

import time
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .metrics import compute_metrics
from .model import build_model


def get_device(prefer: str = "auto") -> torch.device:
    if prefer == "cpu":
        return torch.device("cpu")
    if prefer == "cuda" and not torch.cuda.is_available():
        raise RuntimeError("--device cuda requested but CUDA is not available")
    if torch.cuda.is_available() and prefer in ("auto", "cuda"):
        return torch.device("cuda")
    return torch.device("cpu")


def build_optimizer(model: nn.Module, cfg: dict) -> torch.optim.Optimizer:
    name = cfg.get("optimizer", "adam").lower()
    if name != "adam":
        raise ValueError(f"Only 'adam' is implemented (config asked for '{name}')")
    return torch.optim.Adam(model.parameters(), lr=cfg["lr"], weight_decay=cfg.get("weight_decay", 0.0))


def build_scheduler(optimizer: torch.optim.Optimizer, cfg: dict):
    sched_cfg = cfg.get("scheduler")
    if not sched_cfg:
        return None
    if sched_cfg["type"] != "reduce_on_plateau":
        raise ValueError(f"Only 'reduce_on_plateau' is implemented (config asked for '{sched_cfg['type']}')")
    return torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode=sched_cfg.get("mode", "max"), factor=sched_cfg.get("factor", 0.5),
        patience=sched_cfg.get("patience", 7))


def _monitor_value(name: str, tr: "EpochResult", va: "EpochResult") -> float:
    """Look up the named quantity from a train/eval epoch pair. Supports val_macro_f1, val_loss, val_acc,
    train_loss, train_acc -- kept as a single place so early-stopping and the scheduler can watch
    different quantities (Phase 2 watches train_loss for the scheduler since it has no validation set)
    without accidentally sharing the wrong value (see fit())."""
    return {"val_macro_f1": va.macro_f1, "val_loss": va.loss, "val_acc": va.accuracy,
           "train_loss": tr.loss, "train_acc": tr.accuracy}[name]


class EarlyStopper:
    """Stops when `monitor` hasn't improved for `patience` epochs. Ties (equal metric) do NOT reset
    patience, so a run that plateaus exactly does still stop -- only a strict improvement resets it."""

    def __init__(self, mode: str = "max", patience: int = 15):
        if mode not in ("max", "min"):
            raise ValueError(mode)
        self.mode, self.patience = mode, patience
        self.best = -float("inf") if mode == "max" else float("inf")
        self.best_epoch = -1
        self.n_bad_epochs = 0

    def is_better(self, value: float) -> bool:
        return value > self.best if self.mode == "max" else value < self.best

    def step(self, value: float, epoch: int) -> bool:
        """Returns True if this is a new best (caller should checkpoint)."""
        if self.is_better(value):
            self.best, self.best_epoch, self.n_bad_epochs = value, epoch, 0
            return True
        self.n_bad_epochs += 1
        return False

    @property
    def should_stop(self) -> bool:
        return self.n_bad_epochs > self.patience


@dataclass
class EpochResult:
    loss: float
    accuracy: float
    macro_f1: float = float("nan")   # only computed for eval epochs (val/test), not plain train epochs


@dataclass
class History:
    train_loss: list = field(default_factory=list)
    train_acc: list = field(default_factory=list)
    val_loss: list = field(default_factory=list)
    val_acc: list = field(default_factory=list)
    val_macro_f1: list = field(default_factory=list)
    lr: list = field(default_factory=list)

    def to_dict(self) -> dict:
        return self.__dict__


def run_train_epoch(model, loader, optimizer, criterion, device) -> EpochResult:
    model.train()
    total_loss, n_correct, n = 0.0, 0, 0
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        optimizer.zero_grad()
        logits = model(xb)
        loss = criterion(logits, yb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item() * xb.size(0)
        n_correct += (logits.argmax(1) == yb).sum().item()
        n += xb.size(0)
    return EpochResult(total_loss / n, n_correct / n)


@torch.no_grad()
def run_eval_epoch(model, loader, criterion, device, class_names) -> tuple[EpochResult, np.ndarray, np.ndarray]:
    model.eval()
    total_loss, n = 0.0, 0
    all_pred, all_true = [], []
    for xb, yb in loader:
        xb, yb = xb.to(device, non_blocking=True), yb.to(device, non_blocking=True)
        logits = model(xb)
        loss = criterion(logits, yb)
        total_loss += loss.item() * xb.size(0)
        n += xb.size(0)
        all_pred.append(logits.argmax(1).cpu().numpy())
        all_true.append(yb.cpu().numpy())
    y_pred, y_true = np.concatenate(all_pred), np.concatenate(all_true)
    m = compute_metrics(y_true, y_pred, class_names)
    acc = float((y_pred == y_true).mean())
    return EpochResult(total_loss / n, acc, m.macro_f1), y_pred, y_true


def save_checkpoint(path: Path, model: nn.Module, extra: dict) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save({"state_dict": model.state_dict(), **extra}, path)


def load_checkpoint(path: Path, device=None) -> dict:
    return torch.load(Path(path), map_location=device or "cpu", weights_only=False)


def set_all_seeds(seed: int) -> None:
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


@dataclass
class FitResult:
    run_name: str
    model_name: str
    seed: int
    device: str
    best_epoch: int
    epochs_ran: int
    stopped_early: bool
    train_seconds: float
    best_val_loss: float
    best_val_acc: float
    best_val_macro_f1: float
    checkpoint_path: str
    curves_path: str


def fit(model_name: str, train_loader, val_loader, class_names, cfg: dict, seed: int, run_name: str,
       checkpoint_dir: Path, metrics_dir: Path, device: torch.device, max_epochs: int | None = None,
       normalization_stats: dict | None = None, channels=None, quiet: bool = False,
       checkpoint_mode: str = "best") -> FitResult:
    """Train one model once (one architecture, one seed). Used identically by Phase 1 (per-seed sweep)
    and Phase 2 (single final run) -- only the loaders/run_name/checkpoint location/checkpoint_mode differ.

    checkpoint_mode="best" (Phase 1): save whenever `early_stopping.monitor` improves on val_loader,
        and early-stop when it hasn't improved for `early_stopping.patience` epochs. This is what a
        held-out validation set is for.
    checkpoint_mode="last" (Phase 2): there is no held-out validation set in Phase 2 by design (see
        scripts/train_final.py), so nothing here is a legitimate model-selection signal. Instead, train
        for exactly `max_epochs` (fixed beforehand from the Phase-1 result, not tuned here) and save
        whatever the model is after the final epoch -- no in-training checkpoint selection.
    """
    if checkpoint_mode not in ("best", "last"):
        raise ValueError(checkpoint_mode)
    set_all_seeds(seed)
    model = build_model(model_name).to(device)
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
        save_now = is_best if checkpoint_mode == "best" else (epoch == max_epochs)
        if save_now:
            save_checkpoint(ckpt_path, model, {
                "model_name": model_name, "seed": seed, "epoch": epoch, "run_name": run_name,
                "val_loss": va.loss, "val_acc": va.accuracy, "val_macro_f1": va.macro_f1,
                "class_names": list(class_names), "channels": list(channels) if channels else None,
                "normalization_stats": normalization_stats, "checkpoint_mode": checkpoint_mode,
            })
        if scheduler is not None:
            sched_monitor_name = cfg.get("scheduler", {}).get("monitor", es_cfg["monitor"])
            scheduler.step(_monitor_value(sched_monitor_name, tr, va))
        if not quiet:
            tag = "*best*" if (checkpoint_mode == "best" and is_best) else ("*saved*" if save_now else "")
            print(f"  [{run_name}] epoch {epoch:3d}/{max_epochs}  train_loss={tr.loss:.4f} train_acc={tr.accuracy:.4f}  "
                 f"eval_loss={va.loss:.4f} eval_acc={va.accuracy:.4f} eval_macroF1={va.macro_f1:.4f}  lr={lr_now:.2e}"
                 + (f"  {tag}" if tag else ""))
        if checkpoint_mode == "best" and stopper.should_stop:
            if not quiet:
                print(f"  [{run_name}] early stopping at epoch {epoch} (best epoch {stopper.best_epoch}, "
                     f"{es_cfg['patience']} epochs without improvement in {es_cfg['monitor']})")
            break

    elapsed = time.time() - t0
    curves_path = Path(metrics_dir) / f"{run_name}_curves.json"
    from .utils import save_json
    save_json({"run_name": run_name, "model_name": model_name, "seed": seed, "history": history.to_dict(),
              "best_epoch": stopper.best_epoch, "epochs_ran": epoch, "train_seconds": elapsed,
              "checkpoint_mode": checkpoint_mode}, curves_path)

    ckpt = load_checkpoint(ckpt_path)
    reported_best_epoch = stopper.best_epoch if checkpoint_mode == "best" else epoch
    return FitResult(run_name, model_name, seed, str(device), reported_best_epoch, epoch,
                     epoch < max_epochs, elapsed, ckpt["val_loss"], ckpt["val_acc"], ckpt["val_macro_f1"],
                     str(ckpt_path), str(curves_path))
