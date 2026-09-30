"""CSV experiment log + Phase-1 architecture/epoch-budget selection. No torch dependency, so this is
testable without a real training run (see tests/test_experiment_log.py).
"""
from __future__ import annotations

import csv
import statistics
from pathlib import Path

LOG_FIELDS = ["run_name", "model", "seed", "device", "best_epoch", "epochs_ran", "stopped_early",
             "train_seconds", "best_val_loss", "best_val_acc", "best_val_macro_f1", "checkpoint_path", "curves_path"]


def write_log(rows: list, path: Path) -> None:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with open(path, "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=LOG_FIELDS)
        w.writeheader()
        w.writerows(rows)


def read_log(path: Path) -> list:
    with open(path, newline="") as f:
        return list(csv.DictReader(f))


def select_architecture(rows: list) -> dict:
    """rows: list of dicts with at least 'model', 'best_val_macro_f1', 'best_epoch' (as in LOG_FIELDS).
    Selects the architecture with the highest MEAN validation macro-F1 across its seeds, and derives a
    fixed Phase-2 epoch budget for it from the mean of its Phase-1 best_epoch values.
    """
    if not rows:
        raise ValueError("select_architecture called with no rows")
    by_model, epochs_by_model = {}, {}
    for r in rows:
        by_model.setdefault(r["model"], []).append(float(r["best_val_macro_f1"]))
        epochs_by_model.setdefault(r["model"], []).append(float(r["best_epoch"]))
    summary = {}
    for m, v in by_model.items():
        eps = epochs_by_model[m]
        summary[m] = {"mean_val_macro_f1": statistics.mean(v), "std_val_macro_f1": statistics.pstdev(v) if len(v) > 1 else 0.0,
                     "min": min(v), "max": max(v), "n_seeds": len(v), "values": v,
                     "best_epoch_per_seed": eps, "mean_best_epoch": statistics.mean(eps),
                     "recommended_phase2_epochs": max(1, round(statistics.mean(eps)))}
    selected = max(summary, key=lambda m: summary[m]["mean_val_macro_f1"])
    return {"summary": summary, "selected_model": selected,
           "phase2_epochs": summary[selected]["recommended_phase2_epochs"],
           "selection_rule": "architecture with the highest MEAN validation macro-F1 across its seeds",
           "phase2_epochs_rule": "round(mean of Phase-1 best_epoch across that architecture's seeds); "
                                 "fixed before Phase 2 runs, not tuned during Phase 2 (Phase 2 has no validation set)"}
