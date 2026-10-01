"""CSV experiment log + Phase-1 architecture/epoch-budget selection.

Architecture selection uses mean validation macro-F1 across seeds. The fixed Phase-2 training budget
is the rounded mean of the selected architecture's *actual Phase-1 training durations* (epochs_ran).
This keeps the final-training duration development-only while avoiding the earlier mistake of treating
the epoch of peak validation F1 as the amount of optimization required when retraining from scratch.

No torch dependency, so this is testable without a real training run.
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
    """Select by mean validation macro-F1 and derive Phase-2 duration from ``epochs_ran``.

    The Phase-2 rule is fixed from development runs only:
    round(mean(actual Phase-1 training durations for the selected architecture)).
    """
    if not rows:
        raise ValueError("select_architecture called with no rows")
    by_model, best_epochs_by_model, durations_by_model = {}, {}, {}
    for r in rows:
        model = r["model"]
        by_model.setdefault(model, []).append(float(r["best_val_macro_f1"]))
        best_epochs_by_model.setdefault(model, []).append(float(r["best_epoch"]))
        durations_by_model.setdefault(model, []).append(int(float(r["epochs_ran"])))

    summary = {}
    for model, values in by_model.items():
        best_epochs = best_epochs_by_model[model]
        durations = durations_by_model[model]
        phase2_epochs = max(1, round(statistics.mean(durations)))
        summary[model] = {
            "mean_val_macro_f1": statistics.mean(values),
            "std_val_macro_f1": statistics.pstdev(values) if len(values) > 1 else 0.0,
            "min": min(values), "max": max(values), "n_seeds": len(values), "values": values,
            "best_epoch_per_seed": best_epochs,
            "mean_best_epoch": statistics.mean(best_epochs),
            "epochs_ran_per_seed": durations,
            "mean_epochs_ran": statistics.mean(durations),
            "recommended_phase2_epochs": phase2_epochs,
        }

    selected = max(summary, key=lambda m: summary[m]["mean_val_macro_f1"])
    return {
        "summary": summary,
        "selected_model": selected,
        "phase2_epochs": summary[selected]["recommended_phase2_epochs"],
        "selection_rule": "architecture with the highest MEAN validation macro-F1 across its seeds",
        "phase2_epochs_rule": "round(mean of actual Phase-1 epochs_ran across the selected architecture's seeds)",
    }
