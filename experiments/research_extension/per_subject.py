"""Per-validation-subject metric breakdown (RQ: does aggregate validation macro-F1 hide subject-to-
subject variability?). Pure numpy/sklearn, no torch dependency -- the model-specific part (producing
y_pred for the validation set) happens elsewhere; this module only aggregates already-computed
predictions by subject id, so it is trivially testable with synthetic predictions.
"""
from __future__ import annotations

import statistics

import numpy as np

from src.metrics import compute_metrics


def per_subject_metrics(y_true: np.ndarray, y_pred: np.ndarray, subjects: np.ndarray, class_names) -> list[dict]:
    """One row per unique subject id in `subjects` (all three arrays must be the same length, one entry
    per validation window). Rows are sorted by subject id for a stable, reviewable table."""
    if not (len(y_true) == len(y_pred) == len(subjects)):
        raise ValueError(f"length mismatch: y_true={len(y_true)}, y_pred={len(y_pred)}, subjects={len(subjects)}")
    rows = []
    for sid in sorted(int(s) for s in set(subjects.tolist())):
        mask = subjects == sid
        m = compute_metrics(y_true[mask], y_pred[mask], class_names)
        rows.append({"subject_id": sid, "n_windows": int(mask.sum()), "accuracy": m.accuracy,
                    "macro_f1": m.macro_f1, "per_class_support": [c["support"] for c in m.per_class]})
    return rows


def summarize_subject_variability(rows: list[dict]) -> dict:
    """Mean/std/min/max of macro-F1 ACROSS subjects (not across windows) -- the quantity the aggregate
    validation macro-F1 can hide if a small number of subjects are much harder than the rest."""
    if not rows:
        raise ValueError("summarize_subject_variability called with no rows")
    f1s = [r["macro_f1"] for r in rows]
    return {"n_subjects": len(f1s), "mean_subject_macro_f1": statistics.mean(f1s),
           "std_subject_macro_f1": statistics.pstdev(f1s) if len(f1s) > 1 else 0.0,
           "min_subject_macro_f1": min(f1s), "max_subject_macro_f1": max(f1s),
           "worst_subject_id": rows[[r["macro_f1"] for r in rows].index(min(f1s))]["subject_id"],
           "best_subject_id": rows[[r["macro_f1"] for r in rows].index(max(f1s))]["subject_id"]}
