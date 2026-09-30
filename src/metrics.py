"""Evaluation metrics (sklearn-backed). Pure numpy in/out, no torch dependency, so it is testable
without torch and reusable identically by the training loop, the final evaluation, and error analysis.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from sklearn.metrics import (accuracy_score, confusion_matrix, f1_score,
                             precision_recall_fscore_support)


@dataclass
class Metrics:
    accuracy: float
    macro_f1: float
    weighted_f1: float
    per_class: list          # [{name, precision, recall, f1, support}, ...]
    confusion: np.ndarray     # (K, K) int, rows = true, cols = predicted
    class_names: tuple
    n_samples: int

    def to_dict(self) -> dict:
        return {"accuracy": self.accuracy, "macro_f1": self.macro_f1, "weighted_f1": self.weighted_f1,
                "n_samples": self.n_samples, "class_names": list(self.class_names),
                "per_class": self.per_class, "confusion_matrix": self.confusion.tolist()}


def compute_metrics(y_true: np.ndarray, y_pred: np.ndarray, class_names) -> Metrics:
    """All labels 0..K-1 must be present in `class_names`'s index space; `labels=` is passed explicitly
    to sklearn everywhere so a class absent from a particular batch/split still gets a (zero) row/column
    rather than silently shifting the matrix."""
    class_names = tuple(class_names)
    k = len(class_names)
    labels = list(range(k))
    if y_true.size == 0:
        raise ValueError("compute_metrics called with zero samples")
    if y_true.min() < 0 or y_true.max() >= k or y_pred.min() < 0 or y_pred.max() >= k:
        raise ValueError(f"labels outside [0, {k - 1}]: true in [{y_true.min()},{y_true.max()}], "
                         f"pred in [{y_pred.min()},{y_pred.max()}]")
    acc = accuracy_score(y_true, y_pred)
    macro = f1_score(y_true, y_pred, labels=labels, average="macro", zero_division=0)
    weighted = f1_score(y_true, y_pred, labels=labels, average="weighted", zero_division=0)
    p, r, f, support = precision_recall_fscore_support(y_true, y_pred, labels=labels, zero_division=0)
    per_class = [{"name": class_names[i], "precision": float(p[i]), "recall": float(r[i]),
                 "f1": float(f[i]), "support": int(support[i])} for i in range(k)]
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    return Metrics(float(acc), float(macro), float(weighted), per_class, cm, class_names, int(len(y_true)))


def most_confused_pairs(m: Metrics, top_k: int = 5) -> list:
    """Off-diagonal (true, predicted) pairs, sorted by count desc, as a fraction of that true class."""
    cm = m.confusion.astype(np.float64)
    row_sums = cm.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    frac = cm / row_sums
    pairs = []
    for i in range(len(m.class_names)):
        for j in range(len(m.class_names)):
            if i != j and m.confusion[i, j] > 0:
                pairs.append({"true": m.class_names[i], "predicted": m.class_names[j],
                             "count": int(m.confusion[i, j]), "fraction_of_true_class": float(frac[i, j])})
    pairs.sort(key=lambda d: d["count"], reverse=True)
    return pairs[:top_k]


def class_extremes(m: Metrics) -> dict:
    by_f1 = sorted(m.per_class, key=lambda d: d["f1"])
    return {"lowest_f1": by_f1[0], "highest_f1": by_f1[-1]}


def group_confusion_breakdown(m: Metrics, groups: dict) -> dict:
    """Break all misclassifications down by which activity GROUP the true and predicted classes belong
    to (e.g. groups={"static": [...], "dynamic": [...]}), to answer "are errors concentrated within
    static activities, within dynamic activities, or across the static/dynamic boundary?" -- a question
    the raw confusion matrix answers only implicitly.

    Every class name must appear in exactly one group (this is checked); a class not covered by any
    group is an error, not silently ignored, since an uncovered class would make totals not add up.
    """
    name_to_group = {}
    for g, names in groups.items():
        for n in names:
            if n in name_to_group:
                raise ValueError(f"class '{n}' appears in more than one group ({name_to_group[n]} and {g})")
            name_to_group[n] = g
    missing = [n for n in m.class_names if n not in name_to_group]
    if missing:
        raise ValueError(f"classes not covered by any group: {missing}")

    pair_counts: dict = {}
    total_errors = 0
    for i, ti in enumerate(m.class_names):
        for j, tj in enumerate(m.class_names):
            if i == j:
                continue
            c = int(m.confusion[i, j])
            if c == 0:
                continue
            key = (name_to_group[ti], name_to_group[tj])
            pair_counts[key] = pair_counts.get(key, 0) + c
            total_errors += c

    within = {g: sum(c for (a, b), c in pair_counts.items() if a == g and b == g) for g in groups}
    across = total_errors - sum(within.values())
    return {"groups": {g: list(names) for g, names in groups.items()}, "total_errors": total_errors,
           "within_group_errors": within, "across_group_errors": across,
           "within_group_fraction": {g: (v / total_errors if total_errors else 0.0) for g, v in within.items()},
           "across_group_fraction": (across / total_errors if total_errors else 0.0),
           "group_pair_counts": {f"{a}->{b}": c for (a, b), c in sorted(pair_counts.items(), key=lambda kv: -kv[1])}}
