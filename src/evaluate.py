"""Load a trained checkpoint and evaluate it on a given DataLoader. Used by scripts/evaluate_final.py
for the single, official test-set evaluation, and reusable for validation-set evaluation if needed.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import torch

from .metrics import Metrics, class_extremes, compute_metrics, most_confused_pairs
from .model import build_model
from .train import load_checkpoint


@torch.no_grad()
def predict(model, loader, device) -> tuple[np.ndarray, np.ndarray]:
    model.eval()
    preds, trues = [], []
    for xb, yb in loader:
        xb = xb.to(device, non_blocking=True)
        logits = model(xb)
        preds.append(logits.argmax(1).cpu().numpy())
        trues.append(yb.numpy())
    return np.concatenate(preds), np.concatenate(trues)


def load_model_from_checkpoint(checkpoint_path, device) -> tuple[torch.nn.Module, dict]:
    ckpt = load_checkpoint(checkpoint_path, device)
    model = build_model(ckpt["model_name"]).to(device)
    model.load_state_dict(ckpt["state_dict"])
    model.eval()
    return model, ckpt


def evaluate(checkpoint_path, loader, device, class_names=None) -> tuple[Metrics, dict]:
    model, ckpt = load_model_from_checkpoint(checkpoint_path, device)
    names = tuple(class_names) if class_names is not None else tuple(ckpt["class_names"])
    y_pred, y_true = predict(model, loader, device)
    m = compute_metrics(y_true, y_pred, names)
    analysis = {"most_confused_pairs": most_confused_pairs(m, top_k=5), **class_extremes(m)}
    return m, analysis


def plot_confusion_matrix(m: Metrics, path: Path, title: str = "Confusion matrix") -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import seaborn as sns

    row_sums = m.confusion.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1
    normed = m.confusion / row_sums

    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, data, fmt, sub in ((axes[0], m.confusion, "d", "counts"), (axes[1], normed, ".2f", "row-normalised (recall)")):
        sns.heatmap(data, annot=True, fmt=fmt, cmap="Blues", xticklabels=m.class_names, yticklabels=m.class_names,
                   cbar=False, ax=ax, square=True)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sub)
        ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{title}  (accuracy={m.accuracy:.3f}, macro F1={m.macro_f1:.3f}, n={m.n_samples})")
    fig.tight_layout()
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=140)
    plt.close(fig)
