#!/usr/bin/env python
"""Torch-free, one-shot official-test evaluation of a Phase-2 checkpoint.

Equivalent to scripts/evaluate_final.py (same data loader, saved normalisation statistics, metrics, output
files) but runs the forward pass in NumPy, reading the weights straight out of the .pt file. Written so the
final model can be evaluated on a machine without PyTorch.

Implementation check: `--verify_against <predictions.csv>` re-runs the checkpoint that produced that CSV
and requires every prediction to match, proving this NumPy forward pass equals the PyTorch one.

    python scripts/evaluate_final_numpy.py --config configs/baseline.yaml --data_dir "data/UCI HAR Dataset"
"""
import argparse, collections, csv, io, json, pickle, sys, zipfile
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                                                   # noqa: E402
from src.architectures import (ARCHITECTURES, BNOp, ConvOp, GAPOp, LinearOp, PoolOp, ReLUOp)  # noqa: E402
from src.metrics import class_extremes, compute_metrics, group_confusion_breakdown, most_confused_pairs  # noqa: E402
from src.preprocessing import ChannelStats, standardize                    # noqa: E402
from src.utils import save_json                                             # noqa: E402

ACTIVITY_GROUPS = {"static": ["SITTING", "STANDING", "LAYING"],
                   "dynamic": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}
_DT = {"FloatStorage": np.float32, "LongStorage": np.int64, "DoubleStorage": np.float64}


def read_checkpoint(path):
    """Minimal reader for torch.save() zip checkpoints (no torch needed)."""
    z = zipfile.ZipFile(path)
    root = z.namelist()[0].split("/")[0]

    class U(pickle.Unpickler):
        def find_class(self, mod, name):
            if name == "_rebuild_tensor_v2":
                def rebuild(storage, off, size, stride, *a):
                    key, dt = storage
                    buf = np.frombuffer(z.read(f"{root}/data/{key}"), dtype=dt)
                    n = int(np.prod(size)) if len(size) else 1
                    return buf[off:off + n].reshape(size).copy()
                return rebuild
            if mod == "collections" and name == "OrderedDict":
                return collections.OrderedDict
            if mod.startswith("torch"):
                return name
            return super().find_class(mod, name)

        def persistent_load(self, pid):
            return (pid[2], _DT[pid[1]])

    return U(io.BytesIO(z.read(f"{root}/data.pkl"))).load()


def _conv(x, w, b, pad, groups):
    n, c, L = x.shape
    co, cg, k = w.shape
    xp = np.pad(x, ((0, 0), (0, 0), (pad, pad)))
    win = np.lib.stride_tricks.sliding_window_view(xp, k, axis=2)           # (N, C, Lout, k)
    lo = win.shape[2]
    out = np.empty((n, co, lo), dtype=np.float32)
    go = co // groups
    for g in range(groups):
        out[:, g * go:(g + 1) * go] = np.einsum("nclk,ock->nol", win[:, g * cg:(g + 1) * cg], w[g * go:(g + 1) * go])
    return out + b[None, :, None]


def forward(spec, sd, x):
    idx = 0
    for op in spec:
        if isinstance(op, ConvOp):
            x = _conv(x, sd[f"layers.{idx}.weight"], sd[f"layers.{idx}.bias"], op.padding, op.groups)
        elif isinstance(op, BNOp):
            g, b = sd[f"layers.{idx}.weight"], sd[f"layers.{idx}.bias"]
            m, v = sd[f"layers.{idx}.running_mean"], sd[f"layers.{idx}.running_var"]
            x = (x - m[None, :, None]) / np.sqrt(v[None, :, None] + 1e-5) * g[None, :, None] + b[None, :, None]
        elif isinstance(op, ReLUOp):
            x = np.maximum(x, 0)
        elif isinstance(op, PoolOp):
            L = (x.shape[2] - op.kernel) // op.stride + 1
            x = np.max(np.lib.stride_tricks.sliding_window_view(x, op.kernel, axis=2)[:, :, ::op.stride][:, :, :L], axis=3)
        elif isinstance(op, GAPOp):
            x = x.mean(axis=2)
            idx += 1                                                         # Flatten layer sits here in nn.Sequential
        elif isinstance(op, LinearOp):
            x = x @ sd[f"layers.{idx}.weight"].T + sd[f"layers.{idx}.bias"]
        idx += 1
    return x


def plot_confusion_matrix(m, path, title):
    """Same figure as src.evaluate.plot_confusion_matrix (copied to avoid importing torch)."""
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt, seaborn as sns
    normed = m.confusion / np.maximum(m.confusion.sum(axis=1, keepdims=True), 1)
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5))
    for ax, d, fmt, sub in ((axes[0], m.confusion, "d", "counts"), (axes[1], normed, ".2f", "row-normalised (recall)")):
        sns.heatmap(d, annot=True, fmt=fmt, cmap="Blues", xticklabels=m.class_names, yticklabels=m.class_names, cbar=False, ax=ax, square=True)
        ax.set_xlabel("Predicted"); ax.set_ylabel("True"); ax.set_title(sub); ax.tick_params(axis="x", rotation=45)
    fig.suptitle(f"{title}  (accuracy={m.accuracy:.3f}, macro F1={m.macro_f1:.3f}, n={m.n_samples})")
    fig.tight_layout(); Path(path).parent.mkdir(parents=True, exist_ok=True); fig.savefig(path, dpi=140); plt.close(fig)


def main():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", required=True)
    ap.add_argument("--metrics_dir", default="results/metrics")
    ap.add_argument("--figures_dir", default="results/figures")
    ap.add_argument("--checkpoint", default=None, help="default: checkpoint_path in final_model_info.json")
    ap.add_argument("--verify_against", default=None, help="predictions CSV from the same checkpoint; check equality, write nothing")
    ap.add_argument("--acknowledge-repeat-test", action="store_true")
    a = ap.parse_args()

    out_path = Path(a.metrics_dir) / "final_test_evaluation.json"
    if not a.verify_against and out_path.is_file() and not a.acknowledge_repeat_test:
        print(f"REFUSING: {out_path} exists (official test already evaluated once for this model)."); return 1
    ckpt_path = a.checkpoint or json.loads((Path(a.metrics_dir) / "final_model_info.json").read_text())["checkpoint_path"]
    ckpt_path = ckpt_path.replace("\\", "/")
    ck = read_checkpoint(ckpt_path)
    stats = ChannelStats.from_dict({**ck["normalization_stats"]})
    class_names = tuple(ck["class_names"]); channels = tuple(ck["channels"])
    test_raw, names_file = D.load_test_only(a.data_dir, channels)
    assert tuple(names_file) == class_names
    spec = ARCHITECTURES[ck["model_name"]]
    x = standardize(test_raw.x, stats)
    y_pred = np.concatenate([forward(spec, ck["state_dict"], x[i:i + 512]).argmax(1) for i in range(0, len(x), 512)])
    y_true = test_raw.y
    print(f"checkpoint: {ckpt_path}  (model {ck['model_name']}, epoch {ck['epoch']}, mode {ck['checkpoint_mode']})")

    if a.verify_against:
        ref = np.array([int(r["pred_label"]) for r in csv.DictReader(open(a.verify_against))])
        n_diff = int((ref != y_pred).sum())
        print(f"VERIFY: {n_diff} of {len(ref)} predictions differ from the PyTorch-produced CSV")
        return 0 if n_diff == 0 else 2

    m = compute_metrics(y_true, y_pred, class_names)
    analysis = {"most_confused_pairs": most_confused_pairs(m, top_k=30), **class_extremes(m),
                "activity_group_breakdown": group_confusion_breakdown(m, ACTIVITY_GROUPS)}
    print(f"accuracy {m.accuracy:.4f}  macroF1 {m.macro_f1:.4f}  weightedF1 {m.weighted_f1:.4f}")
    print(m.confusion)
    plot_confusion_matrix(m, Path(a.figures_dir) / "confusion_matrix_final.png", title=f"Final test set - Model {ck['model_name']} (17 epochs)")
    subj = test_raw.subjects.tolist()
    with open(Path(a.metrics_dir) / "final_test_predictions.csv", "w", newline="") as f:
        w = csv.writer(f); w.writerow(["window_index", "subject_id", "true_label", "true_name", "pred_label", "pred_name", "correct"])
        for i in range(len(y_true)):
            t, p = int(y_true[i]), int(y_pred[i]); w.writerow([i, int(subj[i]), t, class_names[t], p, class_names[p], t == p])
    save_json({"model_name": ck["model_name"], "checkpoint_path": ckpt_path, "evaluated_with": "numpy forward pass (scripts/evaluate_final_numpy.py)",
               **m.to_dict(), **analysis, "test_subjects": sorted(set(int(s) for s in subj)),
               "predictions_csv": str(Path(a.metrics_dir) / "final_test_predictions.csv")}, out_path)
    print(f"wrote {out_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
