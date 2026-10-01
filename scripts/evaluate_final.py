#!/usr/bin/env python
"""Phase 2, step 2: the SINGLE, final evaluation on the official UCI HAR test subjects.

Run this exactly once, after scripts/train_final.py has produced the final checkpoint. Refuses to
overwrite a previous test evaluation. The submitted test result is frozen. A repeat requires the deliberately
verbose --acknowledge-repeat-test flag and must be treated as post-hoc/exploratory rather than a replacement
for the original one-shot result.

    python scripts/evaluate_final.py --config configs/baseline.yaml [--data_dir ...]
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                                               # noqa: E402
from src.dataset import make_test_loader_from_stats                     # noqa: E402
from src.evaluate import load_model_from_checkpoint, plot_confusion_matrix, predict  # noqa: E402
from src.metrics import class_extremes, compute_metrics, group_confusion_breakdown, most_confused_pairs  # noqa: E402
from src.preprocessing import ChannelStats                              # noqa: E402
from src.train import get_device, load_checkpoint                       # noqa: E402
from src.utils import default_data_dir, load_config, save_json          # noqa: E402

# The known, fixed UCI HAR class grouping used for the static-vs-dynamic failure-analysis breakdown.
# This is a property of the task/labels, not a modeling choice, so it is not read from config.
ACTIVITY_GROUPS = {"static": ["SITTING", "STANDING", "LAYING"],
                   "dynamic": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--metrics_dir", default="results/metrics")
    ap.add_argument("--figures_dir", default="results/figures")
    ap.add_argument("--acknowledge-repeat-test", action="store_true",
                    help="explicitly acknowledge a repeated official-test evaluation; post-hoc only")
    args = ap.parse_args()

    out_path = Path(args.metrics_dir) / "final_test_evaluation.json"
    if out_path.is_file() and not args.acknowledge_repeat_test:
        print(f"REFUSING to run: {out_path} already exists, meaning the official test set has already been "
             f"evaluated once for this final model. Re-running it would defeat the "
             f"point of evaluating the test set only once. If you intentionally need a post-hoc repeat, use "
             f"--acknowledge-repeat-test and do not present it as a replacement one-shot test result.")
        return 1

    info_path = Path(args.metrics_dir) / "final_model_info.json"
    if not info_path.is_file():
        raise SystemExit(f"{info_path} not found. Run scripts/train_final.py (Phase 2 training) first.")
    info = json.loads(info_path.read_text())
    print(f"Final model: {info['model_name']}  (checkpoint: {info['checkpoint_path']})")

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()

    # Load the SAVED normalization statistics from the Phase-2 checkpoint (never recomputed here, and no
    # training data is loaded in this script at all -- only the test split, plus these saved statistics).
    ckpt = load_checkpoint(info["checkpoint_path"])
    stats = ChannelStats.from_dict(ckpt["normalization_stats"])
    channels = tuple(ckpt["channels"])
    class_names = tuple(ckpt["class_names"])
    print(f"Normalization statistics: loaded from the Phase-2 checkpoint "
         f"(fit on {stats.n_windows} official-train windows; not recomputed here).")

    # Load ONLY the official test split -- src.data.load_test_only never opens any train-split file.
    test_raw, class_names_from_file = D.load_test_only(data_dir, channels)
    if class_names_from_file != class_names:
        raise SystemExit(f"activity_labels.txt order ({class_names_from_file}) does not match the "
                         f"checkpoint's recorded class_names ({class_names}); refusing to evaluate.")
    print(f"Official test subjects: {sorted(int(s) for s in set(test_raw.subjects.tolist()))}  "
         f"({len(test_raw)} windows)")

    device = get_device(args.device)
    test_loader = make_test_loader_from_stats(test_raw.x, test_raw.y, stats, batch_size=cfg["training"]["batch_size"])

    # Single inference pass over the whole test set (model.eval() + torch.no_grad(), no weight update of
    # any kind -- see src/evaluate.py:predict). shuffle=False in make_test_loader_from_stats means y_pred
    # and y_true come back in the SAME order as test_raw, so they can be zipped with test_raw.subjects and
    # a window index below without re-running inference.
    model, ckpt_loaded = load_model_from_checkpoint(info["checkpoint_path"], device)
    y_pred, y_true = predict(model, test_loader, device)
    assert len(y_pred) == len(y_true) == len(test_raw), "predict() must return exactly one label per test window"
    if len(test_raw) != 2947:
        print(f"  [NOTE] test set has {len(test_raw)} windows, not the documented 2947 -- expected on "
             f"synthetic/non-UCI data; verify this is intentional if using the real dataset.")

    m = compute_metrics(y_true, y_pred, class_names)
    analysis = {"most_confused_pairs": most_confused_pairs(m, top_k=30),  # 30 = all possible off-diagonal
                                                                         # cells for 6 classes, i.e. "all of them"
               **class_extremes(m), "activity_group_breakdown": group_confusion_breakdown(m, ACTIVITY_GROUPS)}

    print(f"\n=== FINAL TEST RESULT ({info['model_name']}, n={m.n_samples}) ===")
    print(f"  accuracy     : {m.accuracy:.4f}")
    print(f"  macro F1     : {m.macro_f1:.4f}")
    print(f"  weighted F1  : {m.weighted_f1:.4f}")
    print(f"\n  {'class':<22}{'precision':>10}{'recall':>10}{'f1':>10}{'support':>10}")
    for c in m.per_class:
        print(f"  {c['name']:<22}{c['precision']:>10.3f}{c['recall']:>10.3f}{c['f1']:>10.3f}{c['support']:>10}")

    print(f"\n  Confusion matrix (rows=true, cols=predicted):")
    header = " " * 22 + "".join(f"{n[:10]:>11}" for n in m.class_names)
    print(header)
    for i, n in enumerate(m.class_names):
        print(f"  {n:<20}" + "".join(f"{v:>11d}" for v in m.confusion[i]))

    print(f"\n  Most confused pairs (all, not just top 5):")
    for p in analysis["most_confused_pairs"]:
        print(f"    {p['true']:<20} -> {p['predicted']:<20} {p['count']:4d}  ({100*p['fraction_of_true_class']:.1f}% of true class)")
    print(f"  Lowest F1 : {analysis['lowest_f1']['name']} ({analysis['lowest_f1']['f1']:.3f})")
    print(f"  Highest F1: {analysis['highest_f1']['name']} ({analysis['highest_f1']['f1']:.3f})")

    gb = analysis["activity_group_breakdown"]
    print(f"\n  Error breakdown by activity group ({gb['total_errors']} total misclassified windows):")
    for g, count in gb["within_group_errors"].items():
        print(f"    within {g:<8}: {count:4d}  ({100*gb['within_group_fraction'][g]:.1f}% of all errors)")
    print(f"    across groups : {gb['across_group_errors']:4d}  ({100*gb['across_group_fraction']:.1f}% of all errors)")
    for pair, count in gb["group_pair_counts"].items():
        print(f"      {pair:<20} {count}")

    fig_path = Path(args.figures_dir) / "confusion_matrix_final.png"
    plot_confusion_matrix(m, fig_path, title=f"Final test set - Model {info['model_name']}")
    print(f"\nWrote {fig_path}")

    # Per-window predictions (true/pred label + name, subject id, window index), so failure cases can be
    # inspected later WITHOUT re-running inference on the official test set (which the one-time guard
    # above should prevent anyway).
    pred_path = Path(args.metrics_dir) / "final_test_predictions.csv"
    pred_path.parent.mkdir(parents=True, exist_ok=True)
    test_subjects_list = test_raw.subjects.tolist()
    with open(pred_path, "w", newline="") as f:
        w = csv.writer(f)
        w.writerow(["window_index", "subject_id", "true_label", "true_name", "pred_label", "pred_name", "correct"])
        for idx in range(len(y_true)):
            t, p = int(y_true[idx]), int(y_pred[idx])
            w.writerow([idx, int(test_subjects_list[idx]), t, class_names[t], p, class_names[p], t == p])
    print(f"Wrote {pred_path}  ({len(y_true)} rows: one per test window)")

    save_json({"model_name": info["model_name"], "checkpoint_path": info["checkpoint_path"], **m.to_dict(),
              **analysis, "test_subjects": sorted(int(s) for s in set(test_subjects_list)),
              "predictions_csv": str(pred_path)}, out_path)
    print(f"Wrote {out_path}")
    print("\nThis test set has now been evaluated. Do not evaluate it again for this project.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
