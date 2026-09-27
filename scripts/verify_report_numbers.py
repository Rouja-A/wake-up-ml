#!/usr/bin/env python
"""Static, read-only consistency checks for the frozen experimental record. Runs NO inference and NO
training: it only re-derives arithmetic (accuracy/F1/error-group counts from a confusion matrix; MAC/RF
counts from the architecture spec) and checks it against what is saved on disk, plus a couple of
structural sanity checks (row counts, parameter budget).

    python scripts/verify_report_numbers.py [--metrics_dir results/metrics] [--predictions_csv ...]

Every check here is either (a) pure arithmetic on numbers already saved to disk, requiring no model or
dataset access at all, or (b) a recount from a saved predictions CSV. If the real result files
(results/metrics/final_test_evaluation.json, results/metrics/final_test_predictions.csv) are not present
in this checkout, the corresponding checks are skipped and reported as such -- they are not invented.
"""
import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.architectures import ARCHITECTURES, PARAM_BUDGET   # noqa: E402
from src import shape_calc as S                              # noqa: E402
from src.utils import Checker                                # noqa: E402


def check_param_budget(chk: Checker) -> None:
    print("== Parameter budget (A/B/C, from src/architectures.py) ==")
    for name, spec in ARCHITECTURES.items():
        rep = S.analyze(spec, name=name)
        chk.check(rep.total_params < PARAM_BUDGET, f"Model {name}: {rep.total_params} < {PARAM_BUDGET}")


def check_confusion_matrix_arithmetic(cm, class_names, reported: dict, chk: Checker) -> None:
    """Recompute accuracy/macro-F1/weighted-F1/per-class P,R,F1/error-group split from a confusion
    matrix ALONE and compare against numbers already reported alongside it. This never touches a model
    or the dataset -- it is pure arithmetic on numbers already on disk."""
    import numpy as np
    cm = np.asarray(cm)
    chk.check(cm.shape == (6, 6), "confusion matrix is 6x6", str(cm.shape))
    n = int(cm.sum())
    support = cm.sum(axis=1)
    pred_totals = cm.sum(axis=0)
    with np.errstate(divide="ignore", invalid="ignore"):
        recall = np.where(support > 0, np.diag(cm) / support, 0.0)
        precision = np.where(pred_totals > 0, np.diag(cm) / pred_totals, 0.0)
        f1 = np.where((precision + recall) > 0, 2 * precision * recall / (precision + recall), 0.0)
    acc = np.diag(cm).sum() / n
    macro_f1 = f1.mean()
    weighted_f1 = (f1 * support).sum() / n

    print(f"  n = {n}")
    chk.check(n == reported.get("n_samples", n), "confusion matrix total matches reported n_samples", f"{n} vs {reported.get('n_samples')}")
    chk.check(abs(acc - reported["accuracy"]) < 5e-4, "accuracy recomputed from confusion matrix matches reported value",
             f"{acc:.6f} vs {reported['accuracy']}")
    chk.check(abs(macro_f1 - reported["macro_f1"]) < 5e-4, "macro-F1 recomputed from confusion matrix matches reported value",
             f"{macro_f1:.6f} vs {reported['macro_f1']}")
    chk.check(abs(weighted_f1 - reported["weighted_f1"]) < 5e-4, "weighted-F1 recomputed from confusion matrix matches reported value",
             f"{weighted_f1:.6f} vs {reported['weighted_f1']}")

    static = {"SITTING", "STANDING", "LAYING"}
    dynamic = {"WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"}
    within_static = within_dynamic = across = 0
    for i, ti in enumerate(class_names):
        for j, tj in enumerate(class_names):
            if i == j or cm[i, j] == 0:
                continue
            c = int(cm[i, j])
            if ti in static and tj in static:
                within_static += c
            elif ti in dynamic and tj in dynamic:
                within_dynamic += c
            else:
                across += c
    total_errors = within_static + within_dynamic + across
    chk.check(total_errors == n - int(np.diag(cm).sum()), "error-group counts sum to total misclassified windows")
    print(f"  within_static={within_static}  within_dynamic={within_dynamic}  across={across}  total_errors={total_errors}")

    if "per_class" in reported:
        support_ok = all(int(reported["per_class"][i]["support"]) == int(support[i]) for i in range(len(class_names)))
        chk.check(support_ok, "per-class 'support' in the saved metrics matches confusion-matrix row sums",
                 f"{[c['support'] for c in reported['per_class']]} vs {support.tolist()}")


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--metrics_dir", default="results/metrics")
    args = ap.parse_args()
    chk = Checker()

    check_param_budget(chk)

    eval_path = Path(args.metrics_dir) / "final_test_evaluation.json"
    if eval_path.is_file():
        print(f"\n== Recomputing metrics from {eval_path} ==")
        data = json.loads(eval_path.read_text())
        check_confusion_matrix_arithmetic(data["confusion_matrix"], data["class_names"], data, chk)
    else:
        chk.warn(f"{eval_path} not found in this checkout", "skipping confusion-matrix arithmetic check -- "
                "merge the real Step-5 results into results/metrics/ before submission and re-run this script")

    sel_path = Path(args.metrics_dir) / "architecture_selection.json"
    if sel_path.is_file():
        print(f"\n== Checking {sel_path} ==")
        sel = json.loads(sel_path.read_text())
        chk.check(sel["selected_model"] == "C", "selected architecture is C", sel["selected_model"])
        chk.check(sel["phase2_epochs"] == 1, "fixed Phase-2 epoch budget derived from Phase 1 is 1", str(sel["phase2_epochs"]))
    else:
        chk.warn(f"{sel_path} not found in this checkout", "skipping architecture-selection check")

    info_path = Path(args.metrics_dir) / "final_model_info.json"
    if info_path.is_file():
        print(f"\n== Checking {info_path} ==")
        info = json.loads(info_path.read_text())
        chk.check(info["model_name"] == "C", "Phase-2 final model is C", info["model_name"])
        chk.check(info["seed"] == 123, "Phase-2 final seed is 123", str(info["seed"]))
        chk.check(info["epochs"] == 1, "Phase-2 trained for exactly 1 epoch", str(info["epochs"]))
        chk.check("phase2_final_C_seed123" in info.get("checkpoint_path", ""),
                 "checkpoint path names model/seed consistently", info.get("checkpoint_path"))
        if sel_path.is_file():
            chk.check(info["epochs"] == sel["phase2_epochs"],
                     "final_model_info.json epoch count matches architecture_selection.json's derived budget")
    else:
        chk.warn(f"{info_path} not found in this checkout", "skipping Phase-2 metadata check")

    stream_path = Path(args.metrics_dir) / "streaming_analysis.json"
    if stream_path.is_file():
        print(f"\n== Cross-checking {stream_path} against src/architectures.py ==")
        stream = json.loads(stream_path.read_text())
        rep = S.analyze(ARCHITECTURES["C"], name="C")
        struct = stream["model_C_structure"]
        chk.check(struct["trainable_params"]["value"] == rep.total_params, "streaming_analysis params match recomputed Model C")
        chk.check(struct["macs_per_window"]["value"] == rep.total_macs, "streaming_analysis MACs match recomputed Model C")
        chk.check(struct["local_receptive_field_samples"]["value"] == rep.receptive_field,
                 "streaming_analysis local RF matches recomputed Model C")
    else:
        chk.warn(f"{stream_path} not found in this checkout", "skipping streaming-analysis cross-check")

    pred_path = Path(args.metrics_dir) / "final_test_predictions.csv"
    if pred_path.is_file():
        print(f"\n== Checking {pred_path} ==")
        with open(pred_path, newline="") as f:
            rows = list(csv.DictReader(f))
        chk.check(len(rows) == 2947, "predictions CSV has exactly 2947 data rows (documented UCI HAR test-window count)",
                 str(len(rows)))
        if eval_path.is_file():
            import numpy as np
            cm_expected = np.asarray(data["confusion_matrix"])
            names = data["class_names"]
            cm_recount = np.zeros_like(cm_expected)
            for r in rows:
                cm_recount[int(r["true_label"]), int(r["pred_label"])] += 1
            chk.check(np.array_equal(cm_recount, cm_expected),
                     "confusion matrix recounted from the predictions CSV matches the saved confusion matrix exactly")
    else:
        chk.warn(f"{pred_path} not found in this checkout", "skipping row-count/recount check -- "
                "merge the real Step-5 results into results/metrics/ before submission and re-run this script")

    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
