#!/usr/bin/env python
"""Phase 1: architecture comparison. Trains A/B/C x training.phase1_seeds (default 3 seeds each = 9 runs)
on the FIXED Step-2 subject-held-out split (same split for every run; only model init/shuffle seed
varies). Writes results/experiment_log.csv (one row per run) and, at the end, selects the architecture
with the best MEAN validation macro-F1 across its seeds -- writing that decision to
results/metrics/architecture_selection.json. The official test set is never touched here.

    python scripts/run_experiments.py --config configs/baseline.yaml [--data_dir ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as P                                    # noqa: E402
from src.architectures import ARCHITECTURES                            # noqa: E402
from src.experiment_log import LOG_FIELDS, select_architecture, write_log  # noqa: E402
from src.train import get_device                                       # noqa: E402
from src.utils import default_data_dir, load_config, save_json         # noqa: E402
from scripts.train import run_phase1_single                            # noqa: E402



def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--checkpoint_dir", default="results/checkpoints")
    ap.add_argument("--metrics_dir", default="results/metrics")
    ap.add_argument("--log_path", default="results/experiment_log.csv")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()
    seeds = cfg["training"]["phase1_seeds"]
    print(f"Phase 1 sweep: models={list(ARCHITECTURES)} seeds={seeds} -> {len(ARCHITECTURES) * len(seeds)} runs")
    print(f"Device requested: {args.device} (resolved: {get_device(args.device)})")

    prepared = None  # loaded once, reused across all 9 runs (split/normalization are seed-42 fixed)
    rows = []
    for model_name in ARCHITECTURES:
        for seed in seeds:
            print(f"\n--- {model_name} / seed {seed} ---")
            result, prepared = run_phase1_single(model_name, seed, cfg, data_dir, device_pref=args.device,
                                                 checkpoint_dir=args.checkpoint_dir, metrics_dir=args.metrics_dir,
                                                 prepared=prepared)
            rows.append({"run_name": result.run_name, "model": model_name, "seed": seed, "device": result.device,
                        "best_epoch": result.best_epoch, "epochs_ran": result.epochs_ran,
                        "stopped_early": result.stopped_early, "train_seconds": round(result.train_seconds, 1),
                        "best_val_loss": result.best_val_loss, "best_val_acc": result.best_val_acc,
                        "best_val_macro_f1": result.best_val_macro_f1, "checkpoint_path": result.checkpoint_path,
                        "curves_path": result.curves_path})

    log_path = Path(args.log_path)
    write_log(rows, log_path)
    print(f"\nWrote {log_path}")

    selection = select_architecture(rows)
    print("\n=== Phase 1 result: mean validation macro-F1 per architecture ===")
    for m, s in selection["summary"].items():
        marker = "  <-- selected" if m == selection["selected_model"] else ""
        print(f"  {m}: mean={s['mean_val_macro_f1']:.4f}  std={s['std_val_macro_f1']:.4f}  "
             f"range=[{s['min']:.4f}, {s['max']:.4f}]  (n={s['n_seeds']} seeds, best_epoch={s['best_epoch_per_seed']}){marker}")
    sel_path = Path(args.metrics_dir) / "architecture_selection.json"
    save_json(selection, sel_path)
    print(f"\nSelected architecture: {selection['selected_model']}")
    selected_stats = selection['summary'][selection['selected_model']]
    print(f"Phase 1 training durations for {selection['selected_model']}: {selected_stats['epochs_ran_per_seed']}")
    print(f"Mean training duration: {selected_stats['mean_epochs_ran']:.2f} epochs")
    print(f"Phase 2 fixed training budget: {selection['phase2_epochs']} epochs")
    print("Rule: round(mean actual Phase-1 epochs_ran); best-validation epochs are NOT used for this budget.")
    print(f"Wrote {sel_path}")
    print("\nPhase 1 complete. The official UCI test set was not loaded or used in this script.")
    return 0


if __name__ == "__main__":
    sys.exit(main())
