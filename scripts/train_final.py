#!/usr/bin/env python
"""Phase 2, step 1: retrain the SELECTED architecture from scratch on official-train + official-validation
subjects combined (i.e. the entire official UCI HAR training split). No architecture or hyperparameter
selection happens after this point: the architecture and every training hyperparameter (optimizer, lr,
weight decay, batch size) are exactly as used in Phase 1.

Phase 2 has NO held-out validation set (Phase 1 already used validation to select the architecture, and
there is nothing left to tune here), so there is no legitimate in-training signal to early-stop or
checkpoint-select on. Instead:
  * the epoch budget is FIXED beforehand, as round(mean best_epoch across the selected architecture's 3
    Phase-1 seeds) -- decided by the Phase-1 result, not by anything observed during this run;
  * early stopping is disabled (the run always trains for exactly that many epochs);
  * the checkpoint saved is the model after the FINAL epoch, not a "best" epoch chosen from this run.
The scheduler still runs (ReduceLROnPlateau on training loss, since there's no validation metric to
watch), purely as a training aid -- it does not affect which epoch's weights get saved.

This script loads ONLY the official train split (`src.data.load_train_only` / `src.preprocessing.
prepare_final_train_only`): no test-split signal, label, or subject file is opened, checked, standardized,
or passed into any function here. The official test set is loaded independently, later, by
scripts/evaluate_final.py, which applies the normalization statistics saved by THIS script's checkpoint.

    python scripts/train_final.py --config configs/baseline.yaml [--model B] [--epochs 40] [--data_dir ...]

If --model is omitted, the architecture is read from results/metrics/architecture_selection.json (written
by scripts/run_experiments.py), so Phase 2 cannot silently diverge from what Phase 1 selected. Likewise
--epochs defaults to that file's recommended Phase-2 epoch budget unless overridden.
"""
import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as P                                     # noqa: E402
from src.dataset import make_final_train_loader                        # noqa: E402
from src.train import fit, get_device                                  # noqa: E402
from src.utils import default_data_dir, load_config, save_json         # noqa: E402


def resolve_selection(args, metrics_dir: str) -> tuple[str, int, str]:
    sel_path = Path(metrics_dir) / "architecture_selection.json"
    sel = json.loads(sel_path.read_text()) if sel_path.is_file() else None
    if args.model is None and sel is None:
        raise SystemExit(f"No --model given and {sel_path} does not exist. Run scripts/run_experiments.py "
                         f"(Phase 1) first, or pass --model and --epochs explicitly.")
    model_name = args.model or sel["selected_model"]
    how = "provided via --model" if args.model else f"read from {sel_path} (Phase 1 selection)"
    if args.model and sel and args.model != sel["selected_model"]:
        print(f"  NOTE: --model {args.model} differs from Phase 1's selected model ({sel['selected_model']}); "
             f"proceeding with --model as explicitly requested.")
    if args.epochs is not None:
        epochs, epochs_how = args.epochs, "provided via --epochs"
    elif sel is not None and (not args.model or args.model == sel["selected_model"]):
        epochs, epochs_how = sel["phase2_epochs"], f"read from {sel_path} (Phase 1 result)"
    else:
        raise SystemExit("--model was overridden and does not match the Phase 1 selection file; "
                         "--epochs must be given explicitly in this case.")
    return model_name, epochs, f"{how}; epochs {epochs_how}"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", default=None, choices=[None, "A", "B", "C"])
    ap.add_argument("--epochs", type=int, default=None)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--checkpoint_dir", default="results/checkpoints")
    ap.add_argument("--metrics_dir", default="results/metrics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()
    model_name, epochs, how = resolve_selection(args, args.metrics_dir)
    seed = cfg["training"]["phase2_final_seed"]
    print(f"Phase 2: final model = {model_name}  ({how})")
    print(f"Fixed epoch budget: {epochs}   final_seed: {seed}   checkpoint_mode: last-epoch (no in-training selection)")

    final_data = P.prepare_final_train_only(data_dir, channels=tuple(cfg["data"]["channels"]))
    print(f"  train windows (official train+validation subjects combined): {len(final_data.y_train)}  "
         f"subjects: {sorted(set(int(s) for s in final_data.subjects_train))}")
    print("  official test split: NOT loaded by this script (see scripts/evaluate_final.py, Step 5).")

    device = get_device(args.device)
    train_loader = make_final_train_loader(final_data, batch_size=cfg["training"]["batch_size"], seed=seed)

    phase2_cfg = dict(cfg["training"])
    phase2_cfg["max_epochs"] = epochs
    phase2_cfg["early_stopping"] = {"monitor": "val_macro_f1", "mode": "max", "patience": epochs + 1}  # never triggers
    if cfg["training"].get("scheduler"):
        phase2_cfg["scheduler"] = {**cfg["training"]["scheduler"], "monitor": "train_loss", "mode": "min"}

    run_name = f"phase2_final_{model_name}_seed{seed}"
    # train_loader is passed as both train and "eval" loader purely so the saved curves show a (training-
    # set, eval-mode) loss/accuracy trajectory for the report; that eval pass plays NO role in checkpoint
    # selection when checkpoint_mode="last" (see fit()'s docstring).
    result = fit(model_name, train_loader, train_loader, final_data.class_names, phase2_cfg, seed=seed,
                run_name=run_name, checkpoint_dir=Path(args.checkpoint_dir), metrics_dir=Path(args.metrics_dir),
                device=device, max_epochs=epochs, normalization_stats=final_data.stats.to_dict(),
                channels=final_data.channels, quiet=False, checkpoint_mode="last")

    print(f"\nDone. epochs_ran={result.epochs_ran} (fixed budget, no early stop) time={result.train_seconds:.1f}s")
    print(f"checkpoint (final epoch): {result.checkpoint_path}")
    save_json({"model_name": model_name, "how_selected": how, "seed": seed, "epochs": epochs, "run_name": run_name,
              "checkpoint_path": result.checkpoint_path, "curves_path": result.curves_path,
              "n_train_windows": len(final_data.y_train),
              "train_subjects": sorted(set(int(s) for s in final_data.subjects_train))},
             Path(args.metrics_dir) / "final_model_info.json")
    print(f"Wrote {Path(args.metrics_dir) / 'final_model_info.json'}")
    print("\nPhase 2 training complete. The official test split was never loaded in this script.")
    print("Next: python scripts/evaluate_final.py   (loads the test split and runs the ONE final evaluation)")
    return 0


if __name__ == "__main__":
    sys.exit(main())
