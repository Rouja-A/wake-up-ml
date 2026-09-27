#!/usr/bin/env python
"""Phase 1 (architecture comparison): train ONE model with ONE seed on the Step-2 subject-held-out
split. Used directly for a single run, and imported by scripts/run_experiments.py for the full 3-seed x
A/B/C sweep, and by scripts/smoke_test.py for a short pre-flight check.

    python scripts/train.py --model B --seed 0 --config configs/baseline.yaml [--epochs 2] [--data_dir ...]
"""
import argparse
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import preprocessing as P                                          # noqa: E402
from src.dataset import make_loaders                                        # noqa: E402
from src.train import fit, get_device                                       # noqa: E402
from src.utils import default_data_dir, load_config                        # noqa: E402


def run_phase1_single(model_name: str, seed: int, cfg: dict, data_dir: str, device_pref: str = "auto",
                      checkpoint_dir: str = "results/checkpoints", metrics_dir: str = "results/metrics",
                      max_epochs=None, quiet: bool = False, prepared: "P.PreparedData | None" = None):
    """Runs one Phase-1 training run and returns (FitResult, PreparedData used). Pass `prepared` to reuse
    already-loaded/split/normalized data across a sweep (the split+normalization never change with seed).
    """
    split_cfg, prep_cfg = cfg["split"], cfg["preprocessing"]
    if prepared is None:
        prepared = P.prepare_data(data_dir, channels=tuple(cfg["data"]["channels"]),
                                  n_val_subjects=split_cfg["n_val_subjects"], seed=split_cfg["seed"])
    device = get_device(device_pref)
    loaders = make_loaders(prepared, batch_size=cfg["training"]["batch_size"], seed=seed)
    run_name = f"phase1_{model_name}_seed{seed}"
    result = fit(model_name, loaders["train"], loaders["val"], prepared.class_names, cfg["training"],
                seed=seed, run_name=run_name, checkpoint_dir=Path(checkpoint_dir), metrics_dir=Path(metrics_dir),
                device=device, max_epochs=max_epochs, normalization_stats=prepared.stats.to_dict(),
                channels=prepared.channels, quiet=quiet)
    return result, prepared


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, required=True)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=None, help="override training.max_epochs (used by the smoke test)")
    ap.add_argument("--checkpoint_dir", default="results/checkpoints")
    ap.add_argument("--metrics_dir", default="results/metrics")
    args = ap.parse_args()

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()
    print(f"Device requested: {args.device}")
    result, _ = run_phase1_single(args.model, args.seed, cfg, data_dir, device_pref=args.device,
                                  checkpoint_dir=args.checkpoint_dir, metrics_dir=args.metrics_dir,
                                  max_epochs=args.epochs)
    print(f"\nDone. device={result.device} best_epoch={result.best_epoch} epochs_ran={result.epochs_ran} "
         f"stopped_early={result.stopped_early} best_val_macro_f1={result.best_val_macro_f1:.4f} "
         f"best_val_acc={result.best_val_acc:.4f} best_val_loss={result.best_val_loss:.4f} "
         f"time={result.train_seconds:.1f}s")
    print(f"checkpoint: {result.checkpoint_path}\ncurves:     {result.curves_path}")
    return 0


if __name__ == "__main__":
    sys.exit(main())
