#!/usr/bin/env python
"""Pre-flight check: trains Model A, seed 0, for a couple of epochs and verifies the training pipeline
actually works end to end, BEFORE launching the full 9-run Phase-1 sweep. Writes to
results/smoke_test/... (never touches the real results/experiment_log.csv or results/checkpoints/).

    python scripts/smoke_test.py --config configs/baseline.yaml [--epochs 2] [--data_dir ...]

Checks: which device is actually used; a real batch moves through model+loss+backward without error;
loss is finite (and, ideally, decreases); validation metrics are in valid ranges and the confusion
matrix sums to the validation set size; a saved checkpoint reloads into a fresh model with identical
weights; the curves JSON and a one-row experiment-log CSV are written and well-formed.
"""
import argparse
import csv
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import numpy as np                                                    # noqa: E402
import torch                                                          # noqa: E402

from src import preprocessing as P                                     # noqa: E402
from src.dataset import make_loaders                                   # noqa: E402
from src.evaluate import evaluate, load_model_from_checkpoint          # noqa: E402
from src.train import fit, get_device                                  # noqa: E402
from src.utils import Checker, default_data_dir, load_config, save_json  # noqa: E402
from src.experiment_log import LOG_FIELDS, write_log                    # noqa: E402


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--config", default="configs/baseline.yaml")
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--device", default="auto", choices=["auto", "cpu", "cuda"])
    ap.add_argument("--epochs", type=int, default=2)
    ap.add_argument("--model", default="A", choices=["A", "B", "C"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out_dir", default="results/smoke_test")
    args = ap.parse_args()
    chk = Checker()
    out = Path(args.out_dir)
    ckpt_dir, metrics_dir = out / "checkpoints", out / "metrics"

    cfg = load_config(args.config)
    data_dir = args.data_dir or cfg["data"]["data_dir"] or default_data_dir()
    split_cfg = cfg["split"]

    print("== 1. Device ==")
    device = get_device(args.device)
    print(f"  requested: {args.device}  resolved: {device}  cuda available: {torch.cuda.is_available()}")
    if torch.cuda.is_available():
        chk.check(str(device) == "cuda", "CUDA is available and was selected", str(device))
        print(f"  GPU: {torch.cuda.get_device_name(0)}")
    else:
        chk.warn("CUDA not available in this environment", "falling back to CPU (expected on a CPU-only machine)")

    print("\n== 2. Data ==")
    prepared = P.prepare_data(data_dir, channels=tuple(cfg["data"]["channels"]),
                              n_val_subjects=split_cfg["n_val_subjects"], seed=split_cfg["seed"])
    loaders = make_loaders(prepared, batch_size=cfg["training"]["batch_size"], seed=args.seed)
    xb, yb = next(iter(loaders["train"]))
    print(f"  train batch: x {tuple(xb.shape)} {xb.dtype}, y {tuple(yb.shape)} {yb.dtype}")
    chk.check(tuple(xb.shape)[1:] == (6, 128) and xb.dtype == torch.float32, "batch shape/dtype as expected")

    print(f"\n== 3. Short training run (Model {args.model}, seed {args.seed}, {args.epochs} epoch(s)) ==")
    result = fit(args.model, loaders["train"], loaders["val"], prepared.class_names, cfg["training"], seed=args.seed,
                run_name=f"smoke_{args.model}_seed{args.seed}", checkpoint_dir=ckpt_dir, metrics_dir=metrics_dir, device=device,
                max_epochs=args.epochs, normalization_stats=prepared.stats.to_dict(), channels=prepared.channels,
                quiet=False)
    chk.check(result.epochs_ran >= 1, "training ran at least one epoch", str(result.epochs_ran))
    chk.check(np.isfinite(result.best_val_loss), "final val loss is finite", str(result.best_val_loss))

    print("\n== 4. Loss/metric sanity ==")
    import json
    curves = json.loads(Path(result.curves_path).read_text())
    tl = curves["history"]["train_loss"]
    print(f"  train_loss per epoch: {[round(v, 4) for v in tl]}")
    chk.check(all(np.isfinite(v) and v >= 0 for v in tl), "all per-epoch training losses are finite and non-negative")
    if len(tl) >= 2:
        if tl[-1] < tl[0]:
            print(f"  [PASS] loss decreased over the run ({tl[0]:.4f} -> {tl[-1]:.4f})")
        else:
            chk.warn("loss did not decrease over this short run", f"{tl[0]:.4f} -> {tl[-1]:.4f}; "
                    "not necessarily a problem over only 1-2 epochs, but worth a look if it persists")

    print("\n== 5. Validation metrics on the actual model ==")
    m, analysis = evaluate(result.checkpoint_path, loaders["val"], device, class_names=prepared.class_names)
    chk.check(0.0 <= m.accuracy <= 1.0, "val accuracy in [0, 1]", str(m.accuracy))
    chk.check(0.0 <= m.macro_f1 <= 1.0, "val macro-F1 in [0, 1]", str(m.macro_f1))
    chk.check(int(m.confusion.sum()) == m.n_samples == len(prepared.y_val), "confusion matrix sums to the validation set size",
             f"{int(m.confusion.sum())} vs {len(prepared.y_val)}")
    chk.check(all(0.0 <= c["f1"] <= 1.0 for c in m.per_class), "all per-class F1 scores in [0, 1]")
    print(f"  val accuracy={m.accuracy:.4f}  macro_f1={m.macro_f1:.4f}  n={m.n_samples}")

    print("\n== 6. Checkpoint save/load round trip ==")
    model1, ckpt1 = load_model_from_checkpoint(result.checkpoint_path, device)
    model2, ckpt2 = load_model_from_checkpoint(result.checkpoint_path, device)  # independent reload
    same = all(torch.equal(p1, p2) for p1, p2 in zip(model1.state_dict().values(), model2.state_dict().values()))
    chk.check(same, "two independent loads of the same checkpoint produce identical weights")
    with torch.no_grad():
        x = torch.randn(3, 6, 128, device=device)
        out1, out2 = model1(x), model2(x)
    chk.check(bool(torch.equal(out1, out2)), "reloaded models produce identical forward-pass output on the same input")
    chk.check(bool(torch.isfinite(out1).all()), "forward pass on a reloaded checkpoint has no NaN/Inf")
    chk.check(ckpt1["model_name"] == args.model and ckpt1["seed"] == args.seed, "checkpoint metadata (model_name, seed) round-trips correctly")

    print("\n== 7. Curves / log files ==")
    chk.check(Path(result.curves_path).is_file(), "curves JSON was written", result.curves_path)
    for key in ("train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"):
        chk.check(key in curves["history"] and len(curves["history"][key]) == result.epochs_ran,
                 f"curves JSON has '{key}' with one entry per epoch ran")
    row = {"run_name": result.run_name, "model": args.model, "seed": args.seed, "device": result.device,
          "best_epoch": result.best_epoch, "epochs_ran": result.epochs_ran, "stopped_early": result.stopped_early,
          "train_seconds": round(result.train_seconds, 1), "best_val_loss": result.best_val_loss,
          "best_val_acc": result.best_val_acc, "best_val_macro_f1": result.best_val_macro_f1,
          "checkpoint_path": result.checkpoint_path, "curves_path": result.curves_path}
    log_path = out / "experiment_log.csv"
    write_log([row], log_path)
    with open(log_path) as f:
        read_back = list(csv.DictReader(f))
    chk.check(len(read_back) == 1 and read_back[0]["model"] == args.model, "experiment-log CSV writes and reads back correctly")

    print(f"\nSmoke-test artifacts are under {out}/ (separate from the real results/ sweep outputs).")
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
