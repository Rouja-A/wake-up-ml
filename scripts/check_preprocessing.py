#!/usr/bin/env python
"""Data sanity check to run BEFORE any training.

    python scripts/check_preprocessing.py --data_dir "path/to/UCI HAR Dataset" [--config configs/baseline.yaml]

Loads the six raw channels, builds the subject-level train/val split, fits normalization on the training
windows only, prints subjects / counts / class distributions, runs assertions, plots sample windows and
writes results/metrics/{split.json,normalization_stats.json} and results/figures/sample_windows.png.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                                              # noqa: E402
from src import preprocessing as P                                      # noqa: E402
from src.utils import Checker, default_data_dir, load_config, save_json, set_seed  # noqa: E402


def plot_sample_windows(raw: D.RawSplit, train_idx, class_names, path: Path, seed: int) -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    rng = np.random.RandomState(seed)
    t = np.arange(D.WINDOW_LEN) / D.SAMPLING_HZ
    groups = [("accelerometer [g]", [i for i, c in enumerate(raw.channels) if "acc" in c]),
              ("gyroscope [rad/s]", [i for i, c in enumerate(raw.channels) if "gyro" in c])]
    fig, axes = plt.subplots(len(class_names), 2, figsize=(11, 1.9 * len(class_names)), sharex=True)
    for k, name in enumerate(class_names):
        cand = train_idx[raw.y[train_idx] == k]
        i = int(rng.choice(cand))
        for col, (title, idxs) in enumerate(groups):
            ax = axes[k, col]
            for j in idxs:
                ax.plot(t, raw.x[i, j], lw=1, label=raw.channels[j])
            ax.set_title(f"{name} (subject {raw.subjects[i]}) - {title}", fontsize=8)
            ax.tick_params(labelsize=7)
            ax.legend(fontsize=6, loc="upper right", ncol=3)
    for ax in axes[-1]:
        ax.set_xlabel("time [s]")
    fig.suptitle("Raw (un-normalised) training windows, one per activity", fontsize=10)
    fig.tight_layout()
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path, dpi=130)
    plt.close(fig)


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=None)
    ap.add_argument("--config", default=None, help="optional yaml; CLI flags override it")
    ap.add_argument("--n_val_subjects", type=int, default=None)
    ap.add_argument("--seed", type=int, default=None)
    ap.add_argument("--out_dir", default="results")
    args = ap.parse_args()

    cfg = load_config(args.config) if args.config else {}
    data_dir = args.data_dir or cfg.get("data", {}).get("data_dir") or default_data_dir()
    channels = tuple(cfg.get("data", {}).get("channels", D.CHANNELS))
    n_val = args.n_val_subjects or cfg.get("split", {}).get("n_val_subjects", 4)
    seed = args.seed if args.seed is not None else cfg.get("split", {}).get("seed", 42)
    batch_size = cfg.get("preprocessing", {}).get("batch_size", 64)
    out = Path(args.out_dir)
    set_seed(seed)
    chk = Checker()

    print("== Loading raw signals (feature files are never read) ==")
    train_raw, test_raw, names = D.load_dataset(data_dir, channels)     # raises on any layout/consistency problem
    p = P.prepare_from_raw(train_raw, test_raw, names, n_val, seed)
    tr, va = p.split.train_idx, p.split.val_idx

    print(f"  channels (model input order): {list(channels)}")
    print(f"  official train raw: {train_raw.x.shape} {train_raw.x.dtype} | official test raw: {test_raw.x.shape}")
    print(f"  after split       : train {p.x_train.shape} | val {p.x_val.shape} | test {p.x_test.shape}\n")
    print(P.describe(p))

    print("\n== Example labels ==")
    print("  first 10 train labels:", [f"{int(v)}={names[v]}" for v in p.y_train[:10]])
    rs = np.random.RandomState(seed).choice(len(p.y_val), 8, replace=False)
    print("  8 random val labels  :", [f"{int(p.y_val[i])}={names[p.y_val[i]]}" for i in rs])

    print("\n== Checks ==")
    n_ch = len(channels)
    for nm, x in (("train", p.x_train), ("val", p.x_val), ("test", p.x_test)):
        chk.check(x.ndim == 3 and x.shape[1:] == (n_ch, D.WINDOW_LEN) and x.dtype == np.float32,
                  f"{nm}: shape (N, {n_ch}, {D.WINDOW_LEN}) float32", f"{x.shape} {x.dtype}")
        chk.check(bool(np.isfinite(x).all()), f"{nm}: all {n_ch} channels finite after normalization")
    for nm, x, y, s in (("train", p.x_train, p.y_train, p.subjects_train), ("val", p.x_val, p.y_val, p.subjects_val),
                        ("test", p.x_test, p.y_test, p.subjects_test)):
        chk.check(len(x) == len(y) == len(s), f"{nm}: windows == labels == subject ids", f"{len(x)}/{len(y)}/{len(s)}")
    chk.check(len(channels) == 6 and tuple(channels) == train_raw.channels, "six channels in the documented order")

    chk.check(names == D.EXPECTED_ACTIVITY_NAMES, "label mapping equals documented activity names", str(names))
    for nm, y in (("train", p.y_train), ("val", p.y_val), ("test", p.y_test)):
        chk.check(set(np.unique(y).tolist()) == set(range(len(names))), f"{nm}: labels are exactly the six classes")
    chk.check(not any(D.is_feature_file(Path(f)) for f in train_raw.source_files + test_raw.source_files),
              "no handcrafted-feature file among the files the pipeline read")

    chk.check(set(p.split.val_subjects) <= set(np.unique(train_raw.subjects).tolist()), "val subjects come from official train subjects")
    chk.check(set(p.split.val_subjects).isdisjoint(np.unique(test_raw.subjects).tolist()), "val subjects disjoint from test subjects")
    chk.check(set(np.unique(p.subjects_train)).isdisjoint(np.unique(p.subjects_val)), "no subject in both train and val")

    ok_ch = [i for i, c in enumerate(channels) if c not in p.stats.degenerate_channels]
    m = p.x_train.astype(np.float64).mean(axis=(0, 2))
    s = p.x_train.astype(np.float64).std(axis=(0, 2))
    print("  train per-channel mean after normalization:", np.round(m, 6).tolist())
    print("  train per-channel std  after normalization:", np.round(s, 6).tolist())
    chk.check(bool(np.all(np.abs(m[ok_ch]) < 1e-4) and np.all(np.abs(s[ok_ch] - 1) < 1e-3)),
              "train data has ~zero mean / unit variance per channel")

    # Validation / test must be exactly (raw - TRAIN mean) / TRAIN std.
    mean, std = p.stats.arrays()
    chk.check(np.allclose(p.x_val, ((train_raw.x[va] - mean) / std).astype(np.float32), atol=1e-5),
              "val = (raw - train_mean) / train_std")
    chk.check(np.allclose(p.x_test, ((test_raw.x - mean) / std).astype(np.float32), atol=1e-5),
              "test = (raw - train_mean) / train_std")
    refit = P.fit_channel_stats(train_raw.x[tr], channels)
    chk.check(refit == p.stats, "stats are reproducible from training windows alone")
    with_val = P.fit_channel_stats(train_raw.x, channels)
    chk.check(with_val != p.stats, "stats differ from stats fitted on train+val (i.e. val did not leak in)")
    vm, vs = p.x_val.mean(axis=(0, 2), dtype=np.float64), p.x_val.std(axis=(0, 2), dtype=np.float64)
    print("  val   per-channel mean after normalization:", np.round(vm, 3).tolist(), "(expected NOT exactly 0)")
    print("  val   per-channel std  after normalization:", np.round(vs, 3).tolist(), "(expected NOT exactly 1)")

    out_metrics = out / "metrics"
    p.stats.save(out_metrics / "normalization_stats.json")
    save_json({**p.split.to_dict(), "test_subjects": sorted(int(s) for s in np.unique(p.subjects_test)),
               "channels": list(channels)}, out_metrics / "split.json")
    chk.check(P.ChannelStats.load(out_metrics / "normalization_stats.json") == p.stats, "normalization stats survive a save/load round trip")
    print(f"  wrote {out_metrics / 'normalization_stats.json'} and {out_metrics / 'split.json'}")

    fig_path = out / "figures" / "sample_windows.png"
    plot_sample_windows(train_raw, tr, names, fig_path, seed)
    print(f"  wrote {fig_path}")

    print("\n== DataLoader check ==")
    try:
        import torch
        from src.dataset import make_loaders
    except ImportError:
        chk.warn("torch not installed - DataLoader check skipped", "install requirements.txt locally")
    else:
        loaders = make_loaders(p, batch_size=batch_size, seed=seed)
        xb, yb = next(iter(loaders["train"]))
        chk.check(tuple(xb.shape) == (min(batch_size, len(p.y_train)), n_ch, D.WINDOW_LEN) and xb.dtype == torch.float32
                  and yb.dtype == torch.int64 and 0 <= int(yb.min()) and int(yb.max()) <= 5,
                  "train batch: x (B, 6, 128) float32, y int64 in [0, 5]", f"x={tuple(xb.shape)} {xb.dtype}, y={tuple(yb.shape)} {yb.dtype}")
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
