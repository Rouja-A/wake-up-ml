#!/usr/bin/env python
"""Inspect the actual UCI HAR directory. Nothing about filenames or shapes is assumed silently.

    python scripts/inspect_data.py --data_dir "path/to/UCI HAR Dataset"

Exit code 0 = all checks passed, 1 = a check failed, 2 = directory layout unusable.
Numpy only (no torch). Handcrafted-feature files are listed but never opened.
"""
import argparse
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                      # noqa: E402
from src.utils import Checker, default_data_dir  # noqa: E402


def classify(rel: Path, channels) -> str:
    name = rel.name
    if D.is_feature_file(rel):
        return "NOT USED (handcrafted features / feature docs)"
    for split in D.SPLITS:
        if any(rel == Path(split) / "Inertial Signals" / f"{c}_{split}.txt" for c in channels):
            return "USED  raw signal (model input)"
    if rel.parent.name == "Inertial Signals":
        return "not used (other inertial signal)"
    if name in {"activity_labels.txt", "y_train.txt", "y_test.txt", "subject_train.txt", "subject_test.txt"}:
        return "USED  labels / subject ids"
    return "not used"


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--data_dir", default=default_data_dir())
    args = ap.parse_args()
    channels = D.CHANNELS
    chk = Checker()

    try:
        root = D.resolve_data_dir(args.data_dir)
    except D.DatasetLayoutError as e:
        print(f"ERROR: {e}")
        return 2
    print(f"Dataset root: {root.resolve()}\n\n== 1. Discovered files ==")
    files = D.list_dataset_files(root)
    for rel in files:
        print(f"  {str(rel):<58} {classify(rel, channels)}")
    print(f"  ({len(files)} files)")

    print("\n== 2. Required files ==")
    missing = [str(p.relative_to(root)) for p in D.required_files(root, channels) if not p.is_file()]
    chk.check(not missing, "all required files present", f"missing: {missing}" if missing else "")
    if missing:
        print("ERROR: cannot continue without the required files.")
        chk.summary()
        return 2

    print("\n== 3. Activity label mapping (activity_labels.txt) ==")
    names = D.load_activity_labels(root)
    for i, n in enumerate(names):
        print(f"  file id {i + 1}  ->  class index {i}  ->  {n}")
    chk.check(names == D.EXPECTED_ACTIVITY_NAMES, "label names match the documented six activities", str(names))

    loaded = {}
    for split in D.SPLITS:
        print(f"\n== 4. Split '{split}' ==")
        arrays = {}
        for ch in channels:
            p = D.signal_path(root, split, ch)
            a = D.read_matrix(p)
            arrays[ch] = a
            n_nan, n_inf = int(np.isnan(a).sum()), int(np.isinf(a).sum())
            print(f"  {p.relative_to(root)!s:<52} shape={a.shape} nan={n_nan} inf={n_inf} min={a.min():.3f} max={a.max():.3f}")
            chk.check(a.shape[1] == D.WINDOW_LEN and n_nan == 0 and n_inf == 0, f"{p.name}: 128 columns, finite")
        y = D.read_matrix(D.labels_path(root, split), np.int64, 1)
        s = D.read_matrix(D.subjects_path(root, split), np.int64, 1)
        rows = {a.shape[0] for a in arrays.values()}
        print(f"  {D.labels_path(root, split).relative_to(root)!s:<52} shape={y.shape}")
        print(f"  {D.subjects_path(root, split).relative_to(root)!s:<52} shape={s.shape}")
        chk.check(len(rows) == 1 and y.shape[0] == s.shape[0] == next(iter(rows)),
                  "signals / labels / subjects have identical N", f"signal rows={sorted(rows)}, y={y.shape[0]}, subjects={s.shape[0]}")

        try:
            raw = D.load_split(root, split, channels, len(names), verify=True)
        except D.DatasetLayoutError as e:
            chk.check(False, f"loader verification for '{split}'", str(e))
            continue
        loaded[split] = raw
        chk.check(raw.x.shape == (len(y), 6, D.WINDOW_LEN), "combined tensor shape (N, 6, 128)", str(raw.x.shape))
        chk.check(raw.channels == D.CHANNELS, "channel order", str(raw.channels))
        chk.check(not any(D.is_feature_file(Path(f)) for f in raw.source_files), "no feature file among loader sources")
        chk.check(np.isfinite(raw.x).all(), "combined tensor finite")
        chk.check(set(np.unique(raw.y)) == set(range(len(names))), "labels are exactly the six classes", str(sorted(set(raw.y.tolist()))))

        subj = np.unique(raw.subjects)
        print(f"  subjects ({len(subj)}): {subj.tolist()}")
        print("  label distribution:")
        counts = np.bincount(raw.y, minlength=len(names))
        for i, n in enumerate(names):
            print(f"    {i} {n:<20} {counts[i]:5d}  ({100 * counts[i] / len(raw):4.1f}%)")
        doc = D.DOCUMENTED_COUNTS[split]
        print(f"  documented: {doc['windows']} windows / {doc['subjects']} subjects; "
              f"found: {len(raw)} windows / {len(subj)} subjects"
              + ("" if (len(raw), len(subj)) == (doc['windows'], doc['subjects']) else "   <-- DIFFERS from documentation"))
        rep = D.overlap_report(raw.x, raw.subjects)
        print(f"  50%-overlap check: matched pairs per channel {rep['matched_pairs_per_channel']} of {rep['n_pairs']} consecutive pairs; "
              f"{100 * rep['fraction_same_subject_pairs_matched']:.1f}% of same-subject pairs match in all channels")
        chk.check(rep["channels_consistent"], "row alignment across channel files (identical overlap pattern in all channels)")
        if rep["fraction_same_subject_pairs_matched"] < 0.5:
            chk.warn("few consecutive windows overlap", "row order may not be chronological; check the files")

    if len(loaded) == 2:
        print("\n== 5. Cross-split checks ==")
        tr, te = set(np.unique(loaded["train"].subjects)), set(np.unique(loaded["test"].subjects))
        chk.check(tr.isdisjoint(te), "train and test subjects are disjoint", f"shared: {sorted(tr & te)}")
        print(f"  total subjects: {len(tr | te)}, total windows: {len(loaded['train']) + len(loaded['test'])}")
        print(f"  final model input tensor: train {loaded['train'].x.shape}, test {loaded['test'].x.shape}, dtype {loaded['train'].x.dtype}")
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
