"""Synthetic dataset that mimics the UCI HAR *directory layout and text format*. NOT real data.

Used only to test the loader / preprocessing code without network access. Properties that matter for the tests:
  * same file names, whitespace-delimited scientific-notation text, 128 columns per window
  * windows overlap by 50 % inside a (subject, activity) segment, exactly like the real data
  * per-subject offsets, so statistics fitted on different subject sets genuinely differ
  * X_train.txt / X_test.txt / features.txt contain NON-NUMERIC garbage: any attempt to parse them fails loudly
"""
from __future__ import annotations

from pathlib import Path

import numpy as np

ACTIVITIES = ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING"]
TEST_SUBJECTS = [2, 4, 9, 10, 12, 13, 18, 20, 24]
TRAIN_SUBJECTS = [s for s in range(1, 31) if s not in TEST_SUBJECTS]
SIGNALS = [f"{s}_{a}" for s in ("total_acc", "body_acc", "body_gyro") for a in "xyz"]

# gravity direction (in g) per activity: static classes differ mainly by orientation, as in the real data
_GRAVITY = {0: (0.0, 1.0, 0.0), 1: (0.1, 0.95, 0.1), 2: (-0.1, 0.95, -0.1),
            3: (0.3, 0.9, 0.2), 4: (0.0, 1.0, 0.05), 5: (1.0, 0.0, 0.0)}
_DYN_AMP = {0: 0.5, 1: 0.6, 2: 0.7, 3: 0.02, 4: 0.02, 5: 0.02}


def _write_matrix(path: Path, a: np.ndarray) -> None:
    """Mimic the real files: leading space, two spaces between values, scientific notation."""
    with open(path, "w") as f:
        for row in a:
            f.write(" " + "  ".join(f"{v: .8e}" for v in row) + "\n")


def _segment(rng, act: int, subj_bias: np.ndarray, n_windows: int) -> dict[str, np.ndarray]:
    n = 64 * (n_windows + 1)
    t = np.arange(n) / 50.0
    freq = 1.5 + 0.2 * act
    dyn = _DYN_AMP[act] * np.sin(2 * np.pi * freq * t + rng.uniform(0, 6.28))
    out = {}
    for i, a in enumerate("xyz"):
        grav = _GRAVITY[act][i] + subj_bias[i]
        noise = 0.01 * rng.standard_normal(n)
        out[f"total_acc_{a}"] = grav + dyn * (0.5 + 0.5 * i) + noise
        out[f"body_acc_{a}"] = dyn * (0.5 + 0.5 * i) + noise
        out[f"body_gyro_{a}"] = subj_bias[3 + i] + 2.0 * _DYN_AMP[act] * np.cos(2 * np.pi * freq * t) + 0.02 * rng.standard_normal(n)
    return out


def _windows(sig: np.ndarray, n_windows: int) -> np.ndarray:
    return np.stack([sig[64 * i: 64 * i + 128] for i in range(n_windows)])


def make_synthetic_har(base: Path, seed: int = 0, windows_per_segment: int = 6) -> Path:
    """Create '<base>/UCI HAR Dataset/...' and return that inner directory."""
    rng = np.random.RandomState(seed)
    root = Path(base) / "UCI HAR Dataset"
    (root).mkdir(parents=True, exist_ok=True)
    (root / "activity_labels.txt").write_text("\n".join(f"{i + 1} {n}" for i, n in enumerate(ACTIVITIES)) + "\n")
    (root / "features.txt").write_text("this is not numeric\n")
    (root / "features_info.txt").write_text("this is not numeric\n")

    for split, subjects in (("train", TRAIN_SUBJECTS), ("test", TEST_SUBJECTS)):
        d = root / split
        (d / "Inertial Signals").mkdir(parents=True, exist_ok=True)
        rows = {s: [] for s in SIGNALS}
        ys, subs = [], []
        for subj in subjects:
            bias = 0.05 * rng.standard_normal(6)
            for act in range(6):
                seg = _segment(rng, act, bias, windows_per_segment)
                for s in SIGNALS:
                    rows[s].append(_windows(seg[s], windows_per_segment))
                ys += [act + 1] * windows_per_segment
                subs += [subj] * windows_per_segment
        for s in SIGNALS:
            _write_matrix(d / "Inertial Signals" / f"{s}_{split}.txt", np.concatenate(rows[s]))
        (d / f"y_{split}.txt").write_text("".join(f"{v}\n" for v in ys))
        (d / f"subject_{split}.txt").write_text("".join(f"{v}\n" for v in subs))
        (d / f"X_{split}.txt").write_text("HANDCRAFTED FEATURES - must never be parsed\n")
    return root
