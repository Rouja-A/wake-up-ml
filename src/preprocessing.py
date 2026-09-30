"""Preprocessing (numpy only): subject-level validation split and train-only per-channel standardization.

Pipeline, in order:
  1. subject_split      -- choose validation subjects from the OFFICIAL TRAIN subjects only (seeded)
  2. fit_channel_stats  -- per-channel mean/std from the training windows only (all windows x all time steps)
  3. standardize        -- apply those fixed statistics to train, validation and test
The official test split is only ever transformed with the training statistics; it never influences
the split, the statistics, or any decision.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from .data import CHANNELS, RawSplit, load_dataset, load_train_only

STD_EPS = 1e-6   # channels with std below this are treated as degenerate (see fit_channel_stats)


# ----------------------------------------------------------------------------------- subject split
@dataclass(frozen=True, eq=False)
class SubjectSplit:
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]
    train_idx: np.ndarray        # window indices (into the official train split)
    val_idx: np.ndarray
    seed: int

    def to_dict(self) -> dict:
        return {"seed": self.seed, "train_subjects": list(self.train_subjects), "val_subjects": list(self.val_subjects),
                "n_train_windows": int(len(self.train_idx)), "n_val_windows": int(len(self.val_idx))}


def subject_split(subjects: np.ndarray, n_val_subjects: int, seed: int) -> SubjectSplit:
    """Deterministic subject-level split. `subjects` must be the subject ids of the OFFICIAL TRAIN windows.

    Whole subjects are held out, never individual windows: windows overlap by 50 %, so a window-level
    split would put near-duplicate samples on both sides. Uses numpy's legacy RandomState, whose
    `choice` stream is stable across numpy versions.
    """
    subjects = np.asarray(subjects)
    all_subj = np.unique(subjects)
    if not 0 < n_val_subjects < len(all_subj):
        raise ValueError(f"n_val_subjects must be in [1, {len(all_subj) - 1}], got {n_val_subjects}")
    rng = np.random.RandomState(seed)
    val_subj = np.sort(rng.choice(all_subj, size=n_val_subjects, replace=False))
    train_subj = np.setdiff1d(all_subj, val_subj)
    is_val = np.isin(subjects, val_subj)
    split = SubjectSplit(tuple(int(s) for s in train_subj), tuple(int(s) for s in val_subj),
                         np.flatnonzero(~is_val), np.flatnonzero(is_val), int(seed))
    assert set(split.train_subjects).isdisjoint(split.val_subjects)
    assert len(split.train_idx) + len(split.val_idx) == len(subjects)
    return split


# ----------------------------------------------------------------------------------- normalization
@dataclass(frozen=True)
class ChannelStats:
    """Per-channel standardization statistics. Saved with checkpoints so inference reproduces preprocessing."""
    channels: tuple[str, ...]
    mean: tuple[float, ...]
    std: tuple[float, ...]                   # std actually APPLIED (degenerate channels use 1.0)
    n_windows: int                           # number of training windows used
    n_samples_per_channel: int               # n_windows * window_length
    degenerate_channels: tuple[str, ...] = ()  # channels whose raw std < STD_EPS (left un-scaled)
    fitted_on: str = "training windows only"

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        """(mean, std) shaped (1, C, 1) float64 for broadcasting over (N, C, T)."""
        return (np.asarray(self.mean, dtype=np.float64)[None, :, None],
                np.asarray(self.std, dtype=np.float64)[None, :, None])

    def to_dict(self) -> dict:
        return {k: (list(v) if isinstance(v, tuple) else v) for k, v in self.__dict__.items()}

    @classmethod
    def from_dict(cls, d: dict) -> "ChannelStats":
        return cls(channels=tuple(d["channels"]), mean=tuple(d["mean"]), std=tuple(d["std"]),
                   n_windows=int(d["n_windows"]), n_samples_per_channel=int(d["n_samples_per_channel"]),
                   degenerate_channels=tuple(d.get("degenerate_channels", ())), fitted_on=d.get("fitted_on", ""))

    def save(self, path: str | Path) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(self.to_dict(), indent=2))

    @classmethod
    def load(cls, path: str | Path) -> "ChannelStats":
        return cls.from_dict(json.loads(Path(path).read_text()))


def fit_channel_stats(x_train: np.ndarray, channels: tuple[str, ...]) -> ChannelStats:
    """Mean/std per channel over all training windows and all time steps (float64, population std).

    Degenerate channels (raw std < STD_EPS, e.g. a dead sensor) are NOT divided by ~0: their applied
    std is set to 1.0 (they are only mean-centred) and they are listed in `degenerate_channels`.
    """
    if x_train.ndim != 3 or x_train.shape[1] != len(channels):
        raise ValueError(f"x_train must be (N, {len(channels)}, T), got {x_train.shape}")
    if not np.isfinite(x_train).all():
        raise ValueError("x_train contains NaN/Inf")
    mean = x_train.mean(axis=(0, 2), dtype=np.float64)
    std = x_train.std(axis=(0, 2), dtype=np.float64)
    bad = std < STD_EPS
    if bad.any():
        print(f"  [WARN] near-zero std in channels {[c for c, b in zip(channels, bad) if b]}; "
              f"applying std=1.0 to them (mean-centring only)")
    applied = np.where(bad, 1.0, std)
    return ChannelStats(tuple(channels), tuple(float(m) for m in mean), tuple(float(s) for s in applied),
                        int(x_train.shape[0]), int(x_train.shape[0] * x_train.shape[2]),
                        tuple(c for c, b in zip(channels, bad) if b))


def standardize(x: np.ndarray, stats: ChannelStats) -> np.ndarray:
    """(x - mean) / std with FIXED statistics; float64 arithmetic, float32 output."""
    if x.ndim != 3 or x.shape[1] != len(stats.channels):
        raise ValueError(f"x must be (N, {len(stats.channels)}, T), got {x.shape}")
    mean, std = stats.arrays()
    return ((x.astype(np.float64) - mean) / std).astype(np.float32)


# ----------------------------------------------------------------------------------- assembly
@dataclass(eq=False)
class PreparedData:
    x_train: np.ndarray; y_train: np.ndarray; subjects_train: np.ndarray
    x_val: np.ndarray;   y_val: np.ndarray;   subjects_val: np.ndarray
    x_test: np.ndarray;  y_test: np.ndarray;  subjects_test: np.ndarray   # standardized with TRAIN stats; do not use for selection
    stats: ChannelStats
    split: SubjectSplit
    class_names: tuple[str, ...]
    channels: tuple[str, ...]


def prepare_from_raw(train_raw: RawSplit, test_raw: RawSplit, class_names, n_val_subjects: int, seed: int) -> PreparedData:
    """Split (train subjects only) -> fit stats on training windows only -> standardize all three sets."""
    if train_raw.channels != test_raw.channels:
        raise ValueError("train/test channel order differs")
    split = subject_split(train_raw.subjects, n_val_subjects, seed)
    x_tr_raw = train_raw.x[split.train_idx]
    stats = fit_channel_stats(x_tr_raw, train_raw.channels)          # <- the ONLY place statistics are fitted
    assert stats.n_windows == len(split.train_idx)
    return PreparedData(
        x_train=standardize(x_tr_raw, stats), y_train=train_raw.y[split.train_idx], subjects_train=train_raw.subjects[split.train_idx],
        x_val=standardize(train_raw.x[split.val_idx], stats), y_val=train_raw.y[split.val_idx], subjects_val=train_raw.subjects[split.val_idx],
        x_test=standardize(test_raw.x, stats), y_test=test_raw.y, subjects_test=test_raw.subjects,
        stats=stats, split=split, class_names=tuple(class_names), channels=train_raw.channels)


def prepare_data(data_dir, channels=CHANNELS, n_val_subjects: int = 4, seed: int = 42) -> PreparedData:
    train, test, names = load_dataset(data_dir, channels)
    return prepare_from_raw(train, test, names, n_val_subjects, seed)


# ----------------------------------------------------------------------------------- Phase 2 (final model) data
@dataclass(eq=False)
class FinalPreparedData:
    """Step-4 Phase 2 data: ALL official training-split subjects (train+val from Phase 1 combined) as
    training data, and the official test subjects for the one-time final evaluation. No validation
    subjects are held out here -- Phase 1 already used validation to select the architecture; nothing is
    tuned against this data, so there is nothing left to hold a validation set out FOR.
    """
    x_train: np.ndarray; y_train: np.ndarray; subjects_train: np.ndarray
    x_test: np.ndarray;  y_test: np.ndarray;  subjects_test: np.ndarray
    stats: ChannelStats
    class_names: tuple[str, ...]
    channels: tuple[str, ...]


def prepare_full_train_for_final(train_raw: RawSplit, test_raw: RawSplit, class_names) -> FinalPreparedData:
    """Fit normalization on ALL official-train windows (train+val subjects combined), apply to test."""
    if train_raw.channels != test_raw.channels:
        raise ValueError("train/test channel order differs")
    stats = fit_channel_stats(train_raw.x, train_raw.channels)
    assert stats.n_windows == len(train_raw)
    return FinalPreparedData(
        x_train=standardize(train_raw.x, stats), y_train=train_raw.y, subjects_train=train_raw.subjects,
        x_test=standardize(test_raw.x, stats), y_test=test_raw.y, subjects_test=test_raw.subjects,
        stats=stats, class_names=tuple(class_names), channels=train_raw.channels)


def prepare_final_data(data_dir, channels=CHANNELS) -> FinalPreparedData:
    train, test, names = load_dataset(data_dir, channels)
    return prepare_full_train_for_final(train, test, names)


# ----------------------------------------------------------------------------------- Phase 2 (final model) data,
# TEST-SPLIT-FREE variant -- used by scripts/train_final.py
@dataclass(eq=False)
class FinalTrainOnlyData:
    """Step-4 Phase 2 TRAINING data with NO access whatsoever to the official test split: only
    `load_train_only` is called, which never opens (or checks for the existence of) any test-split
    signal/label/subject file. Normalization statistics are fit on these windows and saved with the
    checkpoint; scripts/evaluate_final.py loads the test split separately (via `load_test_only`) in
    Step 5 and applies these SAVED statistics rather than recomputing anything from training data.
    """
    x_train: np.ndarray; y_train: np.ndarray; subjects_train: np.ndarray
    stats: ChannelStats
    class_names: tuple[str, ...]
    channels: tuple[str, ...]


def prepare_train_only_for_final(train_raw: RawSplit, class_names) -> FinalTrainOnlyData:
    """Fit normalization on ALL official-train windows. Takes only a RawSplit for the train split --
    there is no test_raw parameter, so it is structurally impossible to pass test data in here."""
    stats = fit_channel_stats(train_raw.x, train_raw.channels)
    assert stats.n_windows == len(train_raw)
    return FinalTrainOnlyData(x_train=standardize(train_raw.x, stats), y_train=train_raw.y,
                              subjects_train=train_raw.subjects, stats=stats,
                              class_names=tuple(class_names), channels=train_raw.channels)


def prepare_final_train_only(data_dir, channels=CHANNELS) -> FinalTrainOnlyData:
    """Load ONLY the official train split (`load_train_only` never opens any test-split file) and
    standardize it. This is what scripts/train_final.py calls."""
    train_raw, class_names = load_train_only(data_dir, channels)
    return prepare_train_only_for_final(train_raw, class_names)


def class_counts(y: np.ndarray, n_classes: int) -> np.ndarray:
    return np.bincount(y, minlength=n_classes)


def describe(p: PreparedData) -> str:
    """Exact subject ids, window counts and class distributions."""
    k = len(p.class_names)
    ctr, cva, cte = (class_counts(y, k) for y in (p.y_train, p.y_val, p.y_test))
    lines = [f"Subject-level split (seed={p.split.seed})",
             f"  train subjects ({len(p.split.train_subjects):2d}): {list(p.split.train_subjects)}",
             f"  val   subjects ({len(p.split.val_subjects):2d}): {list(p.split.val_subjects)}",
             f"  test  subjects ({len(np.unique(p.subjects_test)):2d}): {sorted(int(s) for s in np.unique(p.subjects_test))}"
             f"   [official test split; reference only]",
             f"Windows: train={len(p.y_train)}  val={len(p.y_val)}  test={len(p.y_test)}",
             "Class distribution (count, % of split)",
             f"  {'class':<24}{'train':>14}{'val':>14}{'test(ref)':>14}"]
    for i, name in enumerate(p.class_names):
        cells = "".join(f"{c[i]:>7d} ({100 * c[i] / c.sum():4.1f}%)" for c in (ctr, cva, cte))
        lines.append(f"  {i} {name:<22}{cells}")
    return "\n".join(lines)
