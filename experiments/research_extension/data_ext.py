"""Data loading for the research extension. Two hard rules enforced by construction, not just by
convention:

  1. OFFICIAL TEST FIREWALL: every function here is built ONLY on top of `src.data.load_train_only`.
     `src.data.load_dataset` and `src.data.load_test_only` are never imported into this module -- see
     `official_test_firewall.py` for a runtime assertion of that fact, and
     tests/test_research_extension.py::test_official_test_firewall_module_never_imports_test_loaders for
     the regression test.
  2. TRAIN-ONLY NORMALIZATION: for every sensor-channel subset, mean/std statistics are refit from
     scratch on that subset's OWN training windows (never the six-channel statistics sliced down, and
     never anything from validation windows) -- see `prepare_channel_subset` below.

This module reuses `src.preprocessing.subject_split` unmodified, so the extension's train/validation
subjects are IDENTICAL to Phase 1's (same seed, same n_val_subjects, from the same config) -- required by
RQ3's "use exactly the same subject split" and RQ4's "same subject-held-out validation protocol".
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

from src.data import CHANNELS, KNOWN_SIGNALS, load_train_only
from src.preprocessing import ChannelStats, fit_channel_stats, standardize, subject_split

# --------------------------------------------------------------------------------------- sensor subsets
# Full six-channel input (identical to src.architectures / configs/baseline.yaml).
FULL_CHANNELS = CHANNELS   # ("total_acc_x","total_acc_y","total_acc_z","body_gyro_x","body_gyro_y","body_gyro_z")
ACC_ONLY_CHANNELS = tuple(c for c in FULL_CHANNELS if c.startswith("total_acc"))
GYRO_ONLY_CHANNELS = tuple(c for c in FULL_CHANNELS if c.startswith("body_gyro"))
# Optional 4th arm (RQ4): only added if it does not require touching the loader beyond a channel-name
# substitution, which it doesn't -- `total_acc_*` and `body_acc_*` are both already KNOWN_SIGNALS.
BODY_ACC_GYRO_CHANNELS = tuple(f"body_acc_{a}" for a in "xyz") + GYRO_ONLY_CHANNELS

SENSOR_SUBSETS = {
    "acc_gyro_full": FULL_CHANNELS,
    "acc_only": ACC_ONLY_CHANNELS,
    "gyro_only": GYRO_ONLY_CHANNELS,
    "body_acc_gyro": BODY_ACC_GYRO_CHANNELS,
}


def validate_channel_subset(channels: tuple[str, ...]) -> None:
    unknown = [c for c in channels if c not in KNOWN_SIGNALS]
    if unknown:
        raise ValueError(f"Unknown channel(s) {unknown}; known signals: {KNOWN_SIGNALS}")
    if len(set(channels)) != len(channels):
        raise ValueError(f"Duplicate channels in subset: {channels}")


# --------------------------------------------------------------------------------------- train-only prep
@dataclass(eq=False)
class ExtensionPreparedData:
    """Same shape as `src.preprocessing.PreparedData` but with NO `x_test`/`y_test`/`subjects_test`
    fields at all -- there is structurally nothing here that could be accidentally evaluated against the
    official test set, unlike `PreparedData` (which does carry a standardized-but-unused test copy)."""
    x_train: np.ndarray; y_train: np.ndarray; subjects_train: np.ndarray
    x_val: np.ndarray;   y_val: np.ndarray;   subjects_val: np.ndarray
    stats: ChannelStats
    channels: tuple[str, ...]
    class_names: tuple[str, ...]
    train_subjects: tuple[int, ...]
    val_subjects: tuple[int, ...]


def prepare_channel_subset(data_dir, channels: tuple[str, ...], n_val_subjects: int, seed: int) -> ExtensionPreparedData:
    """Load ONLY the official train split (`load_train_only`), for the given channel subset; split
    subjects with the SAME `subject_split` seed/logic Phase 1 uses; fit normalization statistics on this
    subset's OWN training windows only (never sliced from the six-channel stats, never using validation
    windows); standardize train and validation with those statistics.
    """
    validate_channel_subset(channels)
    train_raw, class_names = load_train_only(data_dir, channels)   # <- the ONLY loader call in this function
    split = subject_split(train_raw.subjects, n_val_subjects, seed)
    x_tr_raw = train_raw.x[split.train_idx]
    stats = fit_channel_stats(x_tr_raw, train_raw.channels)         # fit fresh, on THIS subset only
    return ExtensionPreparedData(
        x_train=standardize(x_tr_raw, stats), y_train=train_raw.y[split.train_idx],
        subjects_train=train_raw.subjects[split.train_idx],
        x_val=standardize(train_raw.x[split.val_idx], stats), y_val=train_raw.y[split.val_idx],
        subjects_val=train_raw.subjects[split.val_idx],
        stats=stats, channels=train_raw.channels, class_names=tuple(class_names),
        train_subjects=split.train_subjects, val_subjects=split.val_subjects,
    )


def prepare_full_six_channel(data_dir, n_val_subjects: int, seed: int) -> ExtensionPreparedData:
    """Convenience wrapper: the full six-channel input, used by the linear baseline, Model D, and the
    Pareto/complexity comparison (everything except the RQ4 sensor-ablation arms)."""
    return prepare_channel_subset(data_dir, FULL_CHANNELS, n_val_subjects, seed)
