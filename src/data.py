"""Raw UCI HAR loader (numpy only).

Only these files are ever opened:
  * the selected raw inertial signal files, one per channel and split
      <root>/<split>/Inertial Signals/<channel>_<split>.txt      (N rows x 128 columns)
  * <root>/<split>/y_<split>.txt, <root>/<split>/subject_<split>.txt, <root>/activity_labels.txt

The handcrafted-feature files (X_train.txt, X_test.txt, features*.txt) are NEVER opened. This is enforced
in two places: a filename guard inside `read_matrix`, and a re-check of the `source_files` list that every
`RawSplit` records about itself.

Channel order (= channel axis of the model input) is fixed and documented in CHANNELS.
"""
from __future__ import annotations

from dataclasses import dataclass
from pathlib import Path

import numpy as np

WINDOW_LEN = 128        # samples per window (2.56 s at 50 Hz), from the dataset documentation
SAMPLING_HZ = 50        # from the dataset documentation

# Primary model input. The order below IS the channel axis of the (N, 6, 128) tensor.
CHANNELS = (
    "total_acc_x", "total_acc_y", "total_acc_z",   # accelerometer incl. gravity [g]
    "body_gyro_x", "body_gyro_y", "body_gyro_z",   # gyroscope [rad/s]
)
# Every inertial signal the dataset ships (used to validate requested channel names).
KNOWN_SIGNALS = tuple(f"{s}_{a}" for s in ("total_acc", "body_acc", "body_gyro") for a in "xyz")

SPLITS = ("train", "test")
EXPECTED_ACTIVITY_NAMES = (
    "WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING",
)
# Documented sizes, used only to REPORT discrepancies in inspect_data.py (never to alter loading).
DOCUMENTED_COUNTS = {"train": {"windows": 7352, "subjects": 21}, "test": {"windows": 2947, "subjects": 9}}

FORBIDDEN_FILENAMES = frozenset({"X_train.txt", "X_test.txt", "features.txt", "features_info.txt"})


class DatasetLayoutError(RuntimeError):
    """The dataset directory does not look like UCI HAR, or its files are inconsistent."""


class FeatureFileAccessError(RuntimeError):
    """Something tried to read a handcrafted-feature file."""


# --------------------------------------------------------------------------------------- paths
def resolve_data_dir(path: str | Path) -> Path:
    """Accept either the 'UCI HAR Dataset' folder or its parent; return the former."""
    p = Path(path).expanduser()
    for cand in (p, p / "UCI HAR Dataset"):
        if (cand / "activity_labels.txt").is_file() and (cand / "train").is_dir() and (cand / "test").is_dir():
            return cand
    raise DatasetLayoutError(
        f"'{path}' is not a UCI HAR dataset directory (expected activity_labels.txt and train/ and test/ "
        f"inside it or inside '<path>/UCI HAR Dataset'). Set --data_dir or $UCI_HAR_DIR, or run "
        f"scripts/download_data.py."
    )


def signal_path(root: Path, split: str, channel: str) -> Path:
    return Path(root) / split / "Inertial Signals" / f"{channel}_{split}.txt"


def labels_path(root: Path, split: str) -> Path:
    return Path(root) / split / f"y_{split}.txt"


def subjects_path(root: Path, split: str) -> Path:
    return Path(root) / split / f"subject_{split}.txt"


def activity_labels_path(root: Path) -> Path:
    return Path(root) / "activity_labels.txt"


def required_files(root: Path, channels=CHANNELS) -> list[Path]:
    files = [activity_labels_path(root)]
    for split in SPLITS:
        files += [labels_path(root, split), subjects_path(root, split)]
        files += [signal_path(root, split, ch) for ch in channels]
    return files


def required_files_for_split(root: Path, split: str, channels=CHANNELS) -> list[Path]:
    """Like `required_files`, but for exactly one split. Used so a single-split loader (load_train_only /
    load_test_only) can check its OWN files exist without any reference to the other split's files."""
    return ([activity_labels_path(root), labels_path(root, split), subjects_path(root, split)]
           + [signal_path(root, split, ch) for ch in channels])


def check_required_files(root: Path, channels=CHANNELS) -> None:
    missing = [str(p) for p in required_files(root, channels) if not p.is_file()]
    if missing:
        raise DatasetLayoutError("Missing required dataset files:\n  " + "\n  ".join(missing))


def check_required_files_for_split(root: Path, split: str, channels=CHANNELS) -> None:
    missing = [str(p) for p in required_files_for_split(root, split, channels) if not p.is_file()]
    if missing:
        raise DatasetLayoutError(f"Missing required '{split}' dataset files:\n  " + "\n  ".join(missing))


def list_dataset_files(root: Path) -> list[Path]:
    """All files under root, as sorted paths relative to root."""
    root = Path(root)
    return sorted(p.relative_to(root) for p in root.rglob("*") if p.is_file())


# --------------------------------------------------------------------------------------- reading
def is_feature_file(path: Path) -> bool:
    name = Path(path).name
    return name in FORBIDDEN_FILENAMES or name.startswith("X_") or name.startswith("features")


def assert_not_feature_file(path: Path) -> None:
    if is_feature_file(path):
        raise FeatureFileAccessError(
            f"Refusing to read '{path}': handcrafted-feature files must never be used by the model pipeline."
        )


def read_matrix(path: Path, dtype=np.float64, ndmin: int = 2) -> np.ndarray:
    """Read a whitespace-delimited numeric text file. The ONLY place data files are opened."""
    path = Path(path)
    assert_not_feature_file(path)
    return np.loadtxt(path, dtype=dtype, ndmin=ndmin)


def load_activity_labels(root: Path) -> tuple[str, ...]:
    """Parse activity_labels.txt ("<id> <NAME>" per line).

    Returns names ordered by class index, where class index = file id - 1. The ids must be exactly 1..K.
    """
    mapping: dict[int, str] = {}
    for line in activity_labels_path(root).read_text().splitlines():
        line = line.strip()
        if line:
            idx, name = line.split(maxsplit=1)
            mapping[int(idx)] = name.strip()
    ids = sorted(mapping)
    if ids != list(range(1, len(ids) + 1)):
        raise DatasetLayoutError(f"activity_labels.txt ids are not contiguous 1..K: {ids}")
    return tuple(mapping[i] for i in ids)


# --------------------------------------------------------------------------------------- container
@dataclass(frozen=True)
class RawSplit:
    """One official split (train or test), raw and un-normalised."""
    name: str
    x: np.ndarray                 # (N, C, 128) float32, channel order = `channels`
    y: np.ndarray                 # (N,) int64, class index in [0, K-1] (file label - 1)
    subjects: np.ndarray          # (N,) int64, subject id of each window
    channels: tuple[str, ...]
    source_files: tuple[str, ...]  # every file this object was built from (audited for feature files)

    def __len__(self) -> int:
        return len(self.y)


# --------------------------------------------------------------------------------------- verification
def overlap_masks(x: np.ndarray, atol: float = 1e-6) -> np.ndarray:
    """(N-1, C) bool: does the 2nd half of window i equal the 1st half of window i+1 (per channel)?

    UCI HAR windows overlap by 50 %. If rows of two channel files were not aligned, the pattern of
    matching pairs would differ between channels.
    """
    half = x.shape[2] // 2
    return np.all(np.isclose(x[:-1, :, half:], x[1:, :, :half], rtol=0.0, atol=atol), axis=2)


def overlap_report(x: np.ndarray, subjects: np.ndarray) -> dict:
    masks = overlap_masks(x)
    same_subject = subjects[:-1] == subjects[1:]
    per_channel = masks.sum(axis=0)
    return {
        "n_pairs": int(masks.shape[0]),
        "n_same_subject_pairs": int(same_subject.sum()),
        "matched_pairs_per_channel": [int(v) for v in per_channel],
        "channels_consistent": bool((masks == masks[:, :1]).all()),
        "fraction_same_subject_pairs_matched": float(masks[same_subject].all(axis=1).mean())
        if same_subject.any() else float("nan"),
    }


def verify_raw_split(s: RawSplit, n_classes: int) -> None:
    """Raise on any dimensional / consistency problem. Called by load_split."""
    if s.x.ndim != 3 or s.x.shape[1] != len(s.channels) or s.x.shape[2] != WINDOW_LEN:
        raise DatasetLayoutError(f"[{s.name}] bad tensor shape {s.x.shape}; expected (N, {len(s.channels)}, {WINDOW_LEN})")
    n = s.x.shape[0]
    if s.y.shape != (n,) or s.subjects.shape != (n,):
        raise DatasetLayoutError(f"[{s.name}] rows disagree: x={n}, y={s.y.shape}, subjects={s.subjects.shape}")
    if not np.isfinite(s.x).all():
        raise DatasetLayoutError(f"[{s.name}] signals contain NaN/Inf")
    if s.y.min() < 0 or s.y.max() >= n_classes:
        raise DatasetLayoutError(f"[{s.name}] labels outside [0, {n_classes - 1}]: min={s.y.min()}, max={s.y.max()}")
    for f in s.source_files:
        assert_not_feature_file(Path(f))
    rep = overlap_report(s.x, s.subjects)
    if not rep["channels_consistent"]:
        raise DatasetLayoutError(
            f"[{s.name}] window ordering is inconsistent across channel files (50% overlap between consecutive "
            f"windows holds for a different set of rows per channel): matched pairs per channel = "
            f"{dict(zip(s.channels, rep['matched_pairs_per_channel']))}"
        )


# --------------------------------------------------------------------------------------- loading
def load_split(root: Path, split: str, channels=CHANNELS, n_classes: int = 6, verify: bool = True) -> RawSplit:
    """Load one split into a (N, C, 128) float32 tensor + labels + subject ids."""
    if split not in SPLITS:
        raise ValueError(f"split must be one of {SPLITS}")
    channels = tuple(channels)
    unknown = [c for c in channels if c not in KNOWN_SIGNALS]
    if unknown:
        raise ValueError(f"Unknown channels {unknown}; known: {KNOWN_SIGNALS}")

    sources, per_channel = [], []
    for ch in channels:
        p = signal_path(root, split, ch)
        if not p.is_file():
            raise DatasetLayoutError(f"Missing signal file: {p}")
        a = read_matrix(p, np.float64, 2)
        if a.shape[1] != WINDOW_LEN:
            raise DatasetLayoutError(f"{p.name}: expected {WINDOW_LEN} columns, got {a.shape[1]}")
        per_channel.append(a)
        sources.append(str(p))

    n = per_channel[0].shape[0]
    for ch, a in zip(channels, per_channel):
        if a.shape != (n, WINDOW_LEN):
            raise DatasetLayoutError(f"{ch}_{split}.txt has shape {a.shape}, other channels have ({n}, {WINDOW_LEN})")

    x = np.stack(per_channel, axis=1).astype(np.float32)          # (N, C, 128)

    y_file = read_matrix(labels_path(root, split), np.int64, 1)
    subjects = read_matrix(subjects_path(root, split), np.int64, 1)
    sources += [str(labels_path(root, split)), str(subjects_path(root, split))]
    if not (len(y_file) == len(subjects) == n):
        raise DatasetLayoutError(f"[{split}] row counts differ: signals={n}, labels={len(y_file)}, subjects={len(subjects)}")

    s = RawSplit(name=split, x=x, y=y_file - 1, subjects=subjects, channels=channels, source_files=tuple(sources))
    if verify:
        verify_raw_split(s, n_classes)
    return s


def load_dataset(data_dir: str | Path, channels=CHANNELS, verify: bool = True):
    """Load train and test. Returns (train: RawSplit, test: RawSplit, class_names: tuple[str, ...])."""
    root = resolve_data_dir(data_dir)
    check_required_files(root, channels)
    class_names = load_activity_labels(root)
    train = load_split(root, "train", channels, len(class_names), verify)
    test = load_split(root, "test", channels, len(class_names), verify)
    shared = set(np.unique(train.subjects)) & set(np.unique(test.subjects))
    if shared:
        raise DatasetLayoutError(f"Subjects appear in both official splits: {sorted(shared)}")
    return train, test, class_names


def load_train_only(data_dir: str | Path, channels=CHANNELS, verify: bool = True):
    """Load ONLY the official train split. No test-split file (signals, labels, or subjects) is opened
    or even checked for existence -- only activity_labels.txt (shared label names, not data) plus the
    train split's own files. Used by scripts/train_final.py so Phase 2 cannot access the test split even
    accidentally; scripts/evaluate_final.py is the only script that calls load_test_only / load_dataset.
    Returns (train: RawSplit, class_names: tuple[str, ...]).
    """
    root = resolve_data_dir(data_dir)
    check_required_files_for_split(root, "train", channels)
    class_names = load_activity_labels(root)
    return load_split(root, "train", channels, len(class_names), verify), class_names


def load_test_only(data_dir: str | Path, channels=CHANNELS, verify: bool = True):
    """Load ONLY the official test split (mirror of load_train_only). Returns (test: RawSplit, class_names)."""
    root = resolve_data_dir(data_dir)
    check_required_files_for_split(root, "test", channels)
    class_names = load_activity_labels(root)
    return load_split(root, "test", channels, len(class_names), verify), class_names
