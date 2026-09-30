"""Tests for the data pipeline. Run with plain Python (no pytest needed):

    python tests/test_data_pipeline.py

or with pytest (`python -m pytest tests -q`); every test is a plain function using plain asserts.
They run on a small SYNTHETIC dataset that mimics the UCI HAR layout (tests/synthetic.py), so they need
neither the real dataset nor network access. They verify the code, not the real data: use
scripts/inspect_data.py and scripts/check_preprocessing.py for that.
"""
import dataclasses
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src import data as D                                   # noqa: E402
from src import preprocessing as P                          # noqa: E402
from tests.synthetic import (TEST_SUBJECTS, TRAIN_SUBJECTS,  # noqa: E402
                             _write_matrix, make_synthetic_har)

_CACHE = {}


def _root() -> Path:
    """One shared synthetic dataset for read-only tests."""
    if "root" not in _CACHE:
        _CACHE["tmp"] = tempfile.mkdtemp()
        _CACHE["root"] = make_synthetic_har(Path(_CACHE["tmp"]))
    return _CACHE["root"]


def _fresh_root() -> Path:
    tmp = tempfile.mkdtemp()
    _CACHE.setdefault("cleanup", []).append(tmp)
    return make_synthetic_har(Path(tmp))


def _raw():
    if "raw" not in _CACHE:
        _CACHE["raw"] = D.load_dataset(_root())
    return _CACHE["raw"]


def _prep(seed=0, n_val=4):
    train, test, names = _raw()
    return P.prepare_from_raw(train, test, names, n_val, seed)


def expect_raises(exc, fn, match=None):
    try:
        fn()
    except exc as e:
        assert match is None or match in str(e), f"message {e!s} lacks '{match}'"
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


# ------------------------------------------------------------------------------- loader
def test_shapes_dtypes_labels():
    train, test, names = _raw()
    for s in (train, test):
        assert s.x.shape[1:] == (6, 128) and s.x.dtype == np.float32
        assert s.y.dtype == np.int64 and s.subjects.dtype == np.int64
        assert len(s.x) == len(s.y) == len(s.subjects)
        assert set(np.unique(s.y)) == set(range(6))
        assert np.isfinite(s.x).all()
    assert names == D.EXPECTED_ACTIVITY_NAMES
    assert set(np.unique(train.subjects)) == set(TRAIN_SUBJECTS)
    assert set(np.unique(test.subjects)) == set(TEST_SUBJECTS)


def test_channel_order_matches_files():
    train, _, _ = _raw()
    assert train.channels == D.CHANNELS
    expected = ["total_acc_x", "total_acc_y", "total_acc_z", "body_gyro_x", "body_gyro_y", "body_gyro_z"]
    assert list(D.CHANNELS) == expected
    for i, ch in enumerate(expected):
        from_file = D.read_matrix(D.signal_path(_root(), "train", ch)).astype(np.float32)
        assert np.array_equal(train.x[:, i, :], from_file), f"channel {i} ({ch}) is not the {ch} file"


def test_only_selected_files_are_opened():
    opened = []
    real = np.loadtxt

    def spy(fname, *a, **k):
        opened.append(Path(fname).name)
        return real(fname, *a, **k)

    np.loadtxt = spy
    try:
        D.load_dataset(_root())
    finally:
        np.loadtxt = real
    forbidden = {"X_train.txt", "X_test.txt", "features.txt", "features_info.txt"}
    assert not (set(opened) & forbidden), f"feature file opened: {set(opened) & forbidden}"
    allowed = {f"{c}_{s}.txt" for c in D.CHANNELS for s in D.SPLITS} | {f"{n}_{s}.txt" for n in ("y", "subject") for s in D.SPLITS}
    assert set(opened) == allowed, f"unexpected files opened: {set(opened) ^ allowed}"


def test_feature_files_are_refused():
    for split in D.SPLITS:
        expect_raises(D.FeatureFileAccessError, lambda s=split: D.read_matrix(_root() / s / f"X_{s}.txt"))
    expect_raises(D.FeatureFileAccessError, lambda: D.read_matrix(_root() / "features.txt"))


def test_missing_file_fails_loudly():
    root = _fresh_root()
    (root / "test" / "Inertial Signals" / "total_acc_y_test.txt").unlink()
    expect_raises(D.DatasetLayoutError, lambda: D.load_dataset(root), match="total_acc_y_test.txt")


def test_row_misalignment_between_channels_is_detected():
    root = _fresh_root()
    p = D.signal_path(root, "train", "body_gyro_z")
    a = D.read_matrix(p)
    _write_matrix(p, a[np.random.RandomState(0).permutation(len(a))])
    expect_raises(D.DatasetLayoutError, lambda: D.load_dataset(root), match="inconsistent")


def test_row_count_mismatch_is_detected():
    root = _fresh_root()
    p = D.labels_path(root, "train")
    lines = p.read_text().splitlines()
    p.write_text("\n".join(lines[:-1]) + "\n")
    expect_raises(D.DatasetLayoutError, lambda: D.load_dataset(root), match="row counts differ")


def test_nan_is_detected():
    root = _fresh_root()
    p = D.signal_path(root, "train", "total_acc_x")
    a = D.read_matrix(p)
    a[3, 5] = np.nan
    _write_matrix(p, a)
    expect_raises(D.DatasetLayoutError, lambda: D.load_dataset(root))


# ------------------------------------------------------------------------------- split
def test_subject_split_properties():
    train, test, _ = _raw()
    sp = P.subject_split(train.subjects, 4, seed=0)
    assert len(sp.val_subjects) == 4 and len(sp.train_subjects) == 17
    assert set(sp.train_subjects).isdisjoint(sp.val_subjects)
    assert set(sp.val_subjects) <= set(TRAIN_SUBJECTS)                 # official train subjects only
    assert set(sp.val_subjects).isdisjoint(np.unique(test.subjects))   # never a test subject
    assert set(np.unique(train.subjects[sp.train_idx])) == set(sp.train_subjects)
    assert set(np.unique(train.subjects[sp.val_idx])) == set(sp.val_subjects)
    assert set(sp.train_idx).isdisjoint(sp.val_idx)
    assert len(sp.train_idx) + len(sp.val_idx) == len(train.subjects)


def test_subject_split_deterministic_and_seeded():
    train, _, _ = _raw()
    a, b = P.subject_split(train.subjects, 4, 7), P.subject_split(train.subjects, 4, 7)
    assert a.val_subjects == b.val_subjects and np.array_equal(a.val_idx, b.val_idx)
    assert len({P.subject_split(train.subjects, 4, s).val_subjects for s in range(6)}) > 1


# ------------------------------------------------------------------------------- normalization
def test_stats_come_from_training_windows_only():
    train, _, _ = _raw()
    p = _prep()
    x_tr = train.x[p.split.train_idx].astype(np.float64)
    assert np.allclose(p.stats.mean, x_tr.mean(axis=(0, 2)), rtol=1e-9, atol=0)
    assert np.allclose(p.stats.std, x_tr.std(axis=(0, 2)), rtol=1e-9, atol=0)
    assert p.stats.n_windows == len(p.split.train_idx)
    assert p.stats != P.fit_channel_stats(train.x, train.channels)     # would be equal if val leaked in


def test_val_and_test_cannot_influence_statistics():
    train, test, names = _raw()
    a = _prep()
    x2 = train.x.copy()
    x2[a.split.val_idx] = x2[a.split.val_idx] * 50 + 100               # wreck validation windows
    b = P.prepare_from_raw(dataclasses.replace(train, x=x2), dataclasses.replace(test, x=test.x * 10 + 3), names, 4, 0)
    assert a.stats == b.stats
    assert np.array_equal(a.x_train, b.x_train)
    assert not np.allclose(a.x_val, b.x_val) and not np.allclose(a.x_test, b.x_test)   # they ARE transformed, with the same stats


def test_train_is_zero_mean_unit_var_and_others_use_train_stats():
    train, test, _ = _raw()
    p = _prep()
    x = p.x_train.astype(np.float64)
    assert np.allclose(x.mean(axis=(0, 2)), 0, atol=1e-5) and np.allclose(x.std(axis=(0, 2)), 1, atol=1e-4)
    mean, std = p.stats.arrays()
    assert np.allclose(p.x_val, (train.x[p.split.val_idx] - mean) / std, atol=1e-5)
    assert np.allclose(p.x_test, (test.x - mean) / std, atol=1e-5)
    assert p.x_train.dtype == p.x_val.dtype == p.x_test.dtype == np.float32


def test_degenerate_channel_is_handled_safely():
    train, _, _ = _raw()
    x = train.x.copy()
    x[:, 2, :] = 3.0                                                    # dead channel: std == 0
    stats = P.fit_channel_stats(x, train.channels)
    assert stats.degenerate_channels == (train.channels[2],) and stats.std[2] == 1.0
    z = P.standardize(x, stats)
    assert np.isfinite(z).all() and np.allclose(z[:, 2, :], 0.0)        # centred, not blown up


def test_fit_rejects_nonfinite_and_bad_shape():
    train, _, _ = _raw()
    x = train.x.copy()
    x[0, 0, 0] = np.inf
    expect_raises(ValueError, lambda: P.fit_channel_stats(x, train.channels))
    expect_raises(ValueError, lambda: P.fit_channel_stats(train.x[:, :5], train.channels))


def test_stats_json_roundtrip():
    p = _prep()
    path = Path(tempfile.mkdtemp()) / "s.json"
    p.stats.save(path)
    assert P.ChannelStats.load(path) == p.stats


# ------------------------------------------------------------------------------- runner
def main() -> int:
    tests = [(n, f) for n, f in sorted(globals().items()) if n.startswith("test_") and callable(f)]
    failed = 0
    for name, fn in tests:
        try:
            fn()
            print(f"PASS  {name}")
        except Exception:
            failed += 1
            print(f"FAIL  {name}")
            traceback.print_exc()
    for d in [_CACHE.get("tmp"), *_CACHE.get("cleanup", [])]:
        if d:
            shutil.rmtree(d, ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
