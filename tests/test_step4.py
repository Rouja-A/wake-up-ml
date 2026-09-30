"""Tests for Step 4 (training protocol). Run with plain Python:

    python tests/test_step4.py

Two tiers, like tests/test_model.py:
  * always run: src/experiment_log.py (CSV I/O + architecture/epoch-budget selection) has no torch
    dependency and is fully tested here regardless of environment.
  * torch-gated: EarlyStopper, monitor-value separation, a real (tiny, synthetic-data) end-to-end
    training run, checkpoint round-trip, and Phase-1/Phase-2 data-prep wiring. These need torch and
    print "(skipped: torch not installed)" when it's absent, exactly like tests/test_model.py.
"""
import shutil
import sys
import tempfile
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.experiment_log import LOG_FIELDS, read_log, select_architecture, write_log  # noqa: E402
from src import data as D                                                             # noqa: E402
from src import preprocessing as Pp                                                   # noqa: E402
from tests.synthetic import make_synthetic_har                                        # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from src.dataset import make_final_train_loader, make_loaders                    # noqa: E402
    from src.evaluate import evaluate, load_model_from_checkpoint                     # noqa: E402
    from src.train import EarlyStopper, EpochResult, _monitor_value, fit             # noqa: E402

_CACHE = {}


def _root() -> Path:
    if "root" not in _CACHE:
        tmp = tempfile.mkdtemp()
        _CACHE["tmp"] = tmp
        _CACHE["root"] = make_synthetic_har(Path(tmp), seed=2, windows_per_segment=10)
    return _CACHE["root"]


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


def _fake_row(model, seed, f1, epoch):
    return {"run_name": f"{model}_s{seed}", "model": model, "seed": seed, "device": "cpu", "best_epoch": epoch,
           "epochs_ran": epoch + 5, "stopped_early": True, "train_seconds": 1.0, "best_val_loss": 0.5,
           "best_val_acc": 0.8, "best_val_macro_f1": f1, "checkpoint_path": "x", "curves_path": "y"}


# ============================================================================= always run (no torch)
def test_experiment_log_csv_roundtrip():
    rows = [_fake_row("A", s, 0.8 + 0.01 * s, 10 + s) for s in range(3)]
    path = Path(tempfile.mkdtemp()) / "log.csv"
    write_log(rows, path)
    back = read_log(path)
    assert len(back) == 3 and set(back[0]) == set(LOG_FIELDS)
    assert back[0]["model"] == "A"


def test_select_architecture_picks_highest_mean_f1():
    rows = ([_fake_row("A", s, f1, ep) for s, f1, ep in zip([0, 1, 2], [0.80, 0.82, 0.79], [12, 10, 14])] +
           [_fake_row("B", s, f1, ep) for s, f1, ep in zip([0, 1, 2], [0.90, 0.88, 0.91], [25, 30, 22])] +
           [_fake_row("C", s, f1, ep) for s, f1, ep in zip([0, 1, 2], [0.885, 0.87, 0.895], [18, 20, 19])])
    sel = select_architecture(rows)
    assert sel["selected_model"] == "B"
    assert abs(sel["summary"]["B"]["mean_val_macro_f1"] - (0.90 + 0.88 + 0.91) / 3) < 1e-9
    assert sel["phase2_epochs"] == round((25 + 30 + 22) / 3)


def test_select_architecture_works_on_string_valued_csv_rows():
    rows = [_fake_row("A", s, 0.8, 10) for s in range(3)]
    path = Path(tempfile.mkdtemp()) / "log.csv"
    write_log(rows, path)
    sel = select_architecture(read_log(path))          # values come back as str from csv.DictReader
    assert sel["selected_model"] == "A"


def test_select_architecture_std_zero_for_single_seed():
    sel = select_architecture([_fake_row("A", 0, 0.8, 10)])
    assert sel["summary"]["A"]["std_val_macro_f1"] == 0.0
    assert sel["summary"]["A"]["n_seeds"] == 1


def test_select_architecture_rejects_empty():
    expect_raises(ValueError, lambda: select_architecture([]))


def test_phase2_data_prep_uses_all_official_train_subjects_and_no_leakage():
    train, test, names = D.load_dataset(_root())
    final = Pp.prepare_full_train_for_final(train, test, names)
    assert final.x_train.shape[0] == len(train)   # ALL official-train windows, no held-out subjects
    assert set(np.unique(final.subjects_train)) == set(np.unique(train.subjects))
    assert set(np.unique(final.subjects_test)) == set(np.unique(test.subjects))
    assert set(np.unique(final.subjects_train)).isdisjoint(np.unique(final.subjects_test))
    # stats must come from ALL train windows (not a subset) -- differ from a Phase-1-style subset fit
    phase1 = Pp.prepare_from_raw(train, test, names, n_val_subjects=4, seed=42)
    assert final.stats != phase1.stats
    assert final.stats.n_windows == len(train) > phase1.stats.n_windows
    mean, std = final.stats.arrays()
    assert np.allclose(final.x_test, ((test.x - mean) / std).astype(np.float32), atol=1e-5)


# ============================================================================= torch-gated
def test_phase1_scripts_never_reference_test_data():
    """Source-level guard: Phase 1 (scripts/train.py, scripts/run_experiments.py) must never touch the
    official test split. This is deliberately a static check, not a mock, so it also catches someone
    later adding a `.x_test`/`make_test_loader` reference without exercising it in this run."""
    for rel in ("train.py", "run_experiments.py"):
        src = (Path(__file__).resolve().parents[1] / "scripts" / rel).read_text()
        for forbidden in ("make_test_loader", "make_final_test_loader", "x_test", "y_test", "subjects_test"):
            assert forbidden not in src, f"scripts/{rel} references '{forbidden}' -- Phase 1 must not touch test data"


def test_phase1_scripts_never_reference_test_data():
    """Source-level guard: Phase 1 (scripts/train.py, scripts/run_experiments.py) must never touch the
    official test split. This is deliberately a static check, not a mock, so it also catches someone
    later adding a `.x_test`/`make_test_loader` reference without exercising it in this run."""
    for rel in ("train.py", "run_experiments.py"):
        src = (Path(__file__).resolve().parents[1] / "scripts" / rel).read_text()
        for forbidden in ("make_test_loader", "make_final_test_loader", "x_test", "y_test", "subjects_test"):
            assert forbidden not in src, f"scripts/{rel} references '{forbidden}' -- Phase 1 must not touch test data"


def test_train_final_source_never_references_test_data():
    """Source-level guard specifically for scripts/train_final.py (Phase 2 training): it must not import
    or call anything that can load, standardize, or otherwise touch the official test split. Stronger
    than the Phase-1 guard above because Phase 2 used to (legitimately, but unnecessarily) standardize
    test data as a side effect of `prepare_final_data`/`prepare_full_train_for_final` -- this asserts that
    path is gone, not just that a `.x_test` attribute isn't read."""
    src = (Path(__file__).resolve().parents[1] / "scripts" / "train_final.py").read_text()
    for forbidden in ("prepare_final_data", "prepare_full_train_for_final", "load_dataset", "load_test_only",
                      "make_final_test_loader", "make_test_loader_from_stats", "x_test", "y_test", "subjects_test",
                      "test_raw"):
        assert forbidden not in src, f"scripts/train_final.py references '{forbidden}' -- Phase 2 training must not touch test data"
    assert "prepare_final_train_only" in src, "scripts/train_final.py must use the test-split-free data path"


def test_load_train_only_never_opens_a_test_split_file():
    """Real (not mocked) I/O-level guard: patches np.loadtxt to record every file actually opened by
    load_train_only, and asserts none of them is a test-split file. Runs against the synthetic dataset,
    which has real (if fake) test-split files on disk that WOULD be opened by a bug that reintroduces
    test-split access."""
    root = _root()
    opened = []
    real_loadtxt = np.loadtxt

    def spy(fname, *a, **k):
        opened.append(Path(fname).name)
        return real_loadtxt(fname, *a, **k)

    np.loadtxt = spy
    try:
        train_raw, names = D.load_train_only(root)
    finally:
        np.loadtxt = real_loadtxt

    test_filenames = {f"{c}_test.txt" for c in D.CHANNELS} | {"y_test.txt", "subject_test.txt", "X_test.txt"}
    touched = set(opened) & test_filenames
    assert not touched, f"load_train_only opened test-split file(s): {touched}"
    assert set(np.unique(train_raw.subjects)) <= set(np.unique(train_raw.subjects))  # sanity: loaded something
    assert len(train_raw) > 0


def test_prepare_final_train_only_has_no_test_attribute():
    """FinalTrainOnlyData structurally cannot carry test data: there is no x_test/y_test/subjects_test
    field for a bug to accidentally populate or read."""
    final = Pp.prepare_final_train_only(_root())
    for forbidden in ("x_test", "y_test", "subjects_test"):
        assert not hasattr(final, forbidden), f"FinalTrainOnlyData unexpectedly has '{forbidden}'"
    train_raw, _ = D.load_train_only(_root())
    assert final.x_train.shape[0] == len(train_raw)
    assert set(np.unique(final.subjects_train)) == set(np.unique(train_raw.subjects))


def test_saved_stats_applied_to_independently_loaded_test_data_match_manual_standardization():
    """End-to-end check of the Step-5 contract: train_final.py's saved stats, applied by evaluate_final.py
    to test data loaded via a completely separate call (load_test_only), must equal manually applying
    (x - mean) / std with those same saved statistics."""
    final = Pp.prepare_final_train_only(_root())
    test_raw, _ = D.load_test_only(_root())
    x_std = Pp.standardize(test_raw.x, final.stats)
    mean, std = final.stats.arrays()
    assert np.allclose(x_std, ((test_raw.x - mean) / std).astype(np.float32), atol=1e-5)


def test_group_confusion_breakdown_matches_injected_pattern():
    from src.metrics import compute_metrics, group_confusion_breakdown
    names = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
    groups = {"static": ["SITTING", "STANDING", "LAYING"], "dynamic": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}
    rng = np.random.RandomState(0)
    y_true = rng.randint(0, 6, 300)
    y_pred = y_true.copy()
    mask_static = (y_true == 3) & (rng.rand(300) < 0.4)     # SITTING -> STANDING (within static)
    y_pred[mask_static] = 4
    mask_dynamic = (y_true == 0) & (rng.rand(300) < 0.3)    # WALKING -> WALKING_UPSTAIRS (within dynamic)
    y_pred[mask_dynamic] = 1
    mask_cross = (y_true == 5) & (rng.rand(300) < 0.2)      # LAYING -> WALKING (across groups)
    y_pred[mask_cross] = 0
    m = compute_metrics(y_true, y_pred, names)
    gb = group_confusion_breakdown(m, groups)
    assert gb["total_errors"] == int((y_true != y_pred).sum())
    assert gb["within_group_errors"]["static"] == int(mask_static.sum())
    assert gb["within_group_errors"]["dynamic"] == int(mask_dynamic.sum())
    assert gb["across_group_errors"] == int(mask_cross.sum())
    assert abs(sum(gb["within_group_fraction"].values()) + gb["across_group_fraction"] - 1.0) < 1e-9


def test_group_confusion_breakdown_rejects_incomplete_or_overlapping_groups():
    from src.metrics import compute_metrics, group_confusion_breakdown
    names = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
    m = compute_metrics(np.arange(6) % 6, np.arange(6) % 6, names)
    expect_raises(ValueError, lambda: group_confusion_breakdown(m, {"static": ["SITTING", "STANDING"]}))  # incomplete
    expect_raises(ValueError, lambda: group_confusion_breakdown(m, {"a": ["SITTING"], "b": names}))         # overlap


def test_group_confusion_breakdown_zero_errors_no_division_by_zero():
    from src.metrics import compute_metrics, group_confusion_breakdown
    names = ("WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS", "SITTING", "STANDING", "LAYING")
    groups = {"static": ["SITTING", "STANDING", "LAYING"], "dynamic": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}
    y = np.tile(np.arange(6), 5)
    m = compute_metrics(y, y, names)
    gb = group_confusion_breakdown(m, groups)
    assert gb["total_errors"] == 0 and gb["across_group_fraction"] == 0.0
    assert all(v == 0.0 for v in gb["within_group_fraction"].values())


def _evaluate_final_source() -> str:
    return (Path(__file__).resolve().parents[1] / "scripts" / "evaluate_final.py").read_text()


def test_evaluate_final_guard_is_checked_before_any_computation_and_marker_written_last():
    """The refusal guard must run before any model/data loading, and the JSON that the guard checks for
    (`final_test_evaluation.json`) must be the LAST thing written -- so a crash partway through never
    leaves a false 'already evaluated' marker (only a possibly-stale predictions CSV / figure, which the
    guard does not key off of)."""
    src = _evaluate_final_source()
    idx_guard = src.index("out_path.is_file()")
    idx_predict = src.index("predict(model, test_loader, device)")
    idx_csv = src.index('open(pred_path, "w"')
    idx_fig = src.index("plot_confusion_matrix(")
    idx_final_save = src.rindex("save_json(")   # the LAST save_json call must be the one writing out_path
    assert idx_guard < idx_predict, "guard must be checked before running inference"
    assert idx_predict < idx_csv, "inference must happen before the predictions CSV is written"
    assert idx_fig < idx_final_save and idx_csv < idx_final_save, \
        "the guard-marker JSON (final save_json call) must be written AFTER the CSV and the figure"


def test_evaluate_final_never_recomputes_normalization_statistics():
    src = _evaluate_final_source()
    assert "fit_channel_stats" not in src, "evaluate_final.py must never fit/recompute normalization statistics"
    assert "ChannelStats.from_dict" in src, "evaluate_final.py must load SAVED statistics from the checkpoint"


def test_evaluate_final_runs_inference_exactly_once():
    src = _evaluate_final_source()
    # exactly one call site where predict() is actually invoked with the model/loader/device
    call_sites = [ln for ln in src.splitlines() if "predict(model" in ln]
    assert len(call_sites) == 1, f"expected exactly one predict(model, ...) call, found {len(call_sites)}: {call_sites}"


def test_evaluate_final_saves_all_expected_artifacts():
    src = _evaluate_final_source()
    for expected in ("final_test_predictions.csv", "confusion_matrix_final.png", "final_test_evaluation.json"):
        assert expected in src, f"evaluate_final.py must reference '{expected}'"
    for expected_key in ('"window_index"', '"subject_id"', '"true_label"', '"true_name"',
                        '"pred_label"', '"pred_name"', '"correct"'):
        assert expected_key in src, f"predictions CSV header must include {expected_key}"


def test_evaluate_final_force_flag_defaults_off():
    src = _evaluate_final_source()
    assert '"--force", action="store_true"' in src, "--force must default to False (action='store_true')"
    assert "not args.force" in src, "the guard must check 'not args.force'"


def test_scheduler_monitor_bug_regression():
    """Regression guard for the bug found during Step 4 review: the scheduler must be stepped with the
    value named by ITS OWN 'monitor' config key, not silently reused from early_stopping's monitor. This
    matters because Phase 2 configures scheduler.monitor='train_loss' while early_stopping.monitor stays
    'val_macro_f1' -- reusing the latter's value for the former, interpreted under the wrong 'mode', would
    silently break Phase 2's LR schedule (see the fit() docstring and _monitor_value())."""
    src = (Path(__file__).resolve().parents[1] / "src" / "train.py").read_text()
    assert "scheduler.step(monitor_value)" not in src, "scheduler must not reuse the early-stopping monitor value verbatim"
    assert 'scheduler.step(_monitor_value(sched_monitor_name, tr, va))' in src, \
        "scheduler must be stepped with a value looked up under its OWN configured monitor name"


def test_compute_metrics_rejects_out_of_range_labels():
    from src.metrics import compute_metrics
    names = ("A", "B", "C", "D", "E", "F")
    y = np.array([0, 1, 2])
    expect_raises(ValueError, lambda: compute_metrics(y, np.array([0, 1, 6]), names))     # pred out of range
    expect_raises(ValueError, lambda: compute_metrics(np.array([0, 1, -1]), y, names))    # true out of range
    expect_raises(ValueError, lambda: compute_metrics(np.array([]), np.array([]), names))  # empty


# ============================================================================= torch-gated
def test_device_handling_cpu_and_auto_and_unavailable_cuda_request():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    from src.train import get_device
    assert get_device("cpu").type == "cpu"
    auto = get_device("auto")
    assert auto.type == ("cuda" if torch.cuda.is_available() else "cpu")
    if not torch.cuda.is_available():
        expect_raises(RuntimeError, lambda: get_device("cuda"))
    else:
        assert get_device("cuda").type == "cuda"


def test_early_stopper_basic_behavior():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    es = EarlyStopper(mode="max", patience=2)
    # epoch 1: first value is always an improvement (best starts at -inf)
    assert es.step(0.5, 1) is True and es.best_epoch == 1 and es.n_bad_epochs == 0
    assert es.step(0.4, 2) is False and es.n_bad_epochs == 1     # worse
    assert es.should_stop is False                                 # 1 <= patience(2)
    assert es.step(0.45, 3) is False and es.n_bad_epochs == 2      # still worse than best(0.5)
    assert es.should_stop is False                                 # 2 <= patience(2)
    assert es.step(0.3, 4) is False and es.n_bad_epochs == 3
    assert es.should_stop is True                                  # 3 > patience(2)
    assert es.best == 0.5 and es.best_epoch == 1


def test_early_stopper_min_mode_and_tie_does_not_reset_patience():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    es = EarlyStopper(mode="min", patience=1)
    es.step(1.0, 1)
    assert es.step(1.0, 2) is False and es.n_bad_epochs == 1        # exact tie is NOT an improvement
    assert es.step(0.9, 3) is True and es.n_bad_epochs == 0         # strictly better resets patience


def test_monitor_value_selects_correct_field():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    tr, va = EpochResult(loss=0.7, accuracy=0.6), EpochResult(loss=0.3, accuracy=0.9, macro_f1=0.85)
    assert _monitor_value("val_macro_f1", tr, va) == 0.85
    assert _monitor_value("val_loss", tr, va) == 0.3
    assert _monitor_value("val_acc", tr, va) == 0.9
    assert _monitor_value("train_loss", tr, va) == 0.7
    assert _monitor_value("train_acc", tr, va) == 0.6
    expect_raises(KeyError, lambda: _monitor_value("nonsense", tr, va))


def _tiny_cfg(max_epochs=3):
    return {"optimizer": "adam", "lr": 1e-3, "weight_decay": 1e-4,
           "early_stopping": {"monitor": "val_macro_f1", "mode": "max", "patience": 100},  # won't trigger
           "scheduler": {"type": "reduce_on_plateau", "monitor": "val_macro_f1", "mode": "max", "factor": 0.5, "patience": 100},
           "max_epochs": max_epochs, "batch_size": 16}


def test_fit_end_to_end_phase1_style_on_synthetic_data():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    train, test, names = D.load_dataset(_root())
    prepared = Pp.prepare_from_raw(train, test, names, n_val_subjects=4, seed=42)
    loaders = make_loaders(prepared, batch_size=16, seed=0)
    out = Path(tempfile.mkdtemp())
    result = fit("A", loaders["train"], loaders["val"], prepared.class_names, _tiny_cfg(3), seed=0,
                run_name="test_A_s0", checkpoint_dir=out / "ckpt", metrics_dir=out / "metrics", device=torch.device("cpu"),
                max_epochs=3, normalization_stats=prepared.stats.to_dict(), channels=prepared.channels, quiet=True)
    assert result.epochs_ran == 3 and not result.stopped_early
    assert np.isfinite(result.best_val_loss) and 0 <= result.best_val_acc <= 1 and 0 <= result.best_val_macro_f1 <= 1
    assert Path(result.checkpoint_path).is_file() and Path(result.curves_path).is_file()
    import json
    curves = json.loads(Path(result.curves_path).read_text())
    for key in ("train_loss", "train_acc", "val_loss", "val_acc", "val_macro_f1", "lr"):
        assert len(curves["history"][key]) == 3, key

    m, analysis = evaluate(result.checkpoint_path, loaders["val"], torch.device("cpu"), class_names=prepared.class_names)
    assert 0 <= m.accuracy <= 1 and 0 <= m.macro_f1 <= 1
    assert int(m.confusion.sum()) == m.n_samples == len(prepared.y_val)
    assert "most_confused_pairs" in analysis and "lowest_f1" in analysis and "highest_f1" in analysis


def test_checkpoint_roundtrip_deterministic():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    train, test, names = D.load_dataset(_root())
    prepared = Pp.prepare_from_raw(train, test, names, n_val_subjects=4, seed=42)
    loaders = make_loaders(prepared, batch_size=16, seed=1)
    out = Path(tempfile.mkdtemp())
    result = fit("B", loaders["train"], loaders["val"], prepared.class_names, _tiny_cfg(2), seed=1,
                run_name="test_B_s1", checkpoint_dir=out / "ckpt", metrics_dir=out / "metrics", device=torch.device("cpu"),
                max_epochs=2, normalization_stats=prepared.stats.to_dict(), channels=prepared.channels, quiet=True)
    m1, _ = load_model_from_checkpoint(result.checkpoint_path, torch.device("cpu"))
    m2, _ = load_model_from_checkpoint(result.checkpoint_path, torch.device("cpu"))
    x = torch.randn(4, 6, 128)
    with torch.no_grad():
        assert torch.equal(m1(x), m2(x))


def test_evaluate_final_flow_predictions_align_with_test_windows_and_are_deterministic():
    """Integration test of the exact steps scripts/evaluate_final.py performs (load saved stats -> load
    test split -> build loader with shuffle=False -> predict once), without invoking the script itself.
    Verifies: predictions/labels line up 1:1 with the original test_raw order (so zipping with
    test_raw.subjects for the CSV is valid), the count matches the test set size, and running predict()
    twice on the same model/loader gives bit-identical results (eval mode, no randomness) -- i.e. it is
    safe to treat one predict() call as authoritative and never needs to be re-run."""
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    from src.dataset import make_test_loader_from_stats
    from src.evaluate import load_model_from_checkpoint, predict
    from src.metrics import compute_metrics, group_confusion_breakdown

    train_raw, class_names = D.load_train_only(_root())
    final = Pp.prepare_train_only_for_final(train_raw, class_names)
    loader = make_final_train_loader(final, batch_size=16, seed=123)
    out = Path(tempfile.mkdtemp())
    cfg = _tiny_cfg(1)
    cfg["early_stopping"] = {"monitor": "val_macro_f1", "mode": "max", "patience": 999}
    result = fit("A", loader, loader, final.class_names, cfg, seed=123, run_name="test_eval_final_A",
                checkpoint_dir=out / "ckpt", metrics_dir=out / "metrics", device=torch.device("cpu"),
                max_epochs=1, normalization_stats=final.stats.to_dict(), channels=final.channels,
                quiet=True, checkpoint_mode="last")

    test_raw, _ = D.load_test_only(_root(), final.channels)
    test_loader = make_test_loader_from_stats(test_raw.x, test_raw.y, final.stats, batch_size=16)
    model, ckpt = load_model_from_checkpoint(result.checkpoint_path, torch.device("cpu"))
    y_pred1, y_true1 = predict(model, test_loader, torch.device("cpu"))
    y_pred2, y_true2 = predict(model, test_loader, torch.device("cpu"))

    assert len(y_pred1) == len(y_true1) == len(test_raw)
    assert np.array_equal(y_true1, test_raw.y), "predict() labels must be in the SAME order as test_raw"
    assert np.array_equal(y_pred1, y_pred2) and np.array_equal(y_true1, y_true2), \
        "two predict() calls on the same (eval-mode) model/loader must be bit-identical"

    m = compute_metrics(y_true1, y_pred1, class_names)
    groups = {"static": ["SITTING", "STANDING", "LAYING"], "dynamic": ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"]}
    gb = group_confusion_breakdown(m, groups)
    assert gb["total_errors"] == int((y_true1 != y_pred1).sum())


def test_fit_checkpoint_mode_last_saves_final_epoch_not_best():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)"); return
    train, test, names = D.load_dataset(_root())
    final = Pp.prepare_full_train_for_final(train, test, names)
    loader = make_final_train_loader(final, batch_size=16, seed=123)
    out = Path(tempfile.mkdtemp())
    cfg = _tiny_cfg(3)
    cfg["early_stopping"] = {"monitor": "val_macro_f1", "mode": "max", "patience": 999}
    cfg["scheduler"] = {"type": "reduce_on_plateau", "monitor": "train_loss", "mode": "min", "factor": 0.5, "patience": 999}
    result = fit("A", loader, loader, final.class_names, cfg, seed=123, run_name="test_final_A",
                checkpoint_dir=out / "ckpt", metrics_dir=out / "metrics", device=torch.device("cpu"),
                max_epochs=3, normalization_stats=final.stats.to_dict(), channels=final.channels,
                quiet=True, checkpoint_mode="last")
    assert result.epochs_ran == 3
    ckpt = torch.load(result.checkpoint_path, map_location="cpu", weights_only=False)
    assert ckpt["epoch"] == 3, "checkpoint_mode='last' must save the FINAL epoch, not an in-training best"
    assert ckpt["checkpoint_mode"] == "last"


# ============================================================================= runner
def main() -> int:
    print(f"torch available: {HAVE_TORCH}\n")
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
    if "tmp" in _CACHE:
        shutil.rmtree(_CACHE["tmp"], ignore_errors=True)
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
