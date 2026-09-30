"""Tests for the research extension. Run with plain Python:

    python tests/test_research_extension.py

Two tiers, exactly like tests/test_model.py and tests/test_step4.py:
  * always run: architecture specs (shape_calc_ext, no torch), the numpy reference forward pass
    (np_reference_ext), sensor-subset selection logic, and the official-test firewall source scan --
    none of these need torch or a real dataset.
  * torch-gated: building the REAL nn.Module for Linear/D, cross-checking its params/MACs against the
    tensor-free formulas, and the causal behavior test on the real torch model. These print
    "(skipped: torch not installed)" when torch is absent, same as the existing test files.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from experiments.research_extension.architectures_ext import (LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES,  # noqa: E402
                                                               MODEL_D_PARAM_BUDGET_HARD, MODEL_D_STAGES,
                                                               model_a_variant_spec)
from experiments.research_extension.data_ext import (ACC_ONLY_CHANNELS, FULL_CHANNELS, GYRO_ONLY_CHANNELS,  # noqa: E402
                                                      SENSOR_SUBSETS, validate_channel_subset)
from experiments.research_extension.np_reference_ext import NPModelD, causal_conv1d, run_linear          # noqa: E402
from experiments.research_extension.official_test_firewall import (FORBIDDEN_SYMBOLS,                     # noqa: E402
                                                                    check_package_is_firewalled)
from experiments.research_extension.pareto import ParetoPoint, is_dominated, pareto_frontier               # noqa: E402
from experiments.research_extension.per_subject import per_subject_metrics, summarize_subject_variability  # noqa: E402
from experiments.research_extension.shape_calc_ext import analyze_linear, analyze_model_d, weight_storage_bytes  # noqa: E402
from experiments.research_extension.streaming_state import total_streaming_state                            # noqa: E402
from src import shape_calc as S                                                                              # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from experiments.research_extension.model_ext import (LinearBaseline, TinyCausalCNN, build_model_ext,  # noqa: E402
                                                           build_sensor_ablation_model, dummy_input)


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


# ============================================================================= always run (no torch)
# ---- RQ1: linear baseline -----------------------------------------------------------------------
def test_linear_param_count_formula():
    rep = analyze_linear()
    assert rep.total_params == 4614, rep.total_params
    assert rep.total_macs == 4608, rep.total_macs
    assert LINEAR_IN_FEATURES == 768 and LINEAR_OUT_FEATURES == 6


# ---- RQ2: Model D, tensor-free formula ------------------------------------------------------------
def test_model_d_under_param_budget():
    rep = analyze_model_d()
    assert rep.total_params < MODEL_D_PARAM_BUDGET_HARD, rep.total_params
    assert rep.total_params < 10_000, f"D has {rep.total_params} params, expected comfortably under 10k"


def test_model_d_exact_analytic_counts():
    rep = analyze_model_d()
    assert rep.total_params == 3906, rep.total_params
    assert rep.total_macs == 406560, rep.total_macs
    assert rep.receptive_field == 61, rep.receptive_field
    assert rep.jump == 1
    assert rep.post_gap_coverage == 128


def test_model_d_output_length_preserved_by_causal_padding():
    """Every stage of D uses stride 1 with causal (left-only) padding of dilation*(kernel-1); a stage's
    output length must equal its input length (128) at every layer -- verified from the layer trace."""
    rep = analyze_model_d()
    for layer in rep.layers:
        # GAP intentionally collapses length to 1 (that's the windowed-aggregation step, not a causal
        # conv), and the trailing Linear has no length axis at all -- both are expected exceptions.
        if layer.op_type in ("AdaptiveAvgPool1d", "Linear"):
            continue
        assert layer.out_shape[1] == 128, (layer.op_type, layer.out_shape)


def test_model_d_receptive_field_matches_manual_history_sum():
    """RF - 1 must equal sum over stages of dilation*(kernel-1) (each stage's history contribution),
    since jump stays 1 throughout (stride 1 everywhere)."""
    rep = analyze_model_d()
    manual = 1 + sum(dilation * (kernel - 1) for (_, _, kernel, dilation) in MODEL_D_STAGES)
    assert rep.receptive_field == manual, (rep.receptive_field, manual)


def test_causal_padding_is_not_symmetric_same_padding():
    """For dilation>1, (kernel-1)//2 (the symmetric 'same' padding src.architectures.ConvOp uses) is a
    DIFFERENT, WRONG number for a causal conv: causal left-padding must be dilation*(kernel-1), not
    (kernel-1)//2, and the two only coincide by accident for dilation=1 with even-ish kernels."""
    for (_, _, kernel, dilation) in MODEL_D_STAGES:
        causal_left_pad = dilation * (kernel - 1)
        symmetric_same_pad = (kernel - 1) // 2
        if dilation > 1:
            assert causal_left_pad != symmetric_same_pad, (kernel, dilation)


# ---- causality test on the numpy reference (no torch needed) --------------------------------------
def test_np_model_d_is_causal():
    """Perturbing input samples strictly AFTER time t0 must leave the causal feature map at times
    <= t0 completely unchanged. This is the causality test requested in the brief: implementation +
    test, not just documentation."""
    rng = np.random.RandomState(0)
    x = rng.randn(6, 128)
    m = NPModelD(seed=0)
    feats_before = m.forward_features(x)

    for t0 in (0, 1, 30, 61, 100, 126):
        x_perturbed = x.copy()
        if t0 + 1 < 128:
            x_perturbed[:, t0 + 1:] += rng.randn(*x_perturbed[:, t0 + 1:].shape) * 1000.0
        feats_after = m.forward_features(x_perturbed)
        assert np.allclose(feats_before[:, :t0 + 1], feats_after[:, :t0 + 1], atol=1e-8), \
            f"causality violated: perturbing samples after t0={t0} changed features at/before t0"


def test_np_model_d_perturbation_does_change_later_outputs():
    """Sanity counterpart to the causality test above: the perturbation used there must actually be
    large enough to change LATER outputs, or the causality check would be vacuous (e.g. because the
    network happens to ignore its input)."""
    rng = np.random.RandomState(1)
    x = rng.randn(6, 128)
    m = NPModelD(seed=1)
    feats_before = m.forward_features(x)
    t0 = 50
    x_perturbed = x.copy()
    x_perturbed[:, t0 + 1:] += 1000.0
    feats_after = m.forward_features(x_perturbed)
    assert not np.allclose(feats_before[:, t0 + 1:], feats_after[:, t0 + 1:], atol=1e-3)


def test_np_causal_conv1d_matches_manual_dot_product():
    """Direct, from-scratch check of one causal conv output against a hand-written dot product, for a
    tiny synthetic case (groups=1, dilation=2), independent of the vectorised implementation."""
    rng = np.random.RandomState(0)
    c_in, c_out, k, d, L = 2, 3, 3, 2, 10
    x = rng.randn(c_in, L)
    w = rng.randn(c_out, c_in, k)
    b = rng.randn(c_out)
    out = causal_conv1d(x, w, b, dilation=d, groups=1)
    assert out.shape == (c_out, L)
    left_pad = d * (k - 1)
    xp = np.pad(x, ((0, 0), (left_pad, 0)))
    t = 7
    expected = np.tensordot(w, xp[:, t:t + (k - 1) * d + 1:d], axes=([1, 2], [0, 1])) + b
    assert np.allclose(out[:, t], expected)


def test_run_linear_shape():
    x_flat = np.random.RandomState(0).randn(768)
    out = run_linear(x_flat)
    assert out.shape == (6,)


# ---- RQ4: sensor subset selection -----------------------------------------------------------------
def test_sensor_subset_channel_counts():
    assert len(FULL_CHANNELS) == 6
    assert len(ACC_ONLY_CHANNELS) == 3
    assert len(GYRO_ONLY_CHANNELS) == 3
    assert all(c.startswith("total_acc") for c in ACC_ONLY_CHANNELS)
    assert all(c.startswith("body_gyro") for c in GYRO_ONLY_CHANNELS)
    assert set(ACC_ONLY_CHANNELS) | set(GYRO_ONLY_CHANNELS) == set(FULL_CHANNELS)


def test_sensor_subset_validation_rejects_unknown_and_duplicate():
    expect_raises(ValueError, lambda: validate_channel_subset(("not_a_real_channel",)))
    expect_raises(ValueError, lambda: validate_channel_subset(("total_acc_x", "total_acc_x")))
    validate_channel_subset(SENSOR_SUBSETS["acc_only"])   # must not raise


def test_model_a_variant_spec_only_changes_first_layer_in_channels():
    from src.architectures import MODEL_A

    spec3 = model_a_variant_spec(3)
    spec6 = model_a_variant_spec(6)
    assert spec3[0].in_ch == 3 and spec6[0].in_ch == 6
    assert spec3[1:] == MODEL_A[1:] == spec6[1:]
    assert spec6[0].out_ch == MODEL_A[0].out_ch and spec6[0].kernel == MODEL_A[0].kernel


# ---- per-subject analysis ---------------------------------------------------------------------------
def test_per_subject_metrics_and_variability_synthetic():
    rng = np.random.RandomState(0)
    class_names = ("A", "B")
    subjects = np.array([1] * 50 + [2] * 50)
    y_true = rng.randint(0, 2, 100)
    y_pred = y_true.copy()
    # subject 2 gets every 3rd prediction flipped -> worse than subject 1 (perfect)
    y_pred[50:][::3] = 1 - y_pred[50:][::3]
    rows = per_subject_metrics(y_true, y_pred, subjects, class_names)
    assert [r["subject_id"] for r in rows] == [1, 2]
    assert rows[0]["accuracy"] == 1.0
    assert rows[1]["accuracy"] < 1.0
    summary = summarize_subject_variability(rows)
    assert summary["n_subjects"] == 2
    assert summary["worst_subject_id"] == 2
    assert summary["best_subject_id"] == 1


def test_per_subject_metrics_rejects_length_mismatch():
    expect_raises(ValueError, lambda: per_subject_metrics(np.zeros(3), np.zeros(2), np.zeros(3), ("a",)))


# ---- Pareto -------------------------------------------------------------------------------------------
def test_pareto_dominance_basic():
    a = ParetoPoint("A", metric=0.99, cost=100)
    b = ParetoPoint("B", metric=0.98, cost=200)   # worse metric AND worse (higher) cost than A -> dominated
    c = ParetoPoint("C", metric=0.995, cost=150)  # better metric, worse cost than A -> not dominated by A
    points = [a, b, c]
    dom_b, by = is_dominated(b, points)
    assert dom_b and by == "A"
    dom_a, _ = is_dominated(a, points)
    assert not dom_a   # nothing beats A on cost AND matches/beats it on metric
    result = pareto_frontier(points)
    assert result["B"]["dominated"] is True and result["B"]["dominated_by"] == "A"
    assert result["A"]["dominated"] is False


def test_pareto_ties_do_not_count_as_domination():
    a = ParetoPoint("A", metric=0.99, cost=100)
    a2 = ParetoPoint("A2", metric=0.99, cost=100)   # identical point: no STRICT improvement either way
    dom, _ = is_dominated(a, [a, a2])
    assert not dom


# ---- streaming state -----------------------------------------------------------------------------------
def test_streaming_state_ring_buffer_matches_history_formula():
    state = total_streaming_state()
    manual_total = sum(in_ch * (dilation * (kernel - 1)) for (in_ch, _, kernel, dilation) in MODEL_D_STAGES)
    assert state["ring_buffer_scalars_total"] == manual_total, (state["ring_buffer_scalars_total"], manual_total)
    assert state["persistent_state_fp32_bytes"] == state["persistent_state_scalars_total"] * 4


# ---- official test firewall -----------------------------------------------------------------------------
def test_official_test_firewall_module_never_references_forbidden_symbols():
    pkg_dir = Path(__file__).resolve().parents[1] / "experiments" / "research_extension"
    violations = check_package_is_firewalled(pkg_dir)
    assert violations == {}, f"firewall violation(s) found: {violations}"


def test_official_test_firewall_symbol_list_is_nonempty():
    # a trivial guard against someone accidentally emptying FORBIDDEN_SYMBOLS and making the check vacuous
    assert len(FORBIDDEN_SYMBOLS) >= 4


def test_official_test_firewall_catches_a_planted_violation(tmp_path=None):
    import tempfile

    tmp_dir = Path(tempfile.mkdtemp())
    bad_file = tmp_dir / "bad_module.py"
    bad_file.write_text("from src.data import load_dataset\n\ndef f():\n    return load_dataset('x')\n")
    from experiments.research_extension.official_test_firewall import check_module_is_firewalled
    found = check_module_is_firewalled(bad_file)
    assert found == ["load_dataset"]


# ---- research-extension outputs cannot overwrite frozen result paths ------------------------------------
def test_research_extension_output_paths_are_isolated_from_frozen_paths():
    """The extension's own output roots must never equal, or be a parent of, the frozen result roots
    used by scripts/run_experiments.py and scripts/evaluate_final.py."""
    frozen_roots = {"results/checkpoints", "results/metrics", "results/figures", "results/experiment_log.csv",
                    "report/report.md", "report/step6_streaming_analysis.md"}
    extension_roots = {"results/research_extension/checkpoints", "results/research_extension/metrics",
                       "results/research_extension/figures", "report/research_extension.md"}
    for ext in extension_roots:
        for frozen in frozen_roots:
            assert not ext.startswith(frozen) and not frozen.startswith(ext), (ext, frozen)
            assert ext != frozen



# ---- output guard + entry-script firewall ------------------------------------------------------------
def test_output_guard_allows_extension_dirs_and_rejects_frozen_and_outside():
    from experiments.research_extension.output_guard import assert_output_dir_safe
    root = Path(__file__).resolve().parents[1]
    for sub in ("checkpoints", "metrics", "figures", "_smoke/metrics"):
        assert_output_dir_safe(root / "results" / "research_extension" / sub, root)
    for bad in ("results/checkpoints", "results/metrics", "results/figures", "results", "results/smoke_test",
                "report", "configs", "results/research_extension/../metrics", "/tmp/elsewhere"):
        expect_raises(PermissionError, lambda b=bad: assert_output_dir_safe(b, root))


def test_output_guard_refuses_to_overwrite_existing_extension_outputs():
    import tempfile
    from experiments.research_extension.output_guard import assert_no_existing_outputs
    f = Path(tempfile.mkdtemp()) / "x.pt"
    assert_no_existing_outputs([f], overwrite=False)   # nonexistent: fine
    f.write_text("x")
    expect_raises(FileExistsError, lambda: assert_no_existing_outputs([f], overwrite=False))
    assert_no_existing_outputs([f], overwrite=True)


def test_entry_script_is_firewalled_and_writes_only_via_guarded_dirs():
    from experiments.research_extension.official_test_firewall import check_module_is_firewalled
    script = Path(__file__).resolve().parents[1] / "scripts" / "run_research_extension.py"
    assert check_module_is_firewalled(script) == []
    src = script.read_text()
    assert "assert_output_dir_safe" in src and '"results" / "research_extension"' in src
    for frozen_write in ('"results/checkpoints"', '"results/metrics"', '"results/figures"'):
        assert frozen_write not in src, frozen_write


def test_planned_runs_count_is_18():
    import importlib.util
    spec = importlib.util.spec_from_file_location("rre", Path(__file__).resolve().parents[1] / "scripts" / "run_research_extension.py")
    m = importlib.util.module_from_spec(spec); spec.loader.exec_module(m)
    names = m.planned_run_names([0, 1, 2])
    assert len(names) == 18 and len(set(names)) == 18
    assert all(n.startswith("research_extension_") for n in names)   # can never collide with phase1_/final_ names


def test_normalization_for_subset_is_fit_on_training_windows_only_synthetic():
    """Subset normalization: stats from train windows only, independent of validation values."""
    from src.preprocessing import fit_channel_stats, standardize
    rng = np.random.RandomState(0)
    tr = rng.randn(40, 3, 128) * 2 + 5
    va_a, va_b = rng.randn(10, 3, 128), rng.randn(10, 3, 128) * 100
    st1 = fit_channel_stats(tr, ACC_ONLY_CHANNELS)
    st2 = fit_channel_stats(tr, ACC_ONLY_CHANNELS)
    assert st1.mean == st2.mean and st1.std == st2.std and st1.n_windows == 40
    assert len(st1.mean) == 3
    z = standardize(tr, st1)
    assert abs(z.mean()) < 1e-4 and abs(z.std() - 1) < 1e-3
    assert standardize(va_a, st1).shape == va_a.shape and standardize(va_b, st1).shape == va_b.shape

# ============================================================================= torch-gated
def test_model_d_real_module_shape_and_param_count():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    model = TinyCausalCNN()
    x = dummy_input(batch_size=2)
    out = model(x)
    assert tuple(out.shape) == (2, 6), tuple(out.shape)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 3906, n_params


def test_linear_real_module_shape_and_param_count():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    model = LinearBaseline()
    x = dummy_input(batch_size=4)
    out = model(x)
    assert tuple(out.shape) == (4, 6)
    n_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    assert n_params == 4614, n_params


def test_model_d_real_module_macs_match_formula():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    from experiments.research_extension.shape_calc_ext import analyze_model_d

    model = TinyCausalCNN()
    macs = 0
    x = dummy_input(batch_size=1)
    hooks_out = {}

    def make_hook(mod_name):
        def hook(mod, inp, outp):
            k = mod.kernel_size[0] if hasattr(mod, "kernel_size") else 1
            in_ch = mod.in_channels // mod.groups if hasattr(mod, "in_channels") else mod.in_features
            out_ch = mod.out_channels if hasattr(mod, "out_channels") else mod.out_features
            length = outp.shape[-1] if outp.dim() == 3 else 1
            hooks_out[mod_name] = length * out_ch * k * in_ch
        return hook

    handles = []
    for name, mod in model.named_modules():
        if isinstance(mod, (torch.nn.Conv1d, torch.nn.Linear)):
            handles.append(mod.register_forward_hook(make_hook(name)))
    model.eval()
    with torch.no_grad():
        model(x)
    for h in handles:
        h.remove()
    macs = sum(hooks_out.values())
    formula_macs = analyze_model_d().total_macs
    assert macs == formula_macs, (macs, formula_macs)


def test_model_d_real_module_is_causal():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    model = TinyCausalCNN()
    model.eval()
    torch.manual_seed(0)
    x = torch.randn(1, 6, 128)

    def features(inp):
        h = inp
        for block in model.blocks:
            h = block["depthwise"](h); h = block["pointwise"](h); h = block["bn"](h); h = block["relu"](h)
        return h

    with torch.no_grad():
        feats_before = features(x)
        t0 = 61
        x2 = x.clone()
        x2[:, :, t0 + 1:] += 1000.0
        feats_after = features(x2)
    assert torch.allclose(feats_before[:, :, :t0 + 1], feats_after[:, :, :t0 + 1], atol=1e-5), \
        "real torch Model D violates causality"
    assert not torch.allclose(feats_before[:, :, t0 + 1:], feats_after[:, :, t0 + 1:], atol=1e-3)


def test_sensor_ablation_model_accepts_three_channels():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    model = build_sensor_ablation_model(3)
    x = torch.randn(2, 3, 128)
    out = model(x)
    assert tuple(out.shape) == (2, 6)
    expect_raises(ValueError, lambda: model(torch.randn(2, 6, 128)))


def test_official_test_firewall_extension_has_no_torch_dataset_calls():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    # train_ext.py and run_research_extension.py must not import forbidden test-loader symbols either
    pkg_dir = Path(__file__).resolve().parents[1] / "experiments" / "research_extension"
    violations = check_package_is_firewalled(pkg_dir)
    assert violations == {}


# ============================================================================= runner
def _run_all():
    tests = [(name, fn) for name, fn in globals().items() if name.startswith("test_") and callable(fn)]
    n_pass, n_fail = 0, 0
    for name, fn in tests:
        try:
            fn()
            print(f"[PASS] {name}")
            n_pass += 1
        except Exception:
            print(f"[FAIL] {name}")
            traceback.print_exc()
            n_fail += 1
    print(f"\n{n_pass} passed, {n_fail} failed, {len(tests)} total.  HAVE_TORCH={HAVE_TORCH}")
    return 0 if n_fail == 0 else 1


if __name__ == "__main__":
    sys.exit(_run_all())
