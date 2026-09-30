"""Tests for Step 3 (model + computational analysis). Run with plain Python:

    python tests/test_model.py

If torch is installed, these test the REAL nn.Module (src/model.py) via src/complexity.py, and also
cross-check it against the tensor-free formulas in src/shape_calc.py. If torch is NOT installed (as in
this sandbox), they fall back to the pure-NumPy reference (src/np_reference.py), which still executes a
real forward pass with real arithmetic -- only the specific "on top of real torch ops" claim is weaker;
shapes/params/MACs/receptive-field are exact either way, since they follow from the architecture's
hyperparameters, not from training.
"""
import sys
import traceback
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.architectures import ARCHITECTURES, INPUT_CHANNELS, INPUT_LENGTH, N_CLASSES, PARAM_BUDGET  # noqa: E402
from src import shape_calc as S                                                                        # noqa: E402
from src import np_reference as NP                                                                     # noqa: E402

try:
    import torch  # noqa: F401
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False

if HAVE_TORCH:
    from src.model import build_model, dummy_input          # noqa: E402
    from src import complexity as C                          # noqa: E402


def expect_raises(exc, fn):
    try:
        fn()
    except exc:
        return
    raise AssertionError(f"expected {exc.__name__} was not raised")


# ------------------------------------------------------------------------------- spec-level (formula) tests, always run
def test_specs_produce_six_class_output_shape():
    for name, spec in ARCHITECTURES.items():
        rep = S.analyze(spec, name=name)
        assert rep.layers[-1].out_shape == (N_CLASSES,), (name, rep.layers[-1].out_shape)


def test_all_models_under_budget_by_formula():
    for name, spec in ARCHITECTURES.items():
        rep = S.analyze(spec, name=name)
        assert rep.total_params < PARAM_BUDGET, (name, rep.total_params)
        S.assert_within_budget(rep, PARAM_BUDGET)


def test_param_counts_reproducible():
    for name, spec in ARCHITECTURES.items():
        a, b = S.analyze(spec, name=name), S.analyze(spec, name=name)
        assert a.total_params == b.total_params and a.total_macs == b.total_macs


def test_macs_and_receptive_field_positive():
    for name, spec in ARCHITECTURES.items():
        rep = S.analyze(spec, name=name)
        assert rep.total_macs > 0
        assert rep.receptive_field > 1
        assert rep.jump >= 1


def test_receptive_field_grows_with_more_layers_or_wider_kernels():
    # Model B stacks one more conv+pool than A; its receptive field must be at least as large.
    a, b = S.analyze(ARCHITECTURES["A"], "A"), S.analyze(ARCHITECTURES["B"], "B")
    assert b.receptive_field > a.receptive_field


def test_post_gap_coverage_covers_full_window_and_never_exceeds_it():
    for name, spec in ARCHITECTURES.items():
        rep = S.analyze(spec, name=name)
        assert rep.gap_input_length > 1
        assert rep.post_gap_coverage_raw >= rep.receptive_field    # GAP coverage >= a single local feature's RF
        assert rep.post_gap_coverage == INPUT_LENGTH                # all three models see the FULL 128-sample window
        assert rep.post_gap_coverage <= INPUT_LENGTH                # capped, must never exceed the real input


def test_local_rf_matches_hand_verified_values():
    # Cross-checked by an independent recurrence trace in the accompanying write-up.
    expected = {"A": 18, "B": 36, "C": 60}
    for name, spec in ARCHITECTURES.items():
        assert S.analyze(spec, name=name).receptive_field == expected[name], name


def test_bad_spec_shape_mismatch_is_caught():
    from src.architectures import ConvOp, GAPOp, LinearOp
    bad_spec = (ConvOp(INPUT_CHANNELS, 8, kernel=3), GAPOp(), LinearOp(999, N_CLASSES))
    expect_raises(ValueError, lambda: S.analyze(bad_spec))


def test_conv_out_length_formula():
    assert S.conv_or_pool_out_length(128, kernel=9, stride=1, padding=4) == 128    # 'same'
    assert S.conv_or_pool_out_length(128, kernel=2, stride=2, padding=0) == 64     # pool halves
    assert S.conv_or_pool_out_length(64, kernel=2, stride=2, padding=0) == 32


# ------------------------------------------------------------------------------- real forward pass (NumPy, always run)
def test_np_reference_forward_matches_formula_and_is_finite():
    for name, spec in ARCHITECTURES.items():
        ref = S.analyze(spec, name=name)
        logits, real = NP.run(name, spec, seed=0)
        assert logits.shape == (N_CLASSES,)
        assert np.isfinite(logits).all()
        assert real.total_params == ref.total_params, (name, "params", real.total_params, ref.total_params)
        assert real.total_macs == ref.total_macs, (name, "macs", real.total_macs, ref.total_macs)
        assert real.receptive_field == ref.receptive_field, (name, "RF", real.receptive_field, ref.receptive_field)
        assert real.post_gap_coverage == ref.post_gap_coverage, (name, "post-GAP", real.post_gap_coverage, ref.post_gap_coverage)
        assert real.peak_activation.elements == ref.peak_activation.elements


def test_np_reference_deterministic_with_seed():
    l1, r1 = NP.run("B", ARCHITECTURES["B"], seed=3)
    l2, r2 = NP.run("B", ARCHITECTURES["B"], seed=3)
    assert np.array_equal(l1, l2) and r1.total_params == r2.total_params


# ------------------------------------------------------------------------------- torch-only tests (skipped without torch)
def test_torch_models_accept_batch_and_output_shape():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    for name in ARCHITECTURES:
        model = build_model(name)
        for B in (1, 4):
            out = model(dummy_input(B))
            assert tuple(out.shape) == (B, N_CLASSES)
        expect_raises(ValueError, lambda m=model: m(torch.randn(1, 3, INPUT_LENGTH)))  # wrong channel count


def test_torch_params_under_budget_from_real_model():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    for name in ARCHITECTURES:
        model = build_model(name)
        br = C.count_parameters(model)
        assert br.trainable == br.total  # no frozen/non-trainable params in these models
        assert br.trainable < PARAM_BUDGET
        assert sum(n for _, _, n in br.by_layer) == br.trainable


def test_torch_hooks_agree_with_formula():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    for name, spec in ARCHITECTURES.items():
        model = build_model(name)
        hook_report = C.measure(model)
        C.cross_check_against_spec(hook_report, spec)   # raises AssertionError on any mismatch


def test_torch_forward_no_nan_inf():
    if not HAVE_TORCH:
        print("  (skipped: torch not installed)")
        return
    for name in ARCHITECTURES:
        model = build_model(name)
        out = model(dummy_input(4))
        assert torch.isfinite(out).all()


# ------------------------------------------------------------------------------- runner
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
    print(f"\n{len(tests) - failed}/{len(tests)} tests passed")
    return 1 if failed else 0


if __name__ == "__main__":
    sys.exit(main())
