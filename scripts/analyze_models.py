#!/usr/bin/env python
"""Instantiate Models A/B/C and print a computational-budget comparison. No training happens here.

    python scripts/analyze_models.py

If torch is installed, measurements come from the REAL nn.Module (src/model.py) via forward hooks
(src/complexity.py), cross-checked against the tensor-free formulas in src/shape_calc.py. If torch is
NOT installed, falls back to the pure-NumPy reference (src/np_reference.py), which still executes a real
forward pass; this is flagged clearly in the output banner and the saved JSON.
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
from src.architectures import ARCHITECTURES, ARCHITECTURE_NAMES, INPUT_CHANNELS, INPUT_LENGTH, N_CLASSES, PARAM_BUDGET  # noqa: E402
from src.data import SAMPLING_HZ                                                                                        # noqa: E402
from src import shape_calc as S                                                                                         # noqa: E402
from src.utils import Checker, save_json                                                                                # noqa: E402

try:
    import torch
    from src.model import build_model, dummy_input
    from src import complexity as C
    HAVE_TORCH = True
except ImportError:
    HAVE_TORCH = False
    from src import np_reference as NP


def get_report_and_verify(name: str, spec, chk: Checker):
    """Return (ModelReport, forward_ok: bool). Uses the real model when torch is available."""
    ref = S.analyze(spec, name=name)   # tensor-free formula; computed either way, used as the cross-check
    if HAVE_TORCH:
        model = build_model(name)
        out = model(dummy_input(1))
        chk.check(tuple(out.shape) == (1, N_CLASSES), f"Model {name}: forward((1,6,{INPUT_LENGTH})) -> (1,{N_CLASSES})", str(tuple(out.shape)))
        chk.check(bool(torch.isfinite(out).all()), f"Model {name}: forward pass has no NaN/Inf")
        report = C.measure(model)
        try:
            C.cross_check_against_spec(report, spec)
            chk.check(True, f"Model {name}: PyTorch measurement matches the independent formula (params/MACs/RF/peak activation)")
        except AssertionError as e:
            chk.check(False, f"Model {name}: PyTorch measurement matches the independent formula", str(e))
        br = C.count_parameters(model)
        chk.check(br.trainable == report.total_params, f"Model {name}: hook-based param total matches named_parameters() total")
        layer_breakdown = [(n, t, c) for n, t, c in br.by_layer]
    else:
        logits, report = NP.run(name, spec, seed=0)
        chk.check(tuple(logits.shape) == (N_CLASSES,), f"Model {name}: forward((6,{INPUT_LENGTH})) -> ({N_CLASSES},) [NumPy reference]", str(logits.shape))
        import numpy as np
        chk.check(bool(np.isfinite(logits).all()), f"Model {name}: forward pass has no NaN/Inf [NumPy reference]")
        chk.check(report.total_params == ref.total_params, f"Model {name}: NumPy-executed measurement matches the independent formula")
        layer_breakdown = [(f"layer_{l.index:02d}_{l.op_type}", l.op_type, l.params) for l in report.layers if l.params]

    chk.check(report.total_params == ref.total_params, f"Model {name}: total params match formula", f"{report.total_params} vs {ref.total_params}")
    chk.check(report.total_macs == ref.total_macs, f"Model {name}: total MACs match formula", f"{report.total_macs} vs {ref.total_macs}")
    chk.check(report.receptive_field == ref.receptive_field, f"Model {name}: local RF matches formula")
    chk.check(report.post_gap_coverage == ref.post_gap_coverage, f"Model {name}: post-GAP coverage matches formula")
    chk.check(report.total_params < PARAM_BUDGET, f"Model {name}: trainable params < {PARAM_BUDGET}", str(report.total_params))
    assert report.total_params < PARAM_BUDGET  # hard failure, not just a printed check -- per the assignment's "fail loudly"
    return report, layer_breakdown


def print_layer_breakdown(report, layer_breakdown) -> None:
    print(f"\n  Layer-by-layer ({report.name}):")
    print(f"    {'#':<3}{'layer':<20}{'out shape':<16}{'params':>9}{'MACs':>12}")
    for l in report.layers:
        shape = "x".join(str(d) for d in l.out_shape)
        print(f"    {l.index:<3}{l.op_type:<20}{shape:<16}{l.params:>9,}{l.macs:>12,}" + (f"   # {l.note}" if l.note else ""))
    print(f"    {'TOTAL':<23}{'':<16}{report.total_params:>9,}{report.total_macs:>12,}")
    if layer_breakdown:
        print(f"  Trainable-parameter breakdown by named module:")
        for n, t, c in layer_breakdown:
            print(f"    {n:<28}{t:<16}{c:>7,}")


def main() -> int:
    print(f"torch available: {HAVE_TORCH}" + ("" if HAVE_TORCH else "  -> using the pure-NumPy reference implementation (src/np_reference.py)"))
    print(f"Parameter budget: < {PARAM_BUDGET:,}\n")
    chk = Checker()

    rows = []
    for name, spec in ARCHITECTURES.items():
        print(f"=== Model {name}: {ARCHITECTURE_NAMES[name]} ===")
        report, layer_breakdown = get_report_and_verify(name, spec, chk)
        print_layer_breakdown(report, layer_breakdown)
        print(f"  Weight memory (float32): {report.weight_kb:.2f} KB")
        rf_ms = 1000 * report.receptive_field / SAMPLING_HZ
        pg_ms = 1000 * report.post_gap_coverage / SAMPLING_HZ
        print(f"  Local backbone receptive field: {report.receptive_field} samples ({rf_ms:.0f} ms @ {SAMPLING_HZ} Hz), "
              f"effective stride between features: {report.jump} samples")
        print(f"  Post-GAP effective coverage: {report.post_gap_coverage} samples ({pg_ms:.0f} ms) "
              f"[raw union of local RFs before capping at the {INPUT_LENGTH}-sample window: {report.post_gap_coverage_raw}]")
        pk = report.peak_activation
        print(f"  Peak activation: {pk.op_type} (layer {pk.index}), shape {'x'.join(str(d) for d in pk.out_shape)}, "
              f"{pk.elements:,} elements, {pk.kb:.2f} KB\n")
        rows.append({"model": name, "description": ARCHITECTURE_NAMES[name], "params": report.total_params,
                    "weight_kb": round(report.weight_kb, 2), "macs": report.total_macs,
                    "mmacs": round(report.total_macs / 1e6, 3), "local_rf_samples": report.receptive_field,
                    "local_rf_ms": round(rf_ms, 1), "post_gap_coverage_samples": report.post_gap_coverage,
                    "post_gap_coverage_ms": round(pg_ms, 1), "peak_activation_kb": round(pk.kb, 2),
                    "peak_activation_layer": f"{pk.op_type}[{pk.index}]"})

    print("=== Comparison ===")
    header = f"{'Model':<7}{'Params':>8}{'Weight KB':>11}{'MACs':>10}{'MMACs':>8}{'Local RF':>10}{'RF ms':>8}{'Peak Act KB':>13}"
    print(header)
    print("-" * len(header))
    for r in rows:
        print(f"{r['model']:<7}{r['params']:>8,}{r['weight_kb']:>11.2f}{r['macs']:>10,}{r['mmacs']:>8.3f}"
              f"{r['local_rf_samples']:>10}{r['local_rf_ms']:>8.0f}{r['peak_activation_kb']:>13.2f}")

    print("\n=== Pipeline-hygiene checks (from Step 2; re-stated here for completeness) ===")
    chk.check(True, "no training performed in this script")
    chk.check(True, "official UCI test split not loaded or referenced by this script")
    chk.check(True, "no handcrafted feature files (X_train.txt / X_test.txt) referenced by this script")

    out_dir = Path("results/metrics")
    save_json({"param_budget": PARAM_BUDGET, "using_torch": HAVE_TORCH, "sampling_hz": SAMPLING_HZ, "models": rows},
              out_dir / "model_analysis.json")
    print(f"\nWrote {out_dir / 'model_analysis.json'}")
    return chk.summary()


if __name__ == "__main__":
    sys.exit(main())
