"""Tensor-free (no torch) analytic computation of params / MACs / receptive field for the extension
models, mirroring `src.shape_calc.analyze` exactly in spirit: every number here is derived purely from
hyperparameters, so it is exact and reproducible with plain Python, and is the reference that
`model_ext.py` (real torch nn.Module, when torch is available) is cross-checked against in
tests/test_research_extension.py -- the same "formula vs. real module" cross-check pattern as
`src/complexity.py::cross_check_against_spec`.
"""
from __future__ import annotations

from src.architectures import BNOp, GAPOp, INPUT_CHANNELS, INPUT_LENGTH, ReLUOp
from src.shape_calc import BYTES_PER_FLOAT32, LayerReport, ModelReport

from .architectures_ext import LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES, MODEL_D_SPEC
from .ops_ext import CausalConvOp


def causal_conv_out_length(L: int, kernel: int, dilation: int, stride: int = 1) -> int:
    """Output length of a causal conv: left-pad by dilation*(kernel-1), no right padding, stride 1
    => output length is always exactly L (never shrinks, never looks into the future)."""
    if stride != 1:
        raise NotImplementedError("Model D uses stride 1 throughout; see ops_ext.CausalConvOp.")
    # left_pad (= dilation*(kernel-1)) exactly cancels the standard conv output-length formula's
    # "- dilation*(kernel-1)" term, so a causal conv at stride 1 always preserves length: L_out == L_in.
    return L


def analyze_model_d(input_channels: int = INPUT_CHANNELS, input_length: int = INPUT_LENGTH) -> ModelReport:
    report = ModelReport(name="D")
    ch, L = input_channels, input_length
    r, j = 1, 1
    layer_histories: list[int] = []   # per-causal-conv `history_required`, in execution order

    for i, op in enumerate(MODEL_D_SPEC):
        if isinstance(op, CausalConvOp):
            if op.in_ch != ch:
                raise ValueError(f"layer {i}: spec says in_ch={op.in_ch} but previous output has {ch} channels")
            L = causal_conv_out_length(L, op.kernel, op.dilation, op.stride)
            params = (ch // op.groups) * op.out_ch * op.kernel + op.out_ch
            macs = L * op.out_ch * op.kernel * (ch // op.groups)
            ch = op.out_ch
            r, j = r + (op.kernel - 1) * op.dilation * j, j * op.stride
            layer_histories.append(op.history_required)
            kind = f"CausalConv1d(g={op.groups},d={op.dilation})" if op.groups > 1 or op.dilation > 1 else "CausalConv1d"
            report.layers.append(LayerReport(i, kind, (ch, L), params, macs))
        elif isinstance(op, BNOp):
            report.layers.append(LayerReport(i, "BatchNorm1d", (ch, L), 2 * ch, 0))
        elif isinstance(op, ReLUOp):
            report.layers.append(LayerReport(i, "ReLU", (ch, L), 0, 0))
        elif isinstance(op, GAPOp):
            report.gap_input_length = L
            report.post_gap_coverage_raw = r + (L - 1) * j
            report.post_gap_coverage = min(report.post_gap_coverage_raw, input_length)
            report.layers.append(LayerReport(i, "AdaptiveAvgPool1d", (ch, 1), 0, 0,
                                             note=f"WINDOWED classifier aggregation over all {L} causal "
                                                  f"features; the backbone up to here is causal/incremental "
                                                  f"-capable, this step is not (see streaming_state.py)"))
            L = 1
        else:
            raise TypeError(f"Unknown op at layer {i}: {op!r}")

    # Trailing Linear(GAP_channels, N_CLASSES), same convention as src.shape_calc.analyze
    linear_in, linear_out = ch, 6
    linear_params = linear_in * linear_out + linear_out
    linear_macs = linear_in * linear_out
    report.layers.append(LayerReport(len(MODEL_D_SPEC), "Linear", (linear_out,), linear_params, linear_macs))

    report.receptive_field, report.jump = r, j
    report.causal_history_per_layer = layer_histories   # extra attribute; ModelReport allows arbitrary attrs
    return report


def analyze_linear() -> ModelReport:
    """Linear baseline: flatten (6,128)->(768,) then Linear(768,6). No conv, no receptive field concept
    beyond "the whole window" (RF = window length, by construction, since every input sample is read)."""
    report = ModelReport(name="Linear")
    params = LINEAR_IN_FEATURES * LINEAR_OUT_FEATURES + LINEAR_OUT_FEATURES
    macs = LINEAR_IN_FEATURES * LINEAR_OUT_FEATURES
    report.layers.append(LayerReport(0, "Flatten", (LINEAR_IN_FEATURES,), 0, 0))
    report.layers.append(LayerReport(1, "Linear", (LINEAR_OUT_FEATURES,), params, macs))
    report.receptive_field = INPUT_LENGTH   # reads every sample of every channel directly
    report.jump = 1
    report.gap_input_length = 0
    report.post_gap_coverage = INPUT_LENGTH
    return report


def weight_storage_bytes(report: ModelReport) -> dict:
    fp32 = report.total_params * BYTES_PER_FLOAT32
    int8_hypothetical = report.total_params * 1   # HYPOTHETICAL: weights-only, no quantization error modeled
    return {"fp32_bytes": fp32, "fp32_kib": fp32 / 1024,
           "int8_hypothetical_bytes": int8_hypothetical, "int8_hypothetical_kib": int8_hypothetical / 1024}
