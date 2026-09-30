"""Measure params / MACs / receptive field / activation memory from the REAL PyTorch model.

This is the authoritative measurement (what the user's environment, with torch + CUDA, should run):
parameter counts come from `model.named_parameters()`, and MACs/activation sizes come from the actual
tensor shapes seen during a real forward pass (via forward hooks), not from a formula assumed in advance.
`src/shape_calc.py` provides an independent, tensor-free cross-check that these hook-derived numbers are
expected to equal; `cross_check_against_spec` asserts the two agree.
"""
from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn as nn

from .shape_calc import BYTES_PER_FLOAT32, ModelReport, analyze


@dataclass
class ParamBreakdown:
    total: int
    trainable: int
    non_trainable: int
    by_layer: list[tuple[str, str, int]]   # (layer name, module type, trainable params)


def count_parameters(model: nn.Module) -> ParamBreakdown:
    total = sum(p.numel() for p in model.parameters())
    trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    by_layer = []
    for name, module in model.named_modules():
        if list(module.children()):     # only leaf modules, to avoid double-counting
            continue
        n = sum(p.numel() for p in module.parameters(recurse=False) if p.requires_grad)
        if n:
            by_layer.append((name or module.__class__.__name__, module.__class__.__name__, n))
    return ParamBreakdown(total, trainable, total - trainable, by_layer)


def weight_memory_kb(model: nn.Module) -> float:
    return sum(p.numel() for p in model.parameters() if p.requires_grad) * BYTES_PER_FLOAT32 / 1024


def profile_forward(model: nn.Module, input_shape=(1, 6, 128)) -> list[dict]:
    """Run one dummy forward pass, recording (name, type, output shape) for every leaf module IN THE
    ORDER THEY EXECUTE (needed for the receptive-field trace, which depends on execution order)."""
    trace: list[dict] = []
    handles = []

    def hook(name):
        def fn(module, inputs, output):
            trace.append({"name": name, "type": module.__class__.__name__, "module": module,
                         "out_shape": tuple(output.shape)})
        return fn

    for name, module in model.named_modules():
        if not list(module.children()):
            handles.append(module.register_forward_hook(hook(name)))
    model.eval()
    with torch.no_grad():
        out = model(torch.randn(*input_shape))
    for h in handles:
        h.remove()
    return trace


def measure(model: nn.Module, input_shape=(1, 6, 128)) -> ModelReport:
    """Hook-based measurement in the same ModelReport shape `shape_calc.analyze` returns, so the two
    are directly comparable. MACs use the REAL output length seen at each Conv1d/Linear during the
    forward pass; receptive field/jump are computed from the REAL execution-order trace.
    """
    from .shape_calc import LayerReport

    trace = profile_forward(model, input_shape)
    report = ModelReport(name=getattr(model, "name", model.__class__.__name__))
    r, j = 1, 1
    input_length = input_shape[-1]
    for i, rec in enumerate(trace):
        m, shape = rec["module"], rec["out_shape"]
        params = sum(p.numel() for p in m.parameters(recurse=False) if p.requires_grad)
        macs = 0
        if isinstance(m, nn.Conv1d):
            out_len = shape[-1]
            macs = out_len * m.out_channels * m.kernel_size[0] * (m.in_channels // m.groups)
            r, j = r + (m.kernel_size[0] - 1) * j, j * m.stride[0]
        elif isinstance(m, nn.Linear):
            macs = m.in_features * m.out_features
        elif isinstance(m, nn.MaxPool1d):
            k = m.kernel_size if isinstance(m.kernel_size, int) else m.kernel_size[0]
            s = m.stride if isinstance(m.stride, int) else m.stride[0]
            r, j = r + (k - 1) * j, j * s
        elif isinstance(m, nn.AdaptiveAvgPool1d):
            # `trace` is execution order, so the previous entry is the feature map GAP just consumed.
            gap_in_len = trace[i - 1]["out_shape"][-1]
            report.gap_input_length = gap_in_len
            report.post_gap_coverage_raw = r + (gap_in_len - 1) * j
            report.post_gap_coverage = min(report.post_gap_coverage_raw, input_length)
        out_shape = shape[1:]  # drop batch dim
        report.layers.append(LayerReport(i, m.__class__.__name__, out_shape, params, macs))
    report.receptive_field, report.jump = r, j
    return report


def cross_check_against_spec(hook_report: ModelReport, spec) -> None:
    """Assert the hook-based (real-tensor) measurement agrees with the tensor-free formula (shape_calc).
    A mismatch would mean the built model doesn't actually match its documented spec."""
    ref = analyze(spec, name=hook_report.name)
    assert hook_report.total_params == ref.total_params, (hook_report.total_params, ref.total_params)
    assert hook_report.total_macs == ref.total_macs, (hook_report.total_macs, ref.total_macs)
    assert hook_report.receptive_field == ref.receptive_field, (hook_report.receptive_field, ref.receptive_field)
    assert hook_report.jump == ref.jump, (hook_report.jump, ref.jump)
    assert hook_report.post_gap_coverage == ref.post_gap_coverage, (hook_report.post_gap_coverage, ref.post_gap_coverage)
    assert hook_report.peak_activation.elements == ref.peak_activation.elements
