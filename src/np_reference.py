"""Pure-NumPy reference implementation of the same architectures (no torch dependency).

Used for two purposes:
  1. Sandbox/CI environments without torch installed can still run a REAL forward pass (real
     convolution/pooling/batchnorm arithmetic on real arrays, not just formulas) to check shapes and
     finiteness end to end.
  2. It is an implementation genuinely independent of src/model.py + src/complexity.py, so agreement
     between the two is real evidence, not a duplicated assumption.

Weights are initialised randomly (not trained); BatchNorm uses the same identity-at-init behaviour
PyTorch has before any training (running_mean=0, running_var=1, weight=1, bias=0), so this checks
structure/shapes/finiteness, not learned behaviour.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from .architectures import BNOp, ConvOp, GAPOp, INPUT_CHANNELS, INPUT_LENGTH, LinearOp, Op, PoolOp, ReLUOp
from .shape_calc import BYTES_PER_FLOAT32, LayerReport, ModelReport, conv_or_pool_out_length


def _conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, stride: int, padding: int, groups: int) -> np.ndarray:
    """x: (C_in, L). weight: (C_out, C_in/groups, K). Direct (non-im2col) loop; input is tiny (<=~128), so
    this is fast enough and easy to verify by eye against the formulas in shape_calc.py."""
    c_in, L = x.shape
    c_out, c_in_g, k = weight.shape
    xp = np.pad(x, ((0, 0), (padding, padding)))
    L_out = conv_or_pool_out_length(L, k, stride, padding)
    out = np.empty((c_out, L_out), dtype=np.float64)
    g_in, g_out = c_in // groups, c_out // groups
    for g in range(groups):
        xg = xp[g * g_in:(g + 1) * g_in]                     # (g_in, L+2p)
        for t in range(L_out):
            window = xg[:, t * stride: t * stride + k]        # (g_in, k)
            out[g * g_out:(g + 1) * g_out, t] = np.tensordot(weight[g * g_out:(g + 1) * g_out], window, axes=([1, 2], [0, 1]))
    return out + bias[:, None]


def _maxpool1d(x: np.ndarray, kernel: int, stride: int) -> np.ndarray:
    C, L = x.shape
    L_out = conv_or_pool_out_length(L, kernel, stride, padding=0)
    out = np.empty((C, L_out), dtype=x.dtype)
    for t in range(L_out):
        out[:, t] = x[:, t * stride: t * stride + kernel].max(axis=1)
    return out


class NPModel:
    """Builds real random weights from `spec` and executes a real forward pass."""

    def __init__(self, spec: tuple[Op, ...], name: str = "", seed: int = 0):
        self.spec, self.name = spec, name
        rng = np.random.RandomState(seed)
        self.params: list[dict] = []
        ch = INPUT_CHANNELS
        for op in spec:
            if isinstance(op, ConvOp):
                fan_in = (ch // op.groups) * op.kernel
                w = rng.randn(op.out_ch, ch // op.groups, op.kernel) / np.sqrt(fan_in)   # Kaiming-ish, just needs to be finite/reasonable
                b = np.zeros(op.out_ch)
                self.params.append({"weight": w, "bias": b})
                ch = op.out_ch
            elif isinstance(op, BNOp):
                self.params.append({"weight": np.ones(op.channels), "bias": np.zeros(op.channels),
                                    "running_mean": np.zeros(op.channels), "running_var": np.ones(op.channels)})
            elif isinstance(op, LinearOp):
                w = rng.randn(op.out_features, op.in_features) / np.sqrt(op.in_features)
                b = np.zeros(op.out_features)
                self.params.append({"weight": w, "bias": b})
            else:
                self.params.append(None)

    def forward_with_trace(self, x: np.ndarray) -> tuple[np.ndarray, ModelReport]:
        """x: (C_in, L). Returns (logits, ModelReport) computed from the REAL executed shapes."""
        report = ModelReport(name=self.name)
        r, j = 1, 1
        flattened = False
        x_input_length = x.shape[-1]
        for i, (op, p) in enumerate(zip(self.spec, self.params)):
            if isinstance(op, ConvOp):
                x = _conv1d(x, p["weight"], p["bias"], op.stride, op.padding, op.groups)
                macs = x.shape[-1] * op.out_ch * op.kernel * (op.in_ch // op.groups)
                params = p["weight"].size + p["bias"].size
                r, j = r + (op.kernel - 1) * j, j * op.stride
                report.layers.append(LayerReport(i, "Conv1d", x.shape, params, macs))
            elif isinstance(op, BNOp):
                eps = 1e-5
                x = (x - p["running_mean"][:, None]) / np.sqrt(p["running_var"][:, None] + eps) * p["weight"][:, None] + p["bias"][:, None]
                report.layers.append(LayerReport(i, "BatchNorm1d", x.shape, p["weight"].size + p["bias"].size, 0))
            elif isinstance(op, ReLUOp):
                x = np.maximum(x, 0)
                report.layers.append(LayerReport(i, "ReLU", x.shape, 0, 0))
            elif isinstance(op, PoolOp):
                x = _maxpool1d(x, op.kernel, op.stride)
                r, j = r + (op.kernel - 1) * j, j * op.stride
                report.layers.append(LayerReport(i, "MaxPool1d", x.shape, 0, 0))
            elif isinstance(op, GAPOp):
                gap_in_len = x.shape[-1]
                report.gap_input_length = gap_in_len
                report.post_gap_coverage_raw = r + (gap_in_len - 1) * j
                report.post_gap_coverage = min(report.post_gap_coverage_raw, x_input_length)
                x = x.mean(axis=1, keepdims=True)
                report.layers.append(LayerReport(i, "AdaptiveAvgPool1d", x.shape, 0, 0))
            elif isinstance(op, LinearOp):
                if not flattened:
                    x = x.reshape(-1)
                    flattened = True
                x = p["weight"] @ x + p["bias"]
                report.layers.append(LayerReport(i, "Linear", x.shape, p["weight"].size + p["bias"].size, p["weight"].size))
            else:
                raise TypeError(op)
        report.receptive_field, report.jump = r, j
        return x, report


def run(name: str, spec: tuple[Op, ...], seed: int = 0) -> tuple[np.ndarray, ModelReport]:
    model = NPModel(spec, name=name, seed=seed)
    x = np.random.RandomState(seed + 1).randn(INPUT_CHANNELS, INPUT_LENGTH)
    return model.forward_with_trace(x)
