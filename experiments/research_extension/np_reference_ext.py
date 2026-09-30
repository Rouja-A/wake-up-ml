"""Pure-NumPy reference forward pass for the Linear baseline and Model D -- mirrors `src/np_reference.py`
so this runs (and is testable) with no torch installed. Also used for the causality unit test: it is
easier to reason about "does output[t] depend on input[>t]" against a hand-written direct convolution
loop than against an opaque nn.Conv1d call, and independence from the torch implementation makes
agreement between the two a real cross-check rather than a duplicated assumption (same rationale as
`src/np_reference.py`'s docstring).
"""
from __future__ import annotations

import numpy as np

from src.architectures import INPUT_CHANNELS, INPUT_LENGTH

from .architectures_ext import LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES, MODEL_D_STAGES


def causal_conv1d(x: np.ndarray, weight: np.ndarray, bias: np.ndarray, dilation: int, groups: int) -> np.ndarray:
    """x: (C_in, L). weight: (C_out, C_in/groups, K). LEFT-ONLY zero padding of dilation*(kernel-1); no
    right padding. Output has the same length L as the input, and output[:, t] is a function only of
    x[:, t], x[:, t-dilation], ..., x[:, t-(k-1)*dilation] -- i.e. never of x[:, >t]. This is the direct
    (non-im2col) loop the causality test in tests/test_research_extension.py exploits.
    """
    c_in, L = x.shape
    c_out, c_in_g, k = weight.shape
    left_pad = dilation * (k - 1)
    xp = np.pad(x, ((0, 0), (left_pad, 0)))   # zeros ONLY on the left
    out = np.empty((c_out, L), dtype=np.float64)
    g_in, g_out = c_in // groups, c_out // groups
    for g in range(groups):
        xg = xp[g * g_in:(g + 1) * g_in]                                    # (g_in, left_pad + L)
        wg = weight[g * g_out:(g + 1) * g_out]                              # (g_out, g_in, k)
        for t in range(L):
            # In the padded array, output position t reads padded indices [t, t + dilation, ..., t + (k-1)*dilation]
            taps = xg[:, t: t + (k - 1) * dilation + 1: dilation] if dilation > 1 else xg[:, t: t + k]
            out[g * g_out:(g + 1) * g_out, t] = np.tensordot(wg, taps, axes=([1, 2], [0, 1]))
    return out + bias[:, None]


class NPModelD:
    """Builds real random weights matching MODEL_D_STAGES and executes a real forward pass."""

    def __init__(self, seed: int = 0):
        rng = np.random.RandomState(seed)
        self.stage_params = []
        for in_ch, out_ch, kernel, dilation in MODEL_D_STAGES:
            dw_w = rng.randn(in_ch, 1, kernel) / np.sqrt(kernel)
            dw_b = np.zeros(in_ch)
            pw_w = rng.randn(out_ch, in_ch, 1) / np.sqrt(in_ch)
            pw_b = np.zeros(out_ch)
            bn = {"weight": np.ones(out_ch), "bias": np.zeros(out_ch),
                 "running_mean": np.zeros(out_ch), "running_var": np.ones(out_ch)}
            self.stage_params.append({"dw_w": dw_w, "dw_b": dw_b, "pw_w": pw_w, "pw_b": pw_b, "bn": bn,
                                      "in_ch": in_ch, "out_ch": out_ch, "kernel": kernel, "dilation": dilation})
        gap_ch = MODEL_D_STAGES[-1][1]
        self.linear_w = rng.randn(6, gap_ch) / np.sqrt(gap_ch)
        self.linear_b = np.zeros(6)

    def forward_features(self, x: np.ndarray) -> np.ndarray:
        """x: (C_in, L). Returns the (channels, L) causal feature map right before GAP (used by the
        causality test -- it inspects THIS tensor, not the final logits, to localise the check to the
        causal backbone as requested)."""
        for p in self.stage_params:
            x = causal_conv1d(x, p["dw_w"], p["dw_b"], dilation=p["dilation"], groups=p["in_ch"])
            x = causal_conv1d(x, p["pw_w"], p["pw_b"], dilation=1, groups=1)
            bn = p["bn"]
            x = (x - bn["running_mean"][:, None]) / np.sqrt(bn["running_var"][:, None] + 1e-5) * bn["weight"][:, None] + bn["bias"][:, None]
            x = np.maximum(x, 0)
        return x

    def forward(self, x: np.ndarray) -> np.ndarray:
        feats = self.forward_features(x)
        pooled = feats.mean(axis=1)   # GAP
        return self.linear_w @ pooled + self.linear_b


def run_linear(x_flat: np.ndarray, seed: int = 0) -> np.ndarray:
    """x_flat: (768,) already-flattened (6,128) input. Random Linear(768,6), matching LinearBaseline."""
    rng = np.random.RandomState(seed)
    w = rng.randn(LINEAR_OUT_FEATURES, LINEAR_IN_FEATURES) / np.sqrt(LINEAR_IN_FEATURES)
    b = np.zeros(LINEAR_OUT_FEATURES)
    return w @ x_flat + b
