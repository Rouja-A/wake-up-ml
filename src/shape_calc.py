"""Analytic (tensor-free) computation of params / MACs / receptive field / activation sizes.

Every quantity here is derived purely from the Op list's hyperparameters (channels, kernel, stride,
padding, groups) using the standard formulas below -- no learned values are involved, so these numbers
are exact, not estimates, regardless of whether a real tensor is ever pushed through the network. This
module has NO torch dependency, so it runs in any Python + NumPy environment.

It plays two roles:
  1. `analyze(spec)` is the single formula-based implementation both `src/complexity.py` (hook-based,
     needs torch) and `src/np_reference.py` (executes real NumPy arithmetic) are checked against.
  2. Where torch is unavailable, `scripts/analyze_models.py` falls back to this module directly.

Formulas used (standard, e.g. Araujo et al. 2019 "Computing Receptive Fields"; Sifre & Mallat 2014 for
depthwise-separable MAC counting):
  conv/pool output length : L_out = floor((L_in + 2*padding - dilation*(kernel-1) - 1) / stride) + 1
  conv params              : (in_channels // groups) * out_channels * kernel + out_channels (bias)
  conv MACs                : out_length * out_channels * kernel * (in_channels // groups)
  linear params / MACs     : in*out + out   /   in*out
  batchnorm params         : 2 * channels  (trainable weight+bias only; running mean/var are buffers)
  receptive field (r) and jump (j), for a conv/pool layer with kernel k, stride s, dilation d=1,
  starting at r=1, j=1 before the first layer:
      r <- r + (k - 1) * d * j
      j <- j * s
  RF is accumulated only through Conv/Pool layers (a 1x1 pointwise conv contributes (k-1)=0, correctly).
  GAP and Linear do not extend this "per-output-pixel" receptive field: GAP is a separate, explicit
  aggregation over the whole remaining feature map, not part of the sliding-window receptive field that
  matters for streaming latency -- see the "streaming relevance" discussion later in the project.
"""
from __future__ import annotations

from dataclasses import dataclass, field

from .architectures import BNOp, ConvOp, GAPOp, INPUT_CHANNELS, INPUT_LENGTH, LinearOp, Op, PoolOp, ReLUOp

BYTES_PER_FLOAT32 = 4


def conv_or_pool_out_length(L: int, kernel: int, stride: int, padding: int = 0, dilation: int = 1) -> int:
    return (L + 2 * padding - dilation * (kernel - 1) - 1) // stride + 1


@dataclass
class LayerReport:
    index: int
    op_type: str
    out_shape: tuple[int, ...]     # (channels, length) or (features,) after flatten/linear
    params: int
    macs: int                      # 0 for BN/ReLU/Pool/GAP (see module docstring convention)
    note: str = ""

    @property
    def elements(self) -> int:
        n = 1
        for d in self.out_shape:
            n *= d
        return n

    @property
    def kb(self) -> float:
        return self.elements * BYTES_PER_FLOAT32 / 1024


@dataclass
class ModelReport:
    name: str
    layers: list[LayerReport] = field(default_factory=list)
    receptive_field: int = 1        # LOCAL backbone RF: context of one pre-GAP feature (samples)
    jump: int = 1                   # stride, in raw samples, between consecutive pre-GAP features
    gap_input_length: int = 1       # temporal length GAP averages over (0 if the model has no GAP)
    post_gap_coverage_raw: int = 1  # local_RF + (gap_input_length - 1) * jump; can exceed input_length
    post_gap_coverage: int = 1      # above, capped at the input window length (see analyze() docstring)

    @property
    def total_params(self) -> int:
        return sum(l.params for l in self.layers)

    @property
    def total_macs(self) -> int:
        return sum(l.macs for l in self.layers)

    @property
    def weight_kb(self) -> float:
        return self.total_params * BYTES_PER_FLOAT32 / 1024

    @property
    def peak_activation(self) -> LayerReport:
        return max(self.layers, key=lambda l: l.elements)


def analyze(spec: tuple[Op, ...], name: str = "", input_channels: int = INPUT_CHANNELS,
           input_length: int = INPUT_LENGTH) -> ModelReport:
    """Walk `spec` once, computing shapes/params/MACs/RF/jump purely from hyperparameters.

    Two receptive-field numbers are reported (see the module docstring for the recurrence):
      * `receptive_field`/`jump` ("local backbone RF"): the temporal context of ONE feature at the last
        conv/pool layer, before GAP. This is the number relevant to a streaming/incremental
        implementation, since it is the span a sliding buffer would need for one output.
      * `post_gap_coverage`: GAP averages over ALL `gap_input_length` remaining positions, so its
        effective input span is the union of their local receptive fields:
            raw = local_RF + (gap_input_length - 1) * jump
        `raw` can exceed the actual input window once padding lets edge features "see" positions beyond
        [0, input_length) (they are simply clipped to zero there), so `post_gap_coverage` is `raw`
        capped at `input_length` -- the true value can't exceed the window that was actually fed in.
    """
    report = ModelReport(name=name)
    ch, L = input_channels, input_length
    r, j = 1, 1
    flattened = False

    for i, op in enumerate(spec):
        if isinstance(op, ConvOp):
            if op.in_ch != ch:
                raise ValueError(f"layer {i}: spec says in_ch={op.in_ch} but previous output has {ch} channels")
            L = conv_or_pool_out_length(L, op.kernel, op.stride, op.padding)
            params = (ch // op.groups) * op.out_ch * op.kernel + op.out_ch
            macs = L * op.out_ch * op.kernel * (ch // op.groups)
            ch = op.out_ch
            r, j = r + (op.kernel - 1) * j, j * op.stride
            report.layers.append(LayerReport(i, f"Conv1d(g={op.groups})" if op.groups > 1 else "Conv1d",
                                             (ch, L), params, macs))
        elif isinstance(op, BNOp):
            if op.channels != ch:
                raise ValueError(f"layer {i}: BN channels={op.channels} != current channels={ch}")
            report.layers.append(LayerReport(i, "BatchNorm1d", (ch, L), 2 * ch, 0))
        elif isinstance(op, ReLUOp):
            report.layers.append(LayerReport(i, "ReLU", (ch, L), 0, 0))
        elif isinstance(op, PoolOp):
            L = conv_or_pool_out_length(L, op.kernel, op.stride, padding=0)
            r, j = r + (op.kernel - 1) * j, j * op.stride
            report.layers.append(LayerReport(i, "MaxPool1d", (ch, L), 0, 0))
        elif isinstance(op, GAPOp):
            report.gap_input_length = L
            report.post_gap_coverage_raw = r + (L - 1) * j
            report.post_gap_coverage = min(report.post_gap_coverage_raw, input_length)
            report.layers.append(LayerReport(i, "AdaptiveAvgPool1d", (ch, 1), 0, 0,
                                             note=f"aggregates over the full remaining length ({L}); "
                                                  f"not part of the sliding-window receptive field"))
            L = 1
        elif isinstance(op, LinearOp):
            if not flattened:
                ch = ch * L                       # implicit flatten of the (channels, 1) GAP output
                flattened = True
            if op.in_features != ch:
                raise ValueError(f"layer {i}: Linear in_features={op.in_features} != flattened size {ch}")
            params = op.in_features * op.out_features + op.out_features
            macs = op.in_features * op.out_features
            report.layers.append(LayerReport(i, "Linear", (op.out_features,), params, macs))
            ch = op.out_features
        else:
            raise TypeError(f"Unknown op at layer {i}: {op!r}")

    report.receptive_field, report.jump = r, j
    return report


def assert_within_budget(report: ModelReport, budget: int) -> None:
    assert report.total_params < budget, (
        f"Model '{report.name}' has {report.total_params} trainable parameters, "
        f">= the {budget}-parameter budget."
    )
