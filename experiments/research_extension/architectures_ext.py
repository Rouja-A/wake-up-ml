"""Declarative specs for RQ1 (linear baseline) and RQ2 (Model D), designed ONCE from first principles
(no hyperparameter search), before seeing any validation result. See report/research_extension.md
Sec. 4-5 for the reasoning behind every choice below.

Both specs reuse `src.architectures` primitives where the semantics are identical (BNOp, ReLUOp, GAPOp)
and add `ops_ext.CausalConvOp` for the strictly-causal, dilated, depthwise/pointwise convolutions.
"""
from __future__ import annotations

from src.architectures import BNOp, GAPOp, INPUT_CHANNELS, INPUT_LENGTH, N_CLASSES, ReLUOp

from .ops_ext import CausalConvOp

# --------------------------------------------------------------------------------------------- RQ1
# Linear raw-window baseline: flatten (6, 128) -> (768,), then a single Linear(768, 6). No handcrafted
# features, no nonlinearity, no temporal structure at all -- the minimum possible reference point for
# "how much do the conv models gain from nonlinear temporal modeling on this input representation?"
LINEAR_IN_FEATURES = INPUT_CHANNELS * INPUT_LENGTH   # 6 * 128 = 768
LINEAR_OUT_FEATURES = N_CLASSES                       # 6
# Expected (verified against the built nn.Linear in tests/test_research_extension.py):
#   params = 768 * 6 + 6 = 4614           MACs = 768 * 6 = 4608

# --------------------------------------------------------------------------------------------- RQ2
# Model D -- Tiny Causal Dilated CNN.
#
# Four depthwise-separable causal stages, dilation doubling (1, 2, 4, 8), kernel 5, stride 1 throughout,
# channel width growing 6 -> 16 -> 24 -> 32 -> 48, then a windowed GAP + Linear classifier.
#
# Design reasoning (kept deliberately simple so it can be explained by hand -- see report Sec. 5):
#  - depthwise (per-channel) conv does the temporal mixing per channel; pointwise (kernel=1) conv mixes
#    channels. This is the same depthwise/pointwise split Model C already uses, so the two architectures
#    are comparable in "kind" of operation, differing in causality/dilation/pooling, not in operation type.
#  - dilation doubling (1, 2, 4, 8) grows the receptive field geometrically without adding parameters
#    (a dilated conv has exactly the same parameter count as a non-dilated one of the same kernel size).
#  - stride is kept at 1 in every layer (no MaxPool, unlike A/B/C). This is a deliberate simplicity choice:
#    every layer's ring buffer then has a FIXED, statically-known length (no down/up-sampling bookkeeping
#    between layers), which is what "conceptually suitable for incremental streaming" is meant to buy.
#    The explicit trade-off (documented, not hidden) is that MAC/window is therefore *not* reduced by
#    pooling the way A/B/C's is: see report Sec. 8.
#  - GAP at the end aggregates the (48, 128) causal feature map into (48,) for classification. This is a
#    WINDOWED classifier: the backbone is causal and incremental-capable, but this final aggregation step
#    still needs the whole window before it can produce a decision. See report Sec. 11 /
#    `streaming_state.py` for exactly what is and is not "streaming" about this.
MODEL_D_STAGES = (
    # (in_ch, out_ch, kernel, dilation)
    (INPUT_CHANNELS, 16, 5, 1),
    (16, 24, 5, 2),
    (24, 32, 5, 4),
    (32, 48, 5, 8),
)
MODEL_D_GAP_CHANNELS = MODEL_D_STAGES[-1][1]   # 48, input width to the final Linear


def build_model_d_spec() -> tuple:
    """Returns the flat Op sequence for Model D: for each stage, (depthwise CausalConvOp, pointwise
    CausalConvOp, BNOp, ReLUOp) -- matching Model C's convention of BN+ReLU only after the pointwise
    conv, not after the depthwise one -- then a trailing GAPOp. (LinearOp is added separately by
    `model_ext.py` / `shape_calc_ext.py` since it consumes the flattened GAP output, exactly as in
    `src.architectures`.)
    """
    ops: list = []
    for in_ch, out_ch, kernel, dilation in MODEL_D_STAGES:
        ops.append(CausalConvOp(in_ch, in_ch, kernel=kernel, dilation=dilation, groups=in_ch))   # depthwise
        ops.append(CausalConvOp(in_ch, out_ch, kernel=1, dilation=1, groups=1))                   # pointwise
        ops.append(BNOp(out_ch))
        ops.append(ReLUOp())
    ops.append(GAPOp())
    return tuple(ops)


MODEL_D_SPEC = build_model_d_spec()
MODEL_D_PARAM_BUDGET_SOFT = 10_000   # "preferably under ~10,000" from the brief
MODEL_D_PARAM_BUDGET_HARD = 20_000   # "fewer than 20,000" -- enforced by a test

# --------------------------------------------------------------------------------------------- RQ4
# Sensor ablation uses Model A's architecture (chosen BEFORE seeing any ablation result -- see report
# Sec. 9 for why: A is the cheapest of the three frozen architectures and its Phase-1 mean validation
# macro-F1 (0.9890) is within noise of C's (0.9898), so it isolates "does the INPUT change the answer"
# from "is the architecture strong enough to use the input", without re-running a model search).
# Only the first conv's in_ch varies (3 for single-modality arms, 6 for the full/body_acc_gyro arms);
# everything else is IDENTICAL to `src.architectures.MODEL_A`.
from src.architectures import ConvOp   # noqa: E402


def model_a_variant_spec(in_channels: int) -> tuple:
    from src.architectures import MODEL_A
    first = MODEL_A[0]
    if not isinstance(first, ConvOp):
        raise TypeError("MODEL_A[0] is expected to be the first ConvOp")
    return (ConvOp(in_channels, first.out_ch, kernel=first.kernel, stride=first.stride, groups=first.groups),
           *MODEL_A[1:])
