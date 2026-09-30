"""Algorithmic analysis of what a genuinely incremental (sample-by-sample) execution of Model D would
require in persistent state. This is NOT a claim of an actual streaming implementation, and NOT a claim
about X-HEEP or any other hardware -- see report/research_extension.md Sec. 11 for the precise wording
this module's numbers are meant to support.

Terminology used throughout (see report Sec. 11 for full definitions):
  - causal feature extractor:        the four depthwise/pointwise causal stages (this module's subject).
  - incremental-capable backbone:    the same stages, described from the "can update per new sample"
                                      angle -- same object, different emphasis.
  - windowed classifier:             GAP + Linear, which needs the whole window before deciding.

Per stage, a genuinely incremental depthwise-separable causal conv needs:
  - a ring buffer of the last `history_required = dilation*(kernel-1)` samples PER INPUT CHANNEL, so a
    new depthwise output can be produced the instant one new input sample arrives (push sample in, pop
    the oldest out, recompute the k-tap dot product for each channel -- O(kernel) work per new sample per
    channel, not a full re-convolution of the window).
  - the pointwise (kernel=1) conv needs NO history at all: it is a per-timestep linear map across
    channels, so it can be applied to a single new depthwise output the instant it exists.
  - BN (inference-mode, fixed running stats) and ReLU are also purely per-timestep, no state.
So the only state that accumulates across stages is the depthwise ring buffers; pointwise/BN/ReLU
contribute activation-sized (not history-sized) memory only for the CURRENT sample.
"""
from __future__ import annotations

from src.shape_calc import BYTES_PER_FLOAT32

from .architectures_ext import MODEL_D_STAGES


def per_stage_state(stages=MODEL_D_STAGES) -> list[dict]:
    rows = []
    for in_ch, out_ch, kernel, dilation in stages:
        history_required = dilation * (kernel - 1)      # ring-buffer LENGTH (in samples) per channel
        ring_buffer_scalars = history_required * in_ch    # one buffer per input channel to the depthwise conv
        rows.append({
            "in_ch": in_ch, "out_ch": out_ch, "kernel": kernel, "dilation": dilation,
            "history_required_samples": history_required,
            "depthwise_ring_buffer_scalars": ring_buffer_scalars,
            "pointwise_state_scalars": 0,          # per-timestep only, no history
            "current_activation_scalars": out_ch,   # this stage's current-timestep output, held briefly
        })
    return rows


def total_streaming_state(stages=MODEL_D_STAGES) -> dict:
    rows = per_stage_state(stages)
    ring_buffer_total = sum(r["depthwise_ring_buffer_scalars"] for r in rows)
    activation_total = sum(r["current_activation_scalars"] for r in rows)   # current-sample activations only
    total_scalars = ring_buffer_total + activation_total
    return {
        "per_stage": rows,
        "ring_buffer_scalars_total": ring_buffer_total,
        "current_activation_scalars_total": activation_total,
        "persistent_state_scalars_total": total_scalars,
        "persistent_state_fp32_bytes": total_scalars * BYTES_PER_FLOAT32,
        "persistent_state_int8_hypothetical_bytes": total_scalars * 1,   # HYPOTHETICAL, weights/activations only
        "note": ("Ring-buffer scalars are the ONLY state that must persist ACROSS incoming samples; "
                "current-activation scalars are transient per-sample working memory, not history. "
                "GAP + Linear (the windowed classifier) are excluded here: they operate on the whole "
                "window's worth of causal features, not on a per-sample incremental basis, so they are "
                "algorithmically a separate, non-streaming aggregation step -- see report Sec. 11."),
    }
