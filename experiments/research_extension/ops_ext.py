"""Extra Op type for the research extension: a strictly-causal, optionally-dilated conv.

`src.architectures.ConvOp` always uses symmetric 'same' padding = (kernel-1)//2 on BOTH sides. That is
NOT causal: for a stride-1 conv, (kernel-1)//2 of those padded/real positions are on the FUTURE side of
the output timestep, so the output at time t would depend on inputs at t+1, t+2, ... To make a temporal
conv causal, all of the padding must go on the LEFT (past) side, and the amount must equal the full
receptive-field extension `dilation * (kernel - 1)`, not half of it -- so history, not future, is what
gets padded when the real input runs out at the start of a stream.

For a causal layer with dilation d and kernel k, an output at time t needs inputs at
t, t-d, t-2d, ..., t-(k-1)*d, i.e. `d*(k-1)` samples of PAST history in addition to the current sample.
Left-padding by exactly that amount reproduces, at train/eval time on a full window, exactly what an
incremental/streaming implementation would compute one sample at a time (see `streaming_state.py`).
"""
from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class CausalConvOp:
    in_ch: int
    out_ch: int
    kernel: int
    dilation: int = 1
    groups: int = 1
    stride: int = 1   # kept at 1 throughout Model D; see report/research_extension.md Sec. 5 for why

    def __post_init__(self):
        if self.in_ch % self.groups or self.out_ch % self.groups:
            raise ValueError(f"in_ch/out_ch must be divisible by groups: {self}")
        if self.stride != 1:
            raise NotImplementedError(
                "CausalConvOp with stride != 1 is not used by Model D (see design notes); the causal "
                "left-padding formula below assumes stride 1, where output length == input length."
            )

    @property
    def left_padding(self) -> int:
        """Samples of PAST history a causal conv needs: dilation * (kernel - 1). Padding only this many
        zeros on the LEFT (and none on the right) keeps output length == input length while guaranteeing
        output[t] never reads input[> t]."""
        return self.dilation * (self.kernel - 1)

    @property
    def history_required(self) -> int:
        """Alias of left_padding, named for the streaming-state discussion (ring-buffer length needed)."""
        return self.left_padding
