"""Declarative architecture specs for Models A/B/C.

This is the single source of truth for the three candidates. `src/model.py` (PyTorch) and
`src/np_reference.py` (pure NumPy, used only where torch is unavailable) both BUILD their networks by
interpreting the same `Op` list, so the two implementations cannot silently diverge, and no parameter
count / MAC / receptive-field number anywhere in the project is typed in by hand: everything downstream
is derived from these lists.

Convention: stride-1 convolutions use "same" padding, i.e. padding = (kernel - 1) // 2 (exact for odd
kernels with dilation=1, which is all that is used here) so the sequence length is unchanged by a conv
and only changes at MaxPool layers.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Union


@dataclass(frozen=True)
class ConvOp:
    in_ch: int
    out_ch: int
    kernel: int
    stride: int = 1
    groups: int = 1

    def __post_init__(self):
        if self.in_ch % self.groups or self.out_ch % self.groups:
            raise ValueError(f"in_ch/out_ch must be divisible by groups: {self}")

    @property
    def padding(self) -> int:
        return (self.kernel - 1) // 2  # 'same' padding, stride 1, dilation 1


@dataclass(frozen=True)
class BNOp:
    channels: int


@dataclass(frozen=True)
class ReLUOp:
    pass


@dataclass(frozen=True)
class PoolOp:
    kernel: int
    stride: int


@dataclass(frozen=True)
class GAPOp:
    pass


@dataclass(frozen=True)
class LinearOp:
    in_features: int
    out_features: int


Op = Union[ConvOp, BNOp, ReLUOp, PoolOp, GAPOp, LinearOp]

INPUT_CHANNELS = 6
INPUT_LENGTH = 128
N_CLASSES = 6
PARAM_BUDGET = 20_000

# ----------------------------------------------------------------------------------------- specs
MODEL_A: tuple[Op, ...] = (
    ConvOp(6, 16, kernel=9), BNOp(16), ReLUOp(), PoolOp(2, 2),
    ConvOp(16, 32, kernel=5), BNOp(32), ReLUOp(),
    GAPOp(), LinearOp(32, N_CLASSES),
)

MODEL_B: tuple[Op, ...] = (
    ConvOp(6, 16, kernel=9), BNOp(16), ReLUOp(), PoolOp(2, 2),
    ConvOp(16, 32, kernel=5), BNOp(32), ReLUOp(), PoolOp(2, 2),
    ConvOp(32, 64, kernel=5), BNOp(64), ReLUOp(),
    GAPOp(), LinearOp(64, N_CLASSES),
)

MODEL_C: tuple[Op, ...] = (
    ConvOp(6, 24, kernel=9), BNOp(24), ReLUOp(), PoolOp(2, 2),
    ConvOp(24, 24, kernel=9, groups=24), ConvOp(24, 48, kernel=1),        # depthwise, pointwise
    BNOp(48), ReLUOp(), PoolOp(2, 2),
    ConvOp(48, 48, kernel=9, groups=48), ConvOp(48, 96, kernel=1),        # depthwise, pointwise
    BNOp(96), ReLUOp(),
    GAPOp(), LinearOp(96, N_CLASSES),
)

ARCHITECTURES: dict[str, tuple[Op, ...]] = {"A": MODEL_A, "B": MODEL_B, "C": MODEL_C}
ARCHITECTURE_NAMES = {"A": "Minimal (2 conv layers)", "B": "3-conv CNN (main candidate)",
                      "C": "Depthwise-separable (3 stages)"}
