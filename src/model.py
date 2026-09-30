"""PyTorch models, built by interpreting the Op lists in src/architectures.py.

Building the network by interpreting the spec (rather than writing three separate hand-coded classes)
guarantees the executed graph matches the documented architecture exactly, and that every downstream
measurement (src/complexity.py) reads it from the real nn.Module via named_parameters()/hooks, never
from a hand-typed number.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .architectures import (ARCHITECTURES, INPUT_CHANNELS, INPUT_LENGTH, N_CLASSES,
                            BNOp, ConvOp, GAPOp, LinearOp, Op, PoolOp, ReLUOp)


def _build_layer(op: Op) -> nn.Module:
    if isinstance(op, ConvOp):
        return nn.Conv1d(op.in_ch, op.out_ch, op.kernel, stride=op.stride, padding=op.padding,
                         groups=op.groups, bias=True)
    if isinstance(op, BNOp):
        return nn.BatchNorm1d(op.channels)
    if isinstance(op, ReLUOp):
        return nn.ReLU(inplace=True)
    if isinstance(op, PoolOp):
        return nn.MaxPool1d(kernel_size=op.kernel, stride=op.stride)
    if isinstance(op, GAPOp):
        return nn.AdaptiveAvgPool1d(1)
    if isinstance(op, LinearOp):
        return nn.Linear(op.in_features, op.out_features)
    raise TypeError(f"Unknown op: {op!r}")


class TinyCNN(nn.Module):
    """Sequential network interpreting an Op list. `flatten` is inserted automatically after the GAP op
    (AdaptiveAvgPool1d(1) leaves a trailing length-1 axis that must be dropped before the Linear layer).
    """

    def __init__(self, name: str):
        super().__init__()
        if name not in ARCHITECTURES:
            raise ValueError(f"Unknown model name '{name}'; choose from {list(ARCHITECTURES)}")
        self.name = name
        spec = ARCHITECTURES[name]
        layers: list[nn.Module] = []
        for op in spec:
            layers.append(_build_layer(op))
            if isinstance(op, GAPOp):
                layers.append(nn.Flatten(1))
        self.layers = nn.ModuleList(layers)
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"expected input (B, {INPUT_CHANNELS}, T), got {tuple(x.shape)}")
        for layer in self.layers:
            x = layer(x)
        return x


def build_model(name: str) -> TinyCNN:
    return TinyCNN(name)


def dummy_input(batch_size: int = 1, length: int = INPUT_LENGTH, device=None) -> torch.Tensor:
    return torch.randn(batch_size, INPUT_CHANNELS, length, device=device)


if __name__ == "__main__":
    for name in ARCHITECTURES:
        m = build_model(name)
        out = m(dummy_input())
        print(name, "output:", tuple(out.shape))
        assert tuple(out.shape) == (1, N_CLASSES)
