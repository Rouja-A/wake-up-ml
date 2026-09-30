"""Real PyTorch modules for the linear baseline (RQ1) and Model D (RQ2).

Guarded import, exactly like `src/model.py` vs `src/np_reference.py`: this module requires torch and is
only imported where torch is available (tests/test_research_extension.py checks `HAVE_TORCH` first, the
same pattern tests/test_model.py already uses). `np_reference_ext.py` is the torch-free equivalent, used
for the causality unit test and for shape/param sanity checks in a torch-less sandbox.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from src.architectures import INPUT_CHANNELS, INPUT_LENGTH, N_CLASSES

from .architectures_ext import (LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES, MODEL_D_GAP_CHANNELS,
                                MODEL_D_STAGES)


class LinearBaseline(nn.Module):
    """Flatten (B, 6, 128) -> (B, 768), then a single Linear(768, 6). No handcrafted features."""

    def __init__(self):
        super().__init__()
        self.name = "Linear"
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(LINEAR_IN_FEATURES, LINEAR_OUT_FEATURES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != INPUT_CHANNELS or x.shape[2] != INPUT_LENGTH:
            raise ValueError(f"expected input (B, {INPUT_CHANNELS}, {INPUT_LENGTH}), got {tuple(x.shape)}")
        return self.linear(self.flatten(x))


class CausalConv1d(nn.Module):
    """A Conv1d with LEFT-ONLY padding of `dilation*(kernel-1)` zeros (no right/future padding), so
    output[..., t] depends only on input[..., <= t]. `nn.Conv1d`'s own `padding=` argument always pads
    symmetrically, which is why this wraps an explicit `nn.ConstantPad1d((left, 0), 0.0)` in front of a
    `padding=0` conv rather than using `padding=(kernel-1)//2` (that would be symmetric = non-causal,
    and would additionally be WRONG for dilation>1, since (kernel-1)//2 ignores dilation entirely).
    """

    def __init__(self, in_ch: int, out_ch: int, kernel: int, dilation: int = 1, groups: int = 1):
        super().__init__()
        left_pad = dilation * (kernel - 1)
        self.pad = nn.ConstantPad1d((left_pad, 0), 0.0)
        self.conv = nn.Conv1d(in_ch, out_ch, kernel, stride=1, padding=0, dilation=dilation,
                              groups=groups, bias=True)
        self.history_required = left_pad

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(self.pad(x))


class TinyCausalCNN(nn.Module):
    """Model D: four causal depthwise-separable stages (dilation 1,2,4,8; kernel 5; stride 1), then a
    windowed GAP + Linear classifier. See architectures_ext.py for the full design rationale and
    report/research_extension.md Sec. 5 for the streaming discussion."""

    def __init__(self):
        super().__init__()
        self.name = "D"
        blocks = []
        for in_ch, out_ch, kernel, dilation in MODEL_D_STAGES:
            blocks.append(nn.ModuleDict({
                "depthwise": CausalConv1d(in_ch, in_ch, kernel, dilation=dilation, groups=in_ch),
                "pointwise": CausalConv1d(in_ch, out_ch, kernel=1, dilation=1, groups=1),
                "bn": nn.BatchNorm1d(out_ch),
                "relu": nn.ReLU(inplace=True),
            }))
        self.blocks = nn.ModuleList(blocks)
        self.gap = nn.AdaptiveAvgPool1d(1)
        self.flatten = nn.Flatten(1)
        self.linear = nn.Linear(MODEL_D_GAP_CHANNELS, N_CLASSES)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != INPUT_CHANNELS:
            raise ValueError(f"expected input (B, {INPUT_CHANNELS}, T), got {tuple(x.shape)}")
        for block in self.blocks:
            x = block["depthwise"](x)
            x = block["pointwise"](x)
            x = block["bn"](x)
            x = block["relu"](x)
        x = self.gap(x)
        x = self.flatten(x)
        return self.linear(x)

    def causal_history_per_stage(self) -> list[int]:
        """The `history_required` (ring-buffer length) of each stage's depthwise conv, in execution
        order -- the pointwise (kernel=1) convs need no history of their own."""
        return [b["depthwise"].history_required for b in self.blocks]


EXTENSION_MODEL_BUILDERS = {"Linear": LinearBaseline, "D": TinyCausalCNN}


def build_model_ext(name: str) -> nn.Module:
    if name not in EXTENSION_MODEL_BUILDERS:
        raise ValueError(f"Unknown extension model '{name}'; choose from {list(EXTENSION_MODEL_BUILDERS)}")
    return EXTENSION_MODEL_BUILDERS[name]()


class VariableChannelCNN(nn.Module):
    """Interprets an `src.architectures` Op spec exactly like `src.model.TinyCNN`, EXCEPT the number of
    input channels is a constructor argument rather than the fixed `src.architectures.INPUT_CHANNELS`
    constant -- needed for the RQ4 sensor ablation, whose 3-channel arms (accelerometer-only,
    gyroscope-only) have a different input width than the frozen A/B/C models. Reuses
    `src.model._build_layer` so the SAME op-to-nn.Module translation is used everywhere in the project;
    the only new logic here is the input-channel check.
    """

    def __init__(self, name: str, spec: tuple, in_channels: int):
        super().__init__()
        from src.model import _build_layer   # reuse the single source of truth for op -> nn.Module
        from src.architectures import GAPOp

        self.name = name
        self.in_channels = in_channels
        layers: list[nn.Module] = []
        for op in spec:
            layers.append(_build_layer(op))
            if isinstance(op, GAPOp):
                layers.append(nn.Flatten(1))
        self.layers = nn.ModuleList(layers)
        self.spec = spec

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if x.dim() != 3 or x.shape[1] != self.in_channels:
            raise ValueError(f"expected input (B, {self.in_channels}, T), got {tuple(x.shape)}")
        for layer in self.layers:
            x = layer(x)
        return x


def build_sensor_ablation_model(in_channels: int) -> nn.Module:
    """Model-A-architecture variant for the RQ4 sensor ablation; see
    `architectures_ext.model_a_variant_spec` for why Model A's architecture (not a new search) is used."""
    from .architectures_ext import model_a_variant_spec

    return VariableChannelCNN("A_variant", model_a_variant_spec(in_channels), in_channels)


def dummy_input(batch_size: int = 1, length: int = INPUT_LENGTH, channels: int = INPUT_CHANNELS, device=None):
    return torch.randn(batch_size, channels, length, device=device)
