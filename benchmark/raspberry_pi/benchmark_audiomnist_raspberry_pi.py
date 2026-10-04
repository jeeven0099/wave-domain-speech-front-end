#!/usr/bin/env python3
"""Reproducible CPU-only Raspberry Pi benchmark for frozen AudioMNIST models.

Edit the data/model paths below, or provide the equivalent command-line
arguments. The built-in adapters reproduce six matched compact architectures
at 16 kHz log-Mel and 400 Hz digital RMS, the same six architectures
retargeted onto the fabricated five-channel 400 Hz AC-RMS data
(fabricated_400hz_baselines), plus the fabricated five-channel Pooled TCN.
The primary fabricated benchmark and the fabricated-baseline comparison both
start from saved high-rate DAQ waveforms. Whisper is intentionally excluded.
"""

from __future__ import annotations

import os

# These must be set before NumPy, SciPy, or PyTorch initialize a thread pool.
os.environ["CUDA_VISIBLE_DEVICES"] = ""
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("VECLIB_MAXIMUM_THREADS", "1")
os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import argparse
import csv
import hashlib
import json
import math
import platform
import statistics
import subprocess
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Iterable, Sequence

import numpy as np
import psutil
import torch
import torch.nn.functional as F
from torch import nn


# ---------------------------------------------------------------------------
# User configuration. Environment variables or CLI flags override these.
# ---------------------------------------------------------------------------
TEST_SPLIT_PATH = Path(os.environ.get("TEST_SPLIT_PATH", "/path/to/protocol.json"))
RAW_AUDIOMNIST_PATH = Path(os.environ.get("RAW_AUDIOMNIST_PATH", "/path/to/AudioMNIST"))
FABRICATED_FEATURE_PATH = Path(
    os.environ.get("FABRICATED_FEATURE_PATH", "/path/to/audiomnist_daq_experiment/captures")
)
MODEL_16K_DIR = Path(os.environ.get("MODEL_16K_DIR", "/path/to/digital_16k_models"))
MODEL_400_DIGITAL_DIR = Path(
    os.environ.get("MODEL_400_DIGITAL_DIR", "/path/to/digital_400hz_models")
)
MODEL_FABRICATED_PATH = Path(
    os.environ.get("MODEL_FABRICATED_PATH", "/path/to/best_envelope_400hz_5d_pooled_tcn_classifier.pt")
)
MODEL_FABRICATED_BASELINES_DIR = Path(
    os.environ.get(
        "MODEL_FABRICATED_BASELINES_DIR", "/path/to/fabricated_400hz_5d_baseline_models"
    )
)

RESULTS_DIR = Path("results")
PRIMARY_SYSTEMS = ("digital_16k", "digital_envelope_400hz", "fabricated_400hz")
OPTIONAL_SYSTEMS = (
    "fabricated_post_envelope",
    "fabricated_400hz_baselines",
    "fabricated_400hz_baselines_post_envelope",
)
SYSTEMS = (*PRIMARY_SYSTEMS, *OPTIONAL_SYSTEMS)
DIGITAL_MODEL_NAMES = (
    "bc_resnet_1",
    "bc_resnet_2",
    "edgespeechnet_a",
    "matchboxnet_3x1x64",
    "mlperf_tiny_dscnn",
    "tc_resnet8",
)
FIXED_SECONDS = 1.0
TARGET_PEAK_DBFS = -9.0
EXPECTED_TEST_EXAMPLES = 6_000


@dataclass(frozen=True)
class RawExample:
    sample_id: str
    waveform: np.ndarray
    source_rate: int
    label: int
    audio_duration_seconds: float = FIXED_SECONDS


@dataclass(frozen=True)
class FabricatedExample:
    sample_id: str
    features: np.ndarray
    label: int
    audio_duration_seconds: float


@dataclass(frozen=True)
class FabricatedRawExample:
    sample_id: str
    waveform: np.ndarray
    source_rate: float
    label: int
    audio_duration_seconds: float


@dataclass(frozen=True)
class PreparedInput:
    args: tuple[torch.Tensor, ...]


@dataclass
class LoadedSystem:
    name: str
    representation: str
    source_sample_rate_hz: int
    feature_rate_hz: int
    channels: int
    model: nn.Module
    mean: np.ndarray
    std: np.ndarray
    architecture: str
    checkpoint_metadata: dict[str, Any]


def _torch_load(path: Path) -> Any:
    try:
        return torch.load(path, map_location="cpu", weights_only=False)
    except TypeError:  # PyTorch before weights_only was introduced.
        return torch.load(path, map_location="cpu")


def configure_torch_runtime() -> None:
    torch.set_num_threads(1)
    try:
        torch.set_num_interop_threads(1)
    except RuntimeError as error:
        raise RuntimeError(
            "torch.set_num_interop_threads(1) must run before any parallel PyTorch work"
        ) from error
    if torch.get_num_threads() != 1 or torch.get_num_interop_threads() != 1:
        raise RuntimeError("PyTorch thread configuration is not 1/1")


# ---------------------------------------------------------------------------
# Exact model definitions copied from the two frozen AudioMNIST experiment
# contracts. Add another checkpoint adapter in load_model() rather than
# inferring a topology from a filename.
# ---------------------------------------------------------------------------
class CausalConv1d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.left_context = dilation * (kernel_size - 1)
        self.conv = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.conv(F.pad(values, (self.left_context, 0)))


class ConvBNAct1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        stride: int = 1,
        dilation: int = 1,
    ) -> None:
        super().__init__(
            CausalConv1d(
                in_channels, out_channels, kernel_size, stride=stride, dilation=dilation
            ),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
        )


class TCSConv1d(nn.Sequential):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int,
        dilation: int = 1,
        activation: bool = True,
    ) -> None:
        layers: list[nn.Module] = [
            CausalConv1d(
                in_channels,
                in_channels,
                kernel_size,
                dilation=dilation,
                groups=in_channels,
            ),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
        ]
        if activation:
            layers.append(nn.ReLU(inplace=True))
        super().__init__(*layers)


class MatchboxResidualBlock(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.main = TCSConv1d(in_channels, channels, kernel_size, activation=False)
        self.residual = (
            nn.Sequential(
                nn.Conv1d(in_channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
            )
            if in_channels != channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.residual(values), inplace=True)


class MatchboxNet3x1x64(nn.Module):
    def __init__(self, input_features: int = 40, num_classes: int = 10) -> None:
        super().__init__()
        self.prologue = TCSConv1d(input_features, 128, 11)
        self.blocks = nn.Sequential(
            MatchboxResidualBlock(128, 64, 13),
            MatchboxResidualBlock(64, 64, 15),
            MatchboxResidualBlock(64, 64, 17),
        )
        self.epilogue = nn.Sequential(
            TCSConv1d(64, 128, 29, dilation=2),
            ConvBNAct1d(128, 128, 1),
        )
        self.classifier = nn.Linear(128, num_classes)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.epilogue(self.blocks(self.prologue(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class TCResidualBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct1d(in_channels, out_channels, 3, stride=stride),
            CausalConv1d(out_channels, out_channels, 3),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class TCResNet8(nn.Module):
    """Exact 40-bin log-Mel TC-ResNet8 from the 16 kHz notebook."""

    def __init__(self, input_features: int = 40, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(input_features, 16, 3)
        self.blocks = nn.Sequential(
            TCResidualBlock(16, 24, 2),
            TCResidualBlock(24, 32, 2),
            TCResidualBlock(32, 48, 2),
        )
        self.classifier = nn.Linear(48, num_classes)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class TCResNet8Envelope(TCResNet8):
    """Exact one-channel TC-ResNet8 adaptation from the 400 Hz notebook."""

    def __init__(self) -> None:
        super().__init__(input_features=1, num_classes=10)


class CausalTimeFirstSamePadConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int],
        groups: int = 1,
        bias: bool = False,
    ) -> None:
        super().__init__()
        self.kernel_size = (
            kernel_size if isinstance(kernel_size, tuple) else (kernel_size, kernel_size)
        )
        self.stride = stride if isinstance(stride, tuple) else (stride, stride)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            self.kernel_size,
            stride=self.stride,
            groups=groups,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        height, width = values.shape[-2:]
        out_h = math.ceil(height / self.stride[0])
        out_w = math.ceil(width / self.stride[1])
        pad_h = max((out_h - 1) * self.stride[0] + self.kernel_size[0] - height, 0)
        pad_w = max((out_w - 1) * self.stride[1] + self.kernel_size[1] - width, 0)
        return self.conv(
            F.pad(values, (pad_w // 2, pad_w - pad_w // 2, pad_h, 0))
        )


class DSConvBlock2d(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            CausalTimeFirstSamePadConv2d(
                channels, channels, 3, 1, groups=channels
            ),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(channels, channels, 1, bias=False),
            nn.BatchNorm2d(channels),
            nn.ReLU(inplace=True),
        )


class MLPerfTinyDSCNN(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalTimeFirstSamePadConv2d(1, 64, (10, 4), (2, 2)),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(*(DSConvBlock2d(64) for _ in range(4)))
        self.classifier = nn.Linear(64, num_classes)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.unsqueeze(1))).mean(dim=-1).transpose(1, 2)
        return self.classifier(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class SubSpectralNorm(nn.Module):
    def __init__(self, channels: int, groups: int = 5) -> None:
        super().__init__()
        self.groups = groups
        self.norm = nn.BatchNorm2d(channels * groups)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        batch, channels, frequency, time_steps = values.shape
        if frequency % self.groups:
            raise RuntimeError(
                f"Frequency dimension {frequency} is not divisible by {self.groups}"
            )
        values = values.reshape(
            batch, channels * self.groups, frequency // self.groups, time_steps
        )
        return self.norm(values).reshape(batch, channels, frequency, time_steps)


class CausalTimeLastConv2d(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        kernel_size: int | tuple[int, int],
        stride: int | tuple[int, int] = 1,
        dilation: int | tuple[int, int] = 1,
        groups: int = 1,
        bias: bool = False,
        frequency_padding: str | int = "same",
    ) -> None:
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
        stride = stride if isinstance(stride, tuple) else (stride,) * 2
        dilation = dilation if isinstance(dilation, tuple) else (dilation,) * 2
        frequency_context = dilation[0] * (kernel_size[0] - 1)
        if frequency_padding == "same":
            self.frequency_padding = (
                frequency_context // 2,
                frequency_context - frequency_context // 2,
            )
        else:
            padding = int(frequency_padding)
            self.frequency_padding = (padding, padding)
        self.time_left_context = dilation[1] * (kernel_size[1] - 1)
        self.conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            dilation=dilation,
            groups=groups,
            bias=bias,
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        frequency_top, frequency_bottom = self.frequency_padding
        return self.conv(
            F.pad(
                values,
                (self.time_left_context, 0, frequency_top, frequency_bottom),
            )
        )


class BCConvBNAct(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int,
        stage_index: int,
        kernel_size: int | tuple[int, int] = 3,
        stride: int | tuple[int, int] = 1,
        groups: int = 1,
        use_dilation: bool = False,
        activation: str | None = "relu",
        batch_norm: bool = True,
        subspectral_norm: bool = False,
    ) -> None:
        super().__init__()
        kernel_size = kernel_size if isinstance(kernel_size, tuple) else (kernel_size,) * 2
        stride = stride if isinstance(stride, tuple) else (stride,) * 2
        dilation = 2**stage_index if use_dilation else 1
        layers: list[nn.Module] = [
            CausalTimeLastConv2d(
                in_channels,
                out_channels,
                kernel_size=kernel_size,
                stride=stride,
                dilation=(dilation, dilation),
                groups=groups,
            )
        ]
        if subspectral_norm:
            layers.append(SubSpectralNorm(out_channels, groups=5))
        elif batch_norm:
            layers.append(nn.BatchNorm2d(out_channels))
        if activation == "relu":
            layers.append(nn.ReLU(inplace=True))
        elif activation == "silu":
            layers.append(nn.SiLU(inplace=True))
        elif activation is not None:
            raise ValueError(activation)
        self.block = nn.Sequential(*layers)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.block(values)


class BCResBlock(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, stage_index: int, stride: int
    ) -> None:
        super().__init__()
        self.transition = in_channels != out_channels
        frequency_layers: list[nn.Module] = []
        if self.transition:
            frequency_layers.append(
                BCConvBNAct(in_channels, out_channels, stage_index, kernel_size=1)
            )
            in_channels = out_channels
        frequency_layers.append(
            BCConvBNAct(
                in_channels,
                out_channels,
                stage_index,
                kernel_size=(3, 1),
                stride=(stride, 1),
                groups=out_channels,
                activation=None,
                subspectral_norm=True,
            )
        )
        self.frequency_layers = nn.Sequential(*frequency_layers)
        self.temporal = nn.Sequential(
            BCConvBNAct(
                out_channels,
                out_channels,
                stage_index,
                kernel_size=(1, 3),
                groups=out_channels,
                use_dilation=True,
                activation="silu",
            ),
            nn.Conv2d(out_channels, out_channels, 1, bias=False),
            nn.Dropout2d(0.1),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        frequency = self.frequency_layers(values)
        output = frequency + self.temporal(frequency.mean(dim=2, keepdim=True))
        if not self.transition:
            output = output + values
        return F.relu(output, inplace=True)


class BCResNet(nn.Module):
    def __init__(self, scale: int = 1, num_classes: int = 10) -> None:
        super().__init__()
        channels = [16 * scale, 8 * scale, 12 * scale, 16 * scale, 20 * scale, 32 * scale]
        self.stem = nn.Sequential(
            CausalTimeLastConv2d(1, channels[0], 5, stride=(2, 1)),
            nn.BatchNorm2d(channels[0]),
            nn.ReLU(inplace=True),
        )
        stages = []
        for stage_index, repeats in enumerate((2, 2, 4, 4)):
            blocks = []
            in_channels = channels[stage_index]
            out_channels = channels[stage_index + 1]
            for block_index in range(repeats):
                stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                blocks.append(BCResBlock(in_channels, out_channels, stage_index, stride))
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final_features = nn.Sequential(
            CausalTimeLastConv2d(
                channels[4],
                channels[4],
                (5, 5),
                groups=channels[4],
                frequency_padding=0,
            ),
            nn.Conv2d(channels[4], channels[5], 1, bias=False),
            nn.BatchNorm2d(channels[5]),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(channels[5], num_classes)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2).unsqueeze(1)
        values = self.stages(self.stem(values))
        values = self.final_features(values).mean(dim=2).transpose(1, 2)
        return self.classifier(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class EdgeResidualBottleneck(nn.Module):
    def __init__(self, middle_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            CausalTimeLastConv2d(39, middle_channels, 3),
            nn.BatchNorm2d(middle_channels),
            nn.ReLU(inplace=True),
            CausalTimeLastConv2d(middle_channels, 39, 3),
            nn.BatchNorm2d(39),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(values + self.layers(values), inplace=True)


class EdgeSpeechNetA(nn.Module):
    def __init__(self, num_classes: int = 10) -> None:
        super().__init__()
        self.stem = nn.Sequential(
            CausalTimeLastConv2d(1, 39, 3),
            nn.BatchNorm2d(39),
            nn.ReLU(inplace=True),
        )
        self.blocks = nn.Sequential(
            *(EdgeResidualBottleneck(width) for width in (20, 15, 25, 22, 22, 25))
        )
        self.final = nn.Sequential(
            CausalTimeLastConv2d(39, 45, 3),
            nn.BatchNorm2d(45),
            nn.ReLU(inplace=True),
        )
        self.classifier = nn.Linear(45, num_classes)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2).unsqueeze(1)
        values = self.final(self.blocks(self.stem(values))).mean(dim=2).transpose(1, 2)
        return self.classifier(values)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class EndpointClassifier400(nn.Module):
    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class MatchboxResidualEnvelope(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.main = TCSConv1d(in_channels, channels, kernel_size, activation=False)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, channels, 1, bias=False),
                nn.BatchNorm1d(channels),
            )
            if in_channels != channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class MatchboxNet3x1x64Envelope(EndpointClassifier400):
    def __init__(self) -> None:
        super().__init__()
        self.prologue = TCSConv1d(1, 128, 11)
        self.blocks = nn.Sequential(
            MatchboxResidualEnvelope(128, 64, 13),
            MatchboxResidualEnvelope(64, 64, 15),
            MatchboxResidualEnvelope(64, 64, 17),
        )
        self.epilogue = nn.Sequential(
            TCSConv1d(64, 128, 29, dilation=2),
            ConvBNAct1d(128, 128, 1),
        )
        self.classifier = nn.Linear(128, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.epilogue(self.blocks(self.prologue(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class DSConvBlock1d(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            CausalConv1d(channels, channels, 3, groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )


class MLPerfTinyDSCNNEnvelope(EndpointClassifier400):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(1, 64, 10, stride=2)
        self.blocks = nn.Sequential(*(DSConvBlock1d(64) for _ in range(4)))
        self.classifier = nn.Linear(64, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class BCResidual1d(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            CausalConv1d(
                in_channels,
                in_channels,
                3,
                stride=stride,
                dilation=dilation,
                groups=in_channels,
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.1),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class BCResNetEnvelope(EndpointClassifier400):
    def __init__(self, scale: int = 1) -> None:
        super().__init__()
        channels = [16 * scale, 8 * scale, 12 * scale, 16 * scale, 20 * scale]
        self.stem = ConvBNAct1d(1, channels[0], 5, stride=2)
        stages = []
        in_channels = channels[0]
        for stage_index, (out_channels, repeats) in enumerate(
            zip(channels[1:], (2, 2, 4, 4))
        ):
            blocks = []
            for block_index in range(repeats):
                stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                blocks.append(
                    BCResidual1d(
                        in_channels,
                        out_channels,
                        dilation=2**stage_index,
                        stride=stride,
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final = ConvBNAct1d(channels[-1], 32 * scale, 5)
        self.classifier = nn.Linear(32 * scale, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.stages(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class EdgeResidual1d(nn.Module):
    def __init__(self, middle_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct1d(39, middle_channels, 3),
            CausalConv1d(middle_channels, 39, 3),
            nn.BatchNorm1d(39),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(values + self.layers(values), inplace=True)


class EdgeSpeechNetAEnvelope(EndpointClassifier400):
    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(1, 39, 3)
        self.blocks = nn.Sequential(
            *(EdgeResidual1d(width) for width in (20, 15, 25, 22, 22, 25))
        )
        self.final = ConvBNAct1d(39, 45, 3)
        self.classifier = nn.Linear(45, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.blocks(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


DIGITAL_16K_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "bc_resnet_1": lambda: BCResNet(scale=1),
    "bc_resnet_2": lambda: BCResNet(scale=2),
    "edgespeechnet_a": EdgeSpeechNetA,
    "matchboxnet_3x1x64": MatchboxNet3x1x64,
    "mlperf_tiny_dscnn": MLPerfTinyDSCNN,
    "tc_resnet8": TCResNet8,
}

DIGITAL_400_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "bc_resnet_1": lambda: BCResNetEnvelope(scale=1),
    "bc_resnet_2": lambda: BCResNetEnvelope(scale=2),
    "edgespeechnet_a": EdgeSpeechNetAEnvelope,
    "matchboxnet_3x1x64": MatchboxNet3x1x64Envelope,
    "mlperf_tiny_dscnn": MLPerfTinyDSCNNEnvelope,
    "tc_resnet8": TCResNet8Envelope,
}


class CausalDepthwiseBlock(nn.Module):
    def __init__(self, channels: int, kernel_size: int, dilation: int, dropout: float) -> None:
        super().__init__()
        self.left_context = dilation * (kernel_size - 1)
        self.depthwise = nn.Conv1d(
            channels,
            channels,
            kernel_size,
            dilation=dilation,
            groups=channels,
        )
        self.pointwise = nn.Conv1d(channels, channels, 1)
        self.normalization = nn.LayerNorm(channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        transformed = F.pad(values, (self.left_context, 0))
        transformed = self.pointwise(self.depthwise(transformed))
        transformed = self.normalization(transformed.transpose(1, 2)).transpose(1, 2)
        return values + self.dropout(F.silu(transformed))


class FabricatedPooledTcn(nn.Module):
    """Exact five-channel Pooled TCN used for the fabricated AudioMNIST result."""

    def __init__(
        self,
        channels: int,
        frontend_kernel_size: int,
        frontend_stride: int,
        tcn_kernel_size: int,
        dilations: Sequence[int],
        dropout: float,
        classes: int,
    ) -> None:
        super().__init__()
        if int(frontend_stride) != 4:
            raise ValueError("The frozen fabricated model contract requires stride 4")
        self.frontend_kernel_size = int(frontend_kernel_size)
        self.frontend_stride = int(frontend_stride)
        self.frontend_depthwise = nn.Conv1d(
            5,
            5,
            self.frontend_kernel_size,
            stride=self.frontend_stride,
            groups=5,
        )
        self.frontend_pointwise = nn.Conv1d(5, int(channels), 1)
        self.frontend_normalization = nn.LayerNorm(int(channels))
        self.frontend_dropout = nn.Dropout(float(dropout))
        self.blocks = nn.ModuleList(
            CausalDepthwiseBlock(
                int(channels), int(tcn_kernel_size), int(dilation), float(dropout)
            )
            for dilation in dilations
        )
        self.pool_dropout = nn.Dropout(float(dropout))
        self.classifier = nn.Linear(2 * int(channels), int(classes))

    def encode_frames(
        self, values: torch.Tensor, lengths: torch.Tensor
    ) -> tuple[torch.Tensor, torch.Tensor]:
        output = values.transpose(1, 2)
        output = F.pad(output, (self.frontend_kernel_size - 1, 0))
        output = self.frontend_pointwise(self.frontend_depthwise(output))
        output = self.frontend_normalization(output.transpose(1, 2)).transpose(1, 2)
        output = self.frontend_dropout(F.silu(output))
        for block in self.blocks:
            output = block(output)
        output_lengths = torch.div(
            lengths + self.frontend_stride - 1,
            self.frontend_stride,
            rounding_mode="floor",
        )
        return output, output_lengths

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frames, output_lengths = self.encode_frames(values, lengths)
        steps = torch.arange(frames.shape[-1], device=frames.device)
        valid = steps.unsqueeze(0) < output_lengths.unsqueeze(1)
        valid_float = valid.unsqueeze(1).to(frames.dtype)
        mean = (frames * valid_float).sum(dim=-1) / output_lengths.to(
            frames.dtype
        ).clamp_min(1).unsqueeze(1)
        maximum = frames.masked_fill(~valid.unsqueeze(1), -torch.inf).amax(dim=-1)
        return self.classifier(self.pool_dropout(torch.cat([mean, maximum], dim=1)))


# ---------------------------------------------------------------------------
# Five-channel baseline-architecture variants: the same six matched compact
# architectures backing digital_envelope_400hz, retargeted onto the identical
# five-channel fabricated 400 Hz AC-RMS data that trains FabricatedPooledTcn,
# for a direct apples-to-apples architecture comparison on identical
# fabricated front-end features. Every backbone block below (stem aside) is
# an exact copy of its one-channel *Envelope sibling above; only the stem's
# in_channels and the endpoint pooling (length-masked, for the variable-length
# training regime) differ.
# ---------------------------------------------------------------------------
FABRICATED_BASELINE_INPUT_CHANNELS = 5


def _fabricated_baseline_output_lengths(
    lengths: torch.Tensor, total_stride: int
) -> torch.Tensor:
    if total_stride == 1:
        return lengths
    return torch.div(lengths + total_stride - 1, total_stride, rounding_mode="floor")


class EndpointClassifierFabricatedBaseline(nn.Module):
    """Shared length-masked mean endpoint pooling for five-channel input."""

    total_stride: int = 1

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frame_logits = self.encode_frames(values)
        output_lengths = _fabricated_baseline_output_lengths(lengths, self.total_stride)
        steps = torch.arange(frame_logits.shape[1], device=frame_logits.device)
        valid = steps.unsqueeze(0) < output_lengths.to(frame_logits.device).unsqueeze(1)
        valid_float = valid.unsqueeze(-1).to(frame_logits.dtype)
        summed = (frame_logits * valid_float).sum(dim=1)
        denominator = (
            output_lengths.to(frame_logits.device, frame_logits.dtype)
            .clamp_min(1)
            .unsqueeze(1)
        )
        return summed / denominator


class MatchboxResidualFabricatedBaseline(nn.Module):
    def __init__(self, in_channels: int, channels: int, kernel_size: int) -> None:
        super().__init__()
        self.main = TCSConv1d(in_channels, channels, kernel_size, activation=False)
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, channels, 1, bias=False), nn.BatchNorm1d(channels)
            )
            if in_channels != channels
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class MatchboxNet3x1x64FabricatedBaseline(EndpointClassifierFabricatedBaseline):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.prologue = TCSConv1d(FABRICATED_BASELINE_INPUT_CHANNELS, 128, 11)
        self.blocks = nn.Sequential(
            MatchboxResidualFabricatedBaseline(128, 64, 13),
            MatchboxResidualFabricatedBaseline(64, 64, 15),
            MatchboxResidualFabricatedBaseline(64, 64, 17),
        )
        self.epilogue = nn.Sequential(
            TCSConv1d(64, 128, 29, dilation=2), ConvBNAct1d(128, 128, 1)
        )
        self.classifier = nn.Linear(128, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.epilogue(self.blocks(self.prologue(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class TCResidualBlockFabricatedBaseline(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, stride: int) -> None:
        super().__init__()
        self.main = nn.Sequential(
            ConvBNAct1d(in_channels, out_channels, 3, stride=stride),
            CausalConv1d(out_channels, out_channels, 3),
            nn.BatchNorm1d(out_channels),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class TCResNet8FabricatedBaseline(EndpointClassifierFabricatedBaseline):
    total_stride = 8

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, 16, 3)
        self.blocks = nn.Sequential(
            TCResidualBlockFabricatedBaseline(16, 24, 2),
            TCResidualBlockFabricatedBaseline(24, 32, 2),
            TCResidualBlockFabricatedBaseline(32, 48, 2),
        )
        self.classifier = nn.Linear(48, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class DSConvBlock1dFabricatedBaseline(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            CausalConv1d(channels, channels, 3, groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )


class MLPerfTinyDSCNNFabricatedBaseline(EndpointClassifierFabricatedBaseline):
    total_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, 64, 10, stride=2)
        self.blocks = nn.Sequential(*(DSConvBlock1dFabricatedBaseline(64) for _ in range(4)))
        self.classifier = nn.Linear(64, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class BCResidual1dFabricatedBaseline(nn.Module):
    def __init__(
        self, in_channels: int, out_channels: int, dilation: int, stride: int = 1
    ) -> None:
        super().__init__()
        self.main = nn.Sequential(
            CausalConv1d(
                in_channels, in_channels, 3, stride=stride, dilation=dilation, groups=in_channels
            ),
            nn.BatchNorm1d(in_channels),
            nn.SiLU(inplace=True),
            nn.Conv1d(in_channels, out_channels, 1, bias=False),
            nn.BatchNorm1d(out_channels),
            nn.Dropout(0.1),
        )
        self.skip = (
            nn.Sequential(
                nn.Conv1d(in_channels, out_channels, 1, stride=stride, bias=False),
                nn.BatchNorm1d(out_channels),
            )
            if in_channels != out_channels or stride != 1
            else nn.Identity()
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(self.main(values) + self.skip(values), inplace=True)


class BCResNetFabricatedBaseline(EndpointClassifierFabricatedBaseline):
    total_stride = 8

    def __init__(self, scale: int = 1) -> None:
        super().__init__()
        channels = [16 * scale, 8 * scale, 12 * scale, 16 * scale, 20 * scale]
        self.stem = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, channels[0], 5, stride=2)
        stages = []
        in_channels = channels[0]
        for stage_index, (out_channels, repeats) in enumerate(zip(channels[1:], (2, 2, 4, 4))):
            blocks = []
            for block_index in range(repeats):
                stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                blocks.append(
                    BCResidual1dFabricatedBaseline(
                        in_channels, out_channels, dilation=2**stage_index, stride=stride
                    )
                )
                in_channels = out_channels
            stages.append(nn.Sequential(*blocks))
        self.stages = nn.Sequential(*stages)
        self.final = ConvBNAct1d(channels[-1], 32 * scale, 5)
        self.classifier = nn.Linear(32 * scale, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.stages(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class EdgeResidual1dFabricatedBaseline(nn.Module):
    def __init__(self, middle_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct1d(39, middle_channels, 3),
            CausalConv1d(middle_channels, 39, 3),
            nn.BatchNorm1d(39),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(values + self.layers(values), inplace=True)


class EdgeSpeechNetAFabricatedBaseline(EndpointClassifierFabricatedBaseline):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, 39, 3)
        self.blocks = nn.Sequential(
            *(EdgeResidual1dFabricatedBaseline(width) for width in (20, 15, 25, 22, 22, 25))
        )
        self.final = ConvBNAct1d(39, 45, 3)
        self.classifier = nn.Linear(45, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.blocks(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


FABRICATED_400_BASELINE_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "bc_resnet_1": lambda: BCResNetFabricatedBaseline(scale=1),
    "bc_resnet_2": lambda: BCResNetFabricatedBaseline(scale=2),
    "edgespeechnet_a": EdgeSpeechNetAFabricatedBaseline,
    "matchboxnet_3x1x64": MatchboxNet3x1x64FabricatedBaseline,
    "mlperf_tiny_dscnn": MLPerfTinyDSCNNFabricatedBaseline,
    "tc_resnet8": TCResNet8FabricatedBaseline,
}
FABRICATED_400_BASELINE_ARCHITECTURE_IDS: dict[str, str] = {
    model_name: f"fabricated_400hz_5d_{model_name}_masked_mean_softmax"
    for model_name in FABRICATED_400_BASELINE_FACTORIES
}


# ---------------------------------------------------------------------------
# Three literature-minimal CNN baselines (cnn_baseline_models.py), fabricated
# form: causal, length-masked-mean classifiers over the identical 5-channel
# 400 Hz AC-RMS input as FABRICATED_400_BASELINE_FACTORIES above. Reported
# under the same fabricated_400hz_baselines* system names as those models --
# the system name identifies the input representation, not the architecture
# family, so these share it rather than getting a system name of their own.
# ---------------------------------------------------------------------------
class CnnOneFstride4Fabricated(EndpointClassifierFabricatedBaseline):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, 16, kernel_size=8)
        self.classifier = nn.Linear(16, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values.transpose(1, 2))
        return self.classifier(values.transpose(1, 2))


class TinyConvFabricated(EndpointClassifierFabricatedBaseline):
    total_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = CausalConv1d(
            FABRICATED_BASELINE_INPUT_CHANNELS, 8, kernel_size=10, stride=2, bias=True
        )
        self.classifier = nn.Linear(8, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class CnnTradFpool3Fabricated(EndpointClassifierFabricatedBaseline):
    total_stride = 3

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBNAct1d(FABRICATED_BASELINE_INPUT_CHANNELS, 16, kernel_size=20, stride=3)
        self.conv2 = ConvBNAct1d(16, 16, kernel_size=10)
        self.classifier = nn.Linear(16, 10)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv2(self.conv1(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


CNN_BASELINE_FABRICATED_FACTORIES: dict[str, Callable[[], nn.Module]] = {
    "cnn_one_fstride4": CnnOneFstride4Fabricated,
    "tiny_conv": TinyConvFabricated,
    "cnn_trad_fpool3": CnnTradFpool3Fabricated,
}
CNN_BASELINE_ARCHITECTURE_IDS: dict[str, str] = {
    model_name: f"basic_cnn_baseline_{model_name}"
    for model_name in CNN_BASELINE_FABRICATED_FACTORIES
}
CNN_BASELINE_MODEL_NAMES = tuple(CNN_BASELINE_FABRICATED_FACTORIES)


def _checkpoint_state(checkpoint: dict[str, Any], *keys: str) -> dict[str, torch.Tensor]:
    for key in keys:
        value = checkpoint.get(key)
        if isinstance(value, dict):
            return value
    raise KeyError(f"Checkpoint has none of the state-dict keys {keys}")


def _normalizer_next_to_checkpoint(path: Path) -> tuple[np.ndarray, np.ndarray]:
    candidates = (
        path.parent / "compact_train_normalizer.npz",
        path.parent.parent / "compact_train_normalizer.npz",
    )
    for candidate in candidates:
        if candidate.is_file():
            with np.load(candidate) as values:
                return (
                    np.asarray(values["mean"], dtype=np.float32),
                    np.asarray(values["std"], dtype=np.float32),
                )
    raise FileNotFoundError(
        "The 16 kHz checkpoint does not store its train-only normalizer. Place "
        f"compact_train_normalizer.npz beside {path.name}."
    )


def load_model(
    system: str, checkpoint_path: Path, model_name: str | None = None
) -> LoadedSystem:
    """Load a frozen checkpoint and validate its exact architecture contract."""
    checkpoint = _torch_load(checkpoint_path)
    if not isinstance(checkpoint, dict):
        raise TypeError(f"Expected a checkpoint dictionary: {checkpoint_path}")

    if system == "digital_16k":
        metadata = dict(checkpoint.get("metadata", {}))
        architecture = str(metadata.get("architecture", ""))
        expected = model_name or architecture
        if expected not in DIGITAL_16K_FACTORIES or architecture != expected:
            raise NotImplementedError(
                f"Unsupported or mismatched 16 kHz architecture: requested={expected!r}, "
                f"checkpoint={architecture!r}"
            )
        model = DIGITAL_16K_FACTORIES[expected]()
        model.load_state_dict(_checkpoint_state(checkpoint, "state_dict"), strict=True)
        mean, std = _normalizer_next_to_checkpoint(checkpoint_path)
        loaded = LoadedSystem(
            name=f"{system}/{expected}",
            representation="pre-resampled 16 kHz waveform -> 40-bin log-Mel",
            source_sample_rate_hz=16_000,
            feature_rate_hz=100,
            channels=1,
            model=model,
            mean=mean,
            std=std,
            architecture=architecture,
            checkpoint_metadata=metadata,
        )
    elif system == "digital_envelope_400hz":
        architecture = str(checkpoint.get("model_name", ""))
        expected_base = model_name or architecture.removesuffix("_1d")
        expected_architecture = f"{expected_base}_1d"
        if (
            expected_base not in DIGITAL_400_FACTORIES
            or architecture != expected_architecture
        ):
            raise NotImplementedError(
                f"Unsupported or mismatched 400 Hz architecture: requested={expected_base!r}, "
                f"checkpoint={architecture!r}"
            )
        model = DIGITAL_400_FACTORIES[expected_base]()
        model.load_state_dict(_checkpoint_state(checkpoint, "model_state"), strict=True)
        mean = np.asarray(checkpoint["train_mean"], dtype=np.float32).reshape(1)
        std = np.asarray(checkpoint["train_std"], dtype=np.float32).reshape(1)
        loaded = LoadedSystem(
            name=f"{system}/{expected_base}",
            representation="48 kHz waveform -> digital 10 ms RMS at 400 Hz",
            source_sample_rate_hz=48_000,
            feature_rate_hz=400,
            channels=1,
            model=model,
            mean=mean,
            std=std,
            architecture=architecture,
            checkpoint_metadata={
                "experiment_contract": checkpoint.get("experiment_contract", {}),
                "epoch": checkpoint.get("epoch"),
            },
        )
    elif system in ("fabricated_400hz", "fabricated_post_envelope"):
        architecture = str(checkpoint.get("architecture", ""))
        expected = "envelope_400hz_c32_fk32_long_tcn_stats_pool_softmax"
        if architecture != expected:
            raise NotImplementedError(
                f"Expected fabricated architecture {expected!r}, found {architecture!r}. "
                "Add its exact constructor to the fabricated_400hz branch of load_model()."
            )
        config = dict(checkpoint["model_config"])
        model = FabricatedPooledTcn(
            channels=int(config["channels"]),
            frontend_kernel_size=int(config["frontend_kernel_size"]),
            frontend_stride=int(config["frontend_stride"]),
            tcn_kernel_size=int(config["tcn_kernel_size"]),
            dilations=tuple(int(value) for value in config["dilations"]),
            dropout=float(config["dropout"]),
            classes=int(config["classes"]),
        )
        model.load_state_dict(_checkpoint_state(checkpoint, "model_state"), strict=True)
        loaded = LoadedSystem(
            name=system,
            representation=(
                "high-rate five-channel DAQ waveform -> digital 10 ms AC-RMS at 400 Hz"
                if system == "fabricated_400hz"
                else "precomputed five-channel 400 Hz envelopes -> normalization (future analog-envelope workload)"
            ),
            source_sample_rate_hz=48_000 if system == "fabricated_400hz" else 400,
            feature_rate_hz=400,
            channels=5,
            model=model,
            mean=np.asarray(checkpoint["input_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["input_std"], dtype=np.float32),
            architecture=architecture,
            checkpoint_metadata={
                "feature_contract": checkpoint.get("feature_contract"),
                "feature_names": checkpoint.get("feature_names"),
                "context_ms": checkpoint.get("context_ms"),
                "epoch": checkpoint.get("epoch"),
            },
        )
    elif system in ("fabricated_400hz_baselines", "fabricated_400hz_baselines_post_envelope"):
        architecture = str(checkpoint.get("architecture", ""))
        expected_base = model_name or dict(checkpoint.get("model_config", {})).get(
            "model_name"
        )
        if expected_base in CNN_BASELINE_FABRICATED_FACTORIES:
            factories = CNN_BASELINE_FABRICATED_FACTORIES
            architecture_ids = CNN_BASELINE_ARCHITECTURE_IDS
            architecture_label = "literature-minimal CNN baseline"
        else:
            factories = FABRICATED_400_BASELINE_FACTORIES
            architecture_ids = FABRICATED_400_BASELINE_ARCHITECTURE_IDS
            architecture_label = "matched compact architecture"
        expected_architecture = architecture_ids.get(expected_base)
        if expected_base not in factories or architecture != expected_architecture:
            raise NotImplementedError(
                f"Unsupported or mismatched fabricated-baseline architecture: "
                f"requested={expected_base!r}, checkpoint={architecture!r}. Add its exact "
                "constructor to the fabricated_400hz_baselines branch of load_model()."
            )
        model = factories[expected_base]()
        model.load_state_dict(_checkpoint_state(checkpoint, "model_state"), strict=True)
        is_post_envelope = system == "fabricated_400hz_baselines_post_envelope"
        loaded = LoadedSystem(
            name=f"{system}/{expected_base}",
            representation=(
                (
                    "precomputed five-channel 400 Hz AC-RMS envelopes (hardware-native; "
                    "no digital downsampling) "
                    f"-> {expected_base} ({architecture_label}, masked-mean endpoint)"
                )
                if is_post_envelope
                else (
                    "high-rate five-channel DAQ waveform -> digital 10 ms AC-RMS at 400 Hz "
                    f"-> {expected_base} ({architecture_label}, masked-mean endpoint)"
                )
            ),
            source_sample_rate_hz=400 if is_post_envelope else 48_000,
            feature_rate_hz=400,
            channels=5,
            model=model,
            mean=np.asarray(checkpoint["input_mean"], dtype=np.float32),
            std=np.asarray(checkpoint["input_std"], dtype=np.float32),
            architecture=architecture,
            checkpoint_metadata={
                "feature_contract": checkpoint.get("feature_contract"),
                "feature_names": checkpoint.get("feature_names"),
                "context_ms": checkpoint.get("context_ms"),
                "epoch": checkpoint.get("epoch"),
            },
        )
    else:
        raise ValueError(f"Unknown system: {system}")

    if np.any(~np.isfinite(loaded.mean)) or np.any(loaded.std <= 0):
        raise ValueError(f"Invalid normalization statistics in {checkpoint_path}")
    loaded.model.cpu().eval()
    for parameter in loaded.model.parameters():
        parameter.requires_grad_(False)
    return loaded


# ---------------------------------------------------------------------------
# Input inventory and in-memory loading
# ---------------------------------------------------------------------------
def canonical_sample_id(speaker: int | str, digit: int, repetition: int) -> str:
    return f"speaker={int(speaker):02d}|digit={int(digit)}|repetition={int(repetition)}"


def _speaker_set_from_json(path: Path) -> set[int] | None:
    document = json.loads(path.read_text(encoding="utf-8"))
    splits = document.get("speaker_splits")
    if isinstance(splits, dict) and "test" in splits:
        return {int(value) for value in splits["test"]}
    return None


def _raw_manifest_rows(test_split_path: Path, dataset_root: Path) -> list[dict[str, Any]]:
    if test_split_path.suffix.lower() == ".json":
        speakers = _speaker_set_from_json(test_split_path)
        if speakers is None:
            document = json.loads(test_split_path.read_text(encoding="utf-8"))
            source_rows = document.get("test", document.get("samples"))
            if not isinstance(source_rows, list):
                raise ValueError(
                    "JSON test split must contain speaker_splits.test or a test/samples list"
                )
            return [dict(row) for row in source_rows]
        rows = []
        for path in sorted((dataset_root / "data").glob("[0-9][0-9]/*.wav")):
            digit, speaker, repetition = (int(value) for value in path.stem.split("_"))
            if speaker in speakers:
                rows.append(
                    {
                        "path": str(path),
                        "label": digit,
                        "speaker": speaker,
                        "repetition": repetition,
                    }
                )
        return rows

    with test_split_path.open("r", encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    if rows and "split" in rows[0]:
        rows = [row for row in rows if str(row["split"]).lower() == "test"]
    return rows


def _resolve_raw_path(row: dict[str, Any], dataset_root: Path) -> Path:
    for key in ("path", "wav_path", "raw_path"):
        raw = row.get(key)
        if raw:
            path = Path(str(raw))
            if not path.is_absolute():
                path = dataset_root / path
            elif not path.is_file():
                # Colab manifests often contain stale absolute /content paths.
                speaker = int(row.get("speaker", row.get("speaker_id")))
                path = dataset_root / "data" / f"{speaker:02d}" / path.name
            return path
    speaker = int(row.get("speaker", row.get("speaker_id")))
    digit = int(row.get("label", row.get("digit")))
    repetition = int(row["repetition"])
    return dataset_root / "data" / f"{speaker:02d}" / f"{digit}_{speaker:02d}_{repetition}.wav"


def _resample_poly(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return values.astype(np.float32, copy=False)
    try:
        from scipy import signal
    except ImportError as error:
        raise ImportError("Exact polyphase resampling requires scipy") from error
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(
        values, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


def load_raw_test_examples(
    test_split_path: Path,
    dataset_root: Path,
    preload_sample_rate: int | None = None,
) -> list[RawExample]:
    try:
        import soundfile as sf
    except ImportError as error:
        raise ImportError("WAV loading requires the soundfile package") from error
    examples: list[RawExample] = []
    rows = _raw_manifest_rows(test_split_path, dataset_root)
    for row in rows:
        wav_path = _resolve_raw_path(row, dataset_root)
        values, sample_rate = sf.read(wav_path, always_2d=False, dtype="float32")
        values = np.asarray(values, dtype=np.float32)
        if values.ndim == 2:
            values = values.mean(axis=1, dtype=np.float32)
        if values.ndim != 1:
            raise ValueError(f"Expected mono waveform: {wav_path}")
        if preload_sample_rate is not None:
            values = _resample_poly(values, int(sample_rate), preload_sample_rate)
            sample_rate = preload_sample_rate
            # Match the frozen 16 kHz evaluation cache's waveform precision. This
            # conversion is deliberately outside the timed frontend.
            values = values.astype(np.float16).astype(np.float32)
        stem_digit, stem_speaker, stem_repetition = (
            int(value) for value in wav_path.stem.split("_")
        )
        label = int(row.get("label", row.get("digit", stem_digit)))
        speaker = int(row.get("speaker", row.get("speaker_id", stem_speaker)))
        repetition = int(row.get("repetition", stem_repetition))
        examples.append(
            RawExample(
                sample_id=canonical_sample_id(speaker, label, repetition),
                waveform=np.ascontiguousarray(values),
                source_rate=int(sample_rate),
                label=label,
            )
        )
    _validate_examples(examples, "raw AudioMNIST")
    return sorted(examples, key=lambda item: item.sample_id)


def _npz_key(archive: Any, *names: str) -> str:
    for name in names:
        if name in archive.files:
            return name
    raise KeyError(f"NPZ requires one of {names}; found {archive.files}")


def _load_fabricated_npz(path: Path) -> list[FabricatedExample]:
    with np.load(path, allow_pickle=True) as archive:
        feature_key = _npz_key(archive, "features", "values", "envelopes")
        label_key = _npz_key(archive, "labels", "digits", "y")
        raw_features = archive[feature_key]
        labels = np.asarray(archive[label_key], dtype=np.int64)
        lengths = (
            np.asarray(archive["lengths"], dtype=np.int64)
            if "lengths" in archive.files
            else None
        )
        speakers = archive["speakers"] if "speakers" in archive.files else None
        repetitions = archive["repetitions"] if "repetitions" in archive.files else None
        sample_ids = archive["sample_ids"] if "sample_ids" in archive.files else None
        durations = (
            np.asarray(archive["audio_duration_seconds"], dtype=np.float64)
            if "audio_duration_seconds" in archive.files
            else None
        )

        examples = []
        for index, label in enumerate(labels):
            values = np.asarray(raw_features[index], dtype=np.float32)
            if lengths is not None:
                values = values[: int(lengths[index])]
            if values.ndim != 2 or values.shape[1] != 5:
                raise ValueError(
                    f"Fabricated sample {index} must have shape [steps, 5], got {values.shape}"
                )
            if sample_ids is not None:
                sample_id = str(sample_ids[index])
            elif speakers is not None and repetitions is not None:
                sample_id = canonical_sample_id(speakers[index], int(label), repetitions[index])
            else:
                sample_id = f"fabricated_index={index:06d}"
            duration = float(durations[index]) if durations is not None else len(values) / 400.0
            examples.append(
                FabricatedExample(
                    sample_id=sample_id,
                    features=np.ascontiguousarray(values),
                    label=int(label),
                    audio_duration_seconds=duration,
                )
            )
    return examples


def _load_fabricated_envelope_capture_root(
    capture_root: Path, test_split_path: Path, context_ms: float
) -> list[FabricatedExample]:
    try:
        from train_envelope_400hz_5d_pooled_tcn_classifier import build_envelope_split
    except ImportError as error:
        raise ImportError(
            "Capture-root loading requires this repository's fabricated AudioMNIST modules. "
            "Alternatively export a consolidated NPZ with features, labels, speakers, and repetitions."
        ) from error

    stimuli_root = test_split_path.parent
    if test_split_path.name.startswith("audiomnist_test_manifest"):
        stimuli_root = test_split_path.parent
    if capture_root.name == "test":
        capture_root = capture_root.parent
    cache_root = RESULTS_DIR / ".fabricated_test_cache"
    source, _, _ = build_envelope_split(
        "test",
        capture_root,
        stimuli_root,
        cache_root,
        context_ms,
        True,
    )
    return [
        FabricatedExample(
            sample_id=canonical_sample_id(item.speaker_id, item.label, item.repetition),
            features=np.ascontiguousarray(item.values, dtype=np.float32),
            label=int(item.label),
            audio_duration_seconds=len(item.values) / 400.0,
        )
        for item in source
    ]


def _expected_test_recording_ids(stimuli_root: Path) -> set[int]:
    manifest_path = stimuli_root / "audiomnist_test_manifest.csv"
    with manifest_path.open("r", encoding="utf-8", newline="") as handle:
        return {int(row["recording_id"]) for row in csv.DictReader(handle)}


def _raw_capture_directories(capture_root: Path, stimuli_root: Path) -> list[Path]:
    if capture_root.name == "test":
        capture_root = capture_root.parent
    expected = _expected_test_recording_ids(stimuli_root)
    by_recording: dict[int, Path] = {}
    for metadata_path in sorted((capture_root / "test").rglob("metadata.json")):
        run_dir = metadata_path.parent
        if not (run_dir / "raw_capture.npz").is_file():
            continue
        if not (run_dir / "aligned_events.csv").is_file():
            continue
        if not (run_dir / "envelopes_400Hz_annotated.csv").is_file():
            raise FileNotFoundError(
                f"{run_dir}: saved 400 Hz envelope timestamps are required to reproduce test segmentation"
            )
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        recording_id = int(metadata.get("manifest_recording_id", -1))
        if recording_id not in expected:
            continue
        if recording_id in by_recording:
            raise RuntimeError(f"Duplicate fabricated capture for recording {recording_id}")
        by_recording[recording_id] = run_dir
    missing = sorted(expected - set(by_recording))
    if missing:
        raise RuntimeError(
            f"Missing {len(missing)} fabricated test captures; first IDs: {missing[:12]}"
        )
    return [by_recording[index] for index in sorted(by_recording)]


def load_fabricated_raw_test_examples(
    capture_root: Path, test_split_path: Path, context_ms: float
) -> list[FabricatedRawExample]:
    """Preload exact high-rate five-channel spans used by held-out evaluation."""
    if not capture_root.is_dir():
        raise ValueError(
            "The primary fabricated benchmark requires FABRICATED_FEATURE_PATH to "
            "point to the capture root containing test/*/raw_capture.npz"
        )
    stimuli_root = test_split_path.parent
    permitted_speakers = _speaker_set_from_json(stimuli_root / "protocol.json")
    if permitted_speakers is None:
        raise ValueError("The fabricated capture loader requires stimuli/protocol.json")

    examples: list[FabricatedRawExample] = []
    run_dirs = _raw_capture_directories(capture_root, stimuli_root)
    for run_index, run_dir in enumerate(run_dirs, start=1):
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
        speaker = int(metadata["dataset_speaker_id"])
        if speaker not in permitted_speakers:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker:02d} is not in the test split")
        with np.load(run_dir / "raw_capture.npz", allow_pickle=False) as archive:
            volts = np.asarray(archive["volts_v"], dtype=np.float32)
            source_rate = float(archive["sample_rate_hz"])
        if volts.ndim != 2 or volts.shape[1] != 5:
            raise ValueError(f"{run_dir.name}: expected [samples, 5] volts, got {volts.shape}")

        envelope_metadata = json.loads(
            (run_dir / "envelopes_400Hz_metadata.json").read_text(encoding="ascii")
        )
        if not math.isclose(float(envelope_metadata["envelope_rate_hz"]), 400.0):
            raise RuntimeError(f"{run_dir.name}: saved envelope rate is not 400 Hz")
        if not math.isclose(float(envelope_metadata["rms_window_duration_ms"]), 10.0):
            raise RuntimeError(f"{run_dir.name}: saved RMS window is not 10 ms")
        frame_length = int(round(source_rate * 0.010))
        hop_length = int(round(source_rate / 400.0))

        with (run_dir / "envelopes_400Hz_annotated.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            saved_envelope_rows = list(csv.DictReader(handle))
        center_times = np.asarray(
            [float(row["time_s"]) for row in saved_envelope_rows], dtype=np.float64
        )
        availability_times = center_times + 0.005
        expected_rows = (len(volts) - frame_length) // hop_length + 1
        if len(availability_times) != expected_rows:
            raise RuntimeError(
                f"{run_dir.name}: raw/envelope row mismatch "
                f"({expected_rows} expected, {len(availability_times)} saved)"
            )

        with (run_dir / "aligned_events.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            events = [
                row for row in csv.DictReader(handle) if str(row["event_type"]) == "letter"
            ]
        for event_index, event in enumerate(
            sorted(events, key=lambda row: float(row["start_s"]))
        ):
            label = int(event["label"])
            start_s = max(float(availability_times[0]), float(event["start_s"]) - context_ms / 1_000.0)
            end_s = min(
                float(availability_times[-1]) + 1.0 / 400.0,
                float(event["end_s"]) + context_ms / 1_000.0,
            )
            first = int(np.searchsorted(availability_times, start_s, side="left"))
            last = int(np.searchsorted(availability_times, end_s, side="left"))
            if last <= first:
                raise RuntimeError(f"{run_dir.name}: empty raw segment for digit {label}")
            raw_first = first * hop_length
            raw_last = (last - 1) * hop_length + frame_length
            segment = np.ascontiguousarray(volts[raw_first:raw_last], dtype=np.float32)
            if (len(segment) - frame_length) // hop_length + 1 != last - first:
                raise RuntimeError(f"{run_dir.name}: raw/envelope segment length mismatch")
            if event_index == 0:
                columns = (
                    "AI0_R300_rms_v",
                    "AI1_R3500_rms_v",
                    "AI2_R7500_rms_v",
                    "AI3_R1800_rms_v",
                    "AI4_R4500_rms_v",
                )
                saved = np.asarray(
                    [
                        [float(saved_envelope_rows[row][column]) for column in columns]
                        for row in range(first, last)
                    ],
                    dtype=np.float32,
                )
                recomputed = _frame_ac_rms_400hz(segment, source_rate)
                if not np.allclose(recomputed, saved, rtol=2e-6, atol=1e-8):
                    difference = float(np.max(np.abs(recomputed - saved)))
                    raise RuntimeError(
                        f"{run_dir.name}: raw AC-RMS does not reproduce saved features; "
                        f"max abs difference={difference:.3g}"
                    )
            repetition = int(float(event["source_repetition"]))
            examples.append(
                FabricatedRawExample(
                    sample_id=canonical_sample_id(speaker, label, repetition),
                    waveform=segment,
                    source_rate=source_rate,
                    label=label,
                    audio_duration_seconds=(last - first) / 400.0,
                )
            )
        print(
            f"loaded fabricated raw test capture [{run_index:03d}/{len(run_dirs):03d}] "
            f"{run_dir.name}",
            flush=True,
        )

    _validate_examples(examples, "fabricated raw AudioMNIST")
    return sorted(examples, key=lambda item: item.sample_id)


def load_fabricated_baseline_envelope_test_examples(
    capture_root: Path, test_split_path: Path, context_ms: float
) -> list[FabricatedExample]:
    """Preload the saved 400 Hz five-channel AC-RMS envelope rows directly.

    Unlike load_fabricated_raw_test_examples, this does not recompute AC-RMS
    from the raw waveform at benchmark time -- that downsampling step belongs
    to the analog front-end hardware in the final design, not the Pi's timed
    inference loop. This reuses the same event-segmentation logic so the two
    loaders select identical spans, but slices envelopes_400Hz_capture.npz
    values straight into FabricatedExample.features.
    """
    if not capture_root.is_dir():
        raise ValueError(
            "fabricated_400hz_baselines_post_envelope requires FABRICATED_FEATURE_PATH "
            "to point to the capture root containing test/*/envelopes_400Hz_capture.npz"
        )
    stimuli_root = test_split_path.parent
    permitted_speakers = _speaker_set_from_json(stimuli_root / "protocol.json")
    if permitted_speakers is None:
        raise ValueError("The fabricated capture loader requires stimuli/protocol.json")

    examples: list[FabricatedExample] = []
    run_dirs = _raw_capture_directories(capture_root, stimuli_root)
    for run_index, run_dir in enumerate(run_dirs, start=1):
        metadata = json.loads((run_dir / "metadata.json").read_text(encoding="ascii"))
        speaker = int(metadata["dataset_speaker_id"])
        if speaker not in permitted_speakers:
            raise RuntimeError(f"{run_dir.name}: speaker {speaker:02d} is not in the test split")

        envelope_metadata = json.loads(
            (run_dir / "envelopes_400Hz_metadata.json").read_text(encoding="ascii")
        )
        if not math.isclose(float(envelope_metadata["envelope_rate_hz"]), 400.0):
            raise RuntimeError(f"{run_dir.name}: saved envelope rate is not 400 Hz")
        if not math.isclose(float(envelope_metadata["rms_window_duration_ms"]), 10.0):
            raise RuntimeError(f"{run_dir.name}: saved RMS window is not 10 ms")

        envelope_npz_path = run_dir / "envelopes_400Hz_capture.npz"
        if not envelope_npz_path.is_file():
            raise FileNotFoundError(
                f"{run_dir.name}: missing envelopes_400Hz_capture.npz -- run "
                "convert_envelopes_400hz_to_npz.py first"
            )
        with np.load(envelope_npz_path, allow_pickle=False) as archive:
            envelope_values = np.asarray(archive["envelope_v"], dtype=np.float32)
            center_times = np.asarray(archive["time_s"], dtype=np.float64)
            if not math.isclose(float(archive["sample_rate_hz"]), 400.0):
                raise RuntimeError(f"{run_dir.name}: envelope npz sample_rate_hz is not 400 Hz")
        if envelope_values.ndim != 2 or envelope_values.shape[1] != 5:
            raise ValueError(
                f"{run_dir.name}: expected [rows, 5] envelope_v, got {envelope_values.shape}"
            )
        availability_times = center_times + 0.005

        with (run_dir / "aligned_events.csv").open(
            "r", encoding="utf-8", newline=""
        ) as handle:
            events = [
                row for row in csv.DictReader(handle) if str(row["event_type"]) == "letter"
            ]
        for event in sorted(events, key=lambda row: float(row["start_s"])):
            label = int(event["label"])
            start_s = max(float(availability_times[0]), float(event["start_s"]) - context_ms / 1_000.0)
            end_s = min(
                float(availability_times[-1]) + 1.0 / 400.0,
                float(event["end_s"]) + context_ms / 1_000.0,
            )
            first = int(np.searchsorted(availability_times, start_s, side="left"))
            last = int(np.searchsorted(availability_times, end_s, side="left"))
            if last <= first:
                raise RuntimeError(f"{run_dir.name}: empty envelope segment for digit {label}")
            features = np.ascontiguousarray(envelope_values[first:last], dtype=np.float32)
            repetition = int(float(event["source_repetition"]))
            examples.append(
                FabricatedExample(
                    sample_id=canonical_sample_id(speaker, label, repetition),
                    features=features,
                    label=label,
                    audio_duration_seconds=(last - first) / 400.0,
                )
            )
        print(
            f"loaded fabricated envelope test capture [{run_index:03d}/{len(run_dirs):03d}] "
            f"{run_dir.name}",
            flush=True,
        )

    _validate_examples(examples, "fabricated envelope AudioMNIST")
    return sorted(examples, key=lambda item: item.sample_id)


def load_fabricated_test_examples(
    feature_path: Path, test_split_path: Path, context_ms: float
) -> list[FabricatedExample]:
    if feature_path.is_file() and feature_path.suffix.lower() == ".npz":
        examples = _load_fabricated_npz(feature_path)
    elif feature_path.is_dir():
        consolidated = feature_path / "fabricated_test_features.npz"
        examples = (
            _load_fabricated_npz(consolidated)
            if consolidated.is_file()
            else _load_fabricated_envelope_capture_root(
                feature_path, test_split_path, context_ms
            )
        )
    else:
        raise FileNotFoundError(f"Fabricated feature path not found: {feature_path}")
    _validate_examples(examples, "fabricated AudioMNIST")
    return sorted(examples, key=lambda item: item.sample_id)


def _validate_examples(examples: Sequence[Any], description: str) -> None:
    if not examples:
        raise ValueError(f"No {description} test examples were loaded")
    if len(examples) != EXPECTED_TEST_EXAMPLES:
        raise ValueError(
            f"Expected the complete {EXPECTED_TEST_EXAMPLES:,}-utterance fixed test set for "
            f"{description}, found {len(examples):,}"
        )
    ids = [item.sample_id for item in examples]
    if len(set(ids)) != len(ids):
        raise ValueError(f"Duplicate sample IDs in {description}")
    labels = {int(item.label) for item in examples}
    if not labels.issubset(set(range(10))):
        raise ValueError(f"Invalid labels in {description}: {sorted(labels)}")


def sample_identity_sha256(examples: Sequence[Any]) -> str:
    rows = sorted(f"{item.sample_id}|label={int(item.label)}" for item in examples)
    return hashlib.sha256("\n".join(rows).encode("utf-8")).hexdigest()


# ---------------------------------------------------------------------------
# Timed preprocessing functions. File I/O and transform construction are not
# present here. Each function returns a batch-one CPU model input.
# ---------------------------------------------------------------------------
class LogMelFrontend:
    def __init__(self, mean: np.ndarray, std: np.ndarray) -> None:
        try:
            import torchaudio
        except ImportError as error:
            raise ImportError(
                "The exact 16 kHz frontend requires torchaudio built for this PyTorch version"
            ) from error
        self.mean = torch.from_numpy(np.asarray(mean, dtype=np.float32)).view(1, 1, 40)
        self.std = torch.from_numpy(np.asarray(std, dtype=np.float32)).view(1, 1, 40)
        self.mel = torchaudio.transforms.MelSpectrogram(
            sample_rate=16_000,
            n_fft=800,
            win_length=800,
            hop_length=160,
            f_min=20.0,
            f_max=8_000.0,
            n_mels=40,
            power=2.0,
            center=False,
            norm=None,
            mel_scale="htk",
        ).cpu().eval()


def _peak_normalize(values: np.ndarray) -> np.ndarray:
    peak = float(np.max(np.abs(values))) if len(values) else 0.0
    if peak <= 1e-12:
        return values.astype(np.float32, copy=False)
    target = 10.0 ** (TARGET_PEAK_DBFS / 20.0)
    return np.asarray(values * (target / peak), dtype=np.float32)


def _crop_or_pad(values: np.ndarray, target_length: int) -> np.ndarray:
    if len(values) > target_length:
        start = (len(values) - target_length) // 2
        return np.asarray(values[start : start + target_length], dtype=np.float32)
    if len(values) < target_length:
        missing = target_length - len(values)
        return np.pad(values, (missing // 2, missing - missing // 2)).astype(np.float32)
    return values.astype(np.float32, copy=False)


def preprocess_logmel(example: RawExample, frontend: LogMelFrontend) -> PreparedInput:
    values = example.waveform
    if example.source_rate != 16_000:
        raise ValueError(
            "digital_16k inputs must be pre-resampled to 16 kHz before timing; "
            f"got {example.source_rate} Hz for {example.sample_id}"
        )
    values = _crop_or_pad(_peak_normalize(values), 16_000)
    waveform = torch.from_numpy(np.ascontiguousarray(values)).unsqueeze(0)
    feature = torch.log(frontend.mel(waveform).clamp_min(1e-8)).transpose(1, 2)
    feature = feature.to(torch.float16).to(torch.float32)
    feature = (feature - frontend.mean) / frontend.std
    return PreparedInput((feature,))


def preprocess_400hz_envelope(
    example: RawExample, mean: np.ndarray, std: np.ndarray
) -> PreparedInput:
    if example.source_rate != 48_000:
        raise ValueError(
            "The frozen 400 Hz digital-envelope experiment requires 48 kHz source WAVs; "
            f"got {example.source_rate} Hz for {example.sample_id}"
        )
    fixed = _crop_or_pad(_peak_normalize(example.waveform), 48_000)
    window_samples = 480  # Exact 10 ms trailing window at 48 kHz.
    end_indices = np.rint(
        (np.arange(400, dtype=np.float64) + 1.0) * 48_000 / 400
    ).astype(np.int64)
    end_indices = np.clip(end_indices, 1, len(fixed))
    start_indices = np.maximum(0, end_indices - window_samples)
    squared = np.square(fixed, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = cumulative[end_indices] - cumulative[start_indices]
    envelope = np.sqrt(np.maximum(energy, 0.0) / window_samples).astype(np.float32)
    normalized = (envelope - np.float32(mean[0])) / np.float32(std[0])
    tensor = torch.from_numpy(np.ascontiguousarray(normalized[:, None])).unsqueeze(0)
    return PreparedInput((tensor,))


def preprocess_fabricated(
    example: FabricatedExample, mean: np.ndarray, std: np.ndarray
) -> PreparedInput:
    # This is the complete digital post-processing used by the frozen dataset:
    # channel-wise train-only standardization and tensor/length construction.
    normalized = (example.features - mean.reshape(1, 5)) / std.reshape(1, 5)
    values = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([len(example.features)], dtype=torch.long)
    return PreparedInput((values, lengths))


def _frame_ac_rms_400hz(values: np.ndarray, sample_rate_hz: float) -> np.ndarray:
    """Exact five-channel implementation used by build_400hz_envelopes.py."""
    frame_length = max(1, int(round(sample_rate_hz * 10.0 / 1_000.0)))
    hop_length = max(1, int(round(sample_rate_hz / 400.0)))
    if len(values) < frame_length:
        return np.empty((0, values.shape[1]), dtype=np.float32)

    starts = np.arange(0, len(values) - frame_length + 1, hop_length, dtype=np.int64)
    ends = starts + frame_length
    channels = []
    for channel in range(values.shape[1]):
        values64 = np.asarray(values[:, channel], dtype=np.float64)
        cumulative = np.empty(len(values64) + 1, dtype=np.float64)
        cumulative[0] = 0.0
        np.cumsum(values64, out=cumulative[1:])
        sums = cumulative[ends] - cumulative[starts]

        np.square(values64, out=values64)
        cumulative[0] = 0.0
        np.cumsum(values64, out=cumulative[1:])
        sums_square = cumulative[ends] - cumulative[starts]
        variance = np.maximum(
            sums_square / frame_length - np.square(sums / frame_length), 0.0
        )
        channels.append(np.sqrt(variance).astype(np.float32))
    return np.column_stack(channels)


def preprocess_fabricated_raw(
    example: FabricatedRawExample, mean: np.ndarray, std: np.ndarray
) -> PreparedInput:
    envelope = _frame_ac_rms_400hz(example.waveform, example.source_rate)
    normalized = (envelope - mean.reshape(1, 5)) / std.reshape(1, 5)
    values = torch.from_numpy(np.ascontiguousarray(normalized, dtype=np.float32)).unsqueeze(0)
    lengths = torch.tensor([len(envelope)], dtype=torch.long)
    return PreparedInput((values, lengths))


def _preprocessor(loaded: LoadedSystem) -> Callable[[Any], PreparedInput]:
    if loaded.name.startswith("digital_16k/"):
        frontend = LogMelFrontend(loaded.mean, loaded.std)
        return lambda example: preprocess_logmel(example, frontend)
    if loaded.name.startswith("digital_envelope_400hz/"):
        return lambda example: preprocess_400hz_envelope(example, loaded.mean, loaded.std)
    if loaded.name.startswith("fabricated_400hz_baselines_post_envelope/"):
        return lambda example: preprocess_fabricated(example, loaded.mean, loaded.std)
    if loaded.name == "fabricated_400hz" or loaded.name.startswith(
        "fabricated_400hz_baselines/"
    ):
        return lambda example: preprocess_fabricated_raw(example, loaded.mean, loaded.std)
    return lambda example: preprocess_fabricated(example, loaded.mean, loaded.std)


def _forward(model: nn.Module, prepared: PreparedInput) -> torch.Tensor:
    output = model(*prepared.args)
    if not isinstance(output, torch.Tensor) or output.ndim != 2 or output.shape[0] != 1:
        raise TypeError(f"Expected batch-one class logits, got {type(output)}")
    return output


def _profile_input(loaded: LoadedSystem, example: Any) -> PreparedInput:
    if loaded.name.startswith("digital_16k/"):
        return PreparedInput((torch.zeros(1, 96, 40),))
    if loaded.name.startswith("digital_envelope_400hz/"):
        return PreparedInput((torch.zeros(1, 400, 1),))
    if loaded.name == "fabricated_400hz" or loaded.name.startswith(
        "fabricated_400hz_baselines/"
    ):
        frame_length = int(round(example.source_rate * 0.010))
        hop_length = int(round(example.source_rate / 400.0))
        steps = (len(example.waveform) - frame_length) // hop_length + 1
    else:
        steps = len(example.features)
    return PreparedInput(
        (torch.zeros(1, steps, 5), torch.tensor([steps], dtype=torch.long))
    )


@torch.no_grad()
def evaluate_accuracy(
    model: nn.Module,
    examples: Sequence[Any],
    preprocess: Callable[[Any], PreparedInput],
) -> tuple[float, list[int]]:
    model.eval()
    predictions = []
    for example in examples:
        prediction = int(_forward(model, preprocess(example)).argmax(dim=1).item())
        predictions.append(prediction)
    correct = sum(prediction == int(example.label) for prediction, example in zip(predictions, examples))
    return 100.0 * correct / len(examples), predictions


# ---------------------------------------------------------------------------
# NN-only MAC profiling. Hooks count multiply-accumulates from actual output
# shapes for Conv/Linear and recurrent modules. Preprocessing never enters here.
# ---------------------------------------------------------------------------
def profile_model(model: nn.Module, prepared: PreparedInput) -> int:
    macs = 0
    handles = []

    def hook(module: nn.Module, inputs: tuple[Any, ...], output: Any) -> None:
        nonlocal macs
        tensor = output[0] if isinstance(output, tuple) else output
        if not isinstance(tensor, torch.Tensor):
            return
        if isinstance(module, (nn.Conv1d, nn.Conv2d, nn.Conv3d)):
            kernel = math.prod(module.kernel_size)
            macs += int(tensor.numel() * kernel * module.in_channels / module.groups)
        elif isinstance(module, nn.Linear):
            macs += int(tensor.numel() * module.in_features)
        elif isinstance(module, (nn.GRU, nn.LSTM, nn.RNN)):
            sequence = inputs[0]
            if module.batch_first:
                batch, steps = int(sequence.shape[0]), int(sequence.shape[1])
            else:
                steps, batch = int(sequence.shape[0]), int(sequence.shape[1])
            gates = 3 if isinstance(module, nn.GRU) else 4 if isinstance(module, nn.LSTM) else 1
            directions = 2 if module.bidirectional else 1
            input_size = module.input_size
            for _ in range(module.num_layers):
                macs += steps * batch * directions * gates * (
                    input_size * module.hidden_size + module.hidden_size * module.hidden_size
                )
                input_size = module.hidden_size * directions

    handled = (nn.Conv1d, nn.Conv2d, nn.Conv3d, nn.Linear, nn.GRU, nn.LSTM, nn.RNN)
    for module in model.modules():
        if isinstance(module, handled):
            handles.append(module.register_forward_hook(hook))
    try:
        with torch.no_grad():
            _forward(model, prepared)
    finally:
        for handle in handles:
            handle.remove()
    if macs <= 0:
        raise RuntimeError("MAC profiler found no supported neural-network operations")
    return int(macs)


def percentile(values: Sequence[float], quantile: float) -> float:
    return float(np.percentile(np.asarray(values, dtype=np.float64), quantile))


def scheduled_example_index(run_index: int, runs: int, example_count: int) -> int:
    if runs <= example_count:
        return min((run_index * example_count) // runs, example_count - 1)
    return run_index % example_count


@torch.no_grad()
def benchmark_pipeline(
    model: nn.Module,
    examples: Sequence[Any],
    preprocess: Callable[[Any], PreparedInput],
    warmup: int,
    runs: int,
) -> tuple[list[dict[str, Any]], float]:
    """Return raw wall-clock timings and aggregate RTF."""
    model.eval()
    for index in range(warmup):
        example_index = scheduled_example_index(index, warmup, len(examples))
        _forward(model, preprocess(examples[example_index]))

    rows = []
    total_processing_ns = 0
    total_audio_seconds = 0.0
    for run_index in range(runs):
        example_index = scheduled_example_index(run_index, runs, len(examples))
        example = examples[example_index]

        frontend_start = time.perf_counter_ns()
        prepared = preprocess(example)
        frontend_stop = time.perf_counter_ns()
        nn_start = time.perf_counter_ns()
        logits = _forward(model, prepared)
        nn_stop = time.perf_counter_ns()
        prediction = int(logits.argmax(dim=1).item())

        frontend_ns = frontend_stop - frontend_start
        nn_ns = nn_stop - nn_start
        total_ns = frontend_ns + nn_ns
        total_processing_ns += total_ns
        total_audio_seconds += float(example.audio_duration_seconds)
        rows.append(
            {
                "run_index": run_index,
                "sample_index": example_index,
                "sample_id": example.sample_id,
                "label": int(example.label),
                "prediction": prediction,
                "audio_duration_seconds": float(example.audio_duration_seconds),
                "frontend_ms": frontend_ns / 1e6,
                "nn_ms": nn_ns / 1e6,
                "total_ms": total_ns / 1e6,
            }
        )
    rtf = (total_processing_ns / 1e9) / total_audio_seconds
    return rows, rtf


@torch.no_grad()
def measure_cpu_time_ms(
    model: nn.Module,
    examples: Sequence[Any],
    preprocess: Callable[[Any], PreparedInput],
    runs: int,
) -> float:
    """Measure frontend plus NN process CPU time without latency-probe overhead."""
    process = psutil.Process()
    cpu_before = process.cpu_times()
    for run_index in range(runs):
        example_index = scheduled_example_index(run_index, runs, len(examples))
        _forward(model, preprocess(examples[example_index]))
    cpu_after = process.cpu_times()
    process_cpu_seconds = (cpu_after.user + cpu_after.system) - (
        cpu_before.user + cpu_before.system
    )
    return 1_000.0 * process_cpu_seconds / runs


@torch.no_grad()
def measure_peak_rss_mb(
    model: nn.Module,
    examples: Sequence[Any],
    preprocess: Callable[[Any], PreparedInput],
    runs: int = 100,
) -> tuple[float, float, float]:
    """Return incremental peak, post-load baseline, and absolute peak RSS."""
    process = psutil.Process()
    stop = threading.Event()
    peak = [0]

    def sample() -> None:
        while not stop.wait(0.001):
            peak[0] = max(peak[0], process.memory_info().rss)

    monitor = threading.Thread(target=sample, daemon=True)
    monitor.start()
    time.sleep(0.01)
    baseline = process.memory_info().rss
    peak[0] = baseline
    try:
        for index in range(runs):
            example_index = scheduled_example_index(index, runs, len(examples))
            _forward(model, preprocess(examples[example_index]))
            peak[0] = max(peak[0], process.memory_info().rss)
    finally:
        stop.set()
        monitor.join()
    scale = 1024.0**2
    return max(0, peak[0] - baseline) / scale, baseline / scale, peak[0] / scale


def _read_text(path: Path) -> str | None:
    try:
        return path.read_text(encoding="utf-8", errors="replace").strip("\x00\n ")
    except OSError:
        return None


def _cpu_governors() -> list[str]:
    values = {
        value
        for path in Path("/sys/devices/system/cpu").glob("cpu[0-9]*/cpufreq/scaling_governor")
        if (value := _read_text(path))
    }
    return sorted(values)


def _frequency_values(name: str) -> list[float]:
    values = []
    for path in Path("/sys/devices/system/cpu").glob(f"cpu[0-9]*/cpufreq/{name}"):
        raw = _read_text(path)
        if raw:
            values.append(float(raw) / 1_000.0)  # sysfs kHz to MHz
    return values


def _cpu_description() -> str:
    cpuinfo = _read_text(Path("/proc/cpuinfo")) or ""
    for key in ("model name", "Model", "Hardware", "Processor"):
        for line in cpuinfo.splitlines():
            if line.lower().startswith(key.lower() + "\t") or line.lower().startswith(key.lower() + " "):
                return line.split(":", 1)[-1].strip()
    return platform.processor() or platform.machine()


def system_information() -> dict[str, Any]:
    current = _frequency_values("scaling_cur_freq")
    maximum = _frequency_values("cpuinfo_max_freq") or _frequency_values("scaling_max_freq")
    temperature_raw = _read_text(Path("/sys/class/thermal/thermal_zone0/temp"))
    temperature_c = float(temperature_raw) / 1_000.0 if temperature_raw else None
    virtual_memory = psutil.virtual_memory()
    return {
        "raspberry_pi_model": _read_text(Path("/proc/device-tree/model")) or "not detected",
        "cpu": _cpu_description(),
        "architecture": platform.machine(),
        "platform": platform.platform(),
        "kernel": platform.release(),
        "python_version": platform.python_version(),
        "pytorch_version": torch.__version__,
        "cpu_governor": _cpu_governors(),
        "current_cpu_frequency_mhz": current,
        "max_cpu_frequency_mhz": maximum,
        "os_cpu_thread_count": os.cpu_count(),
        "physical_cpu_count": psutil.cpu_count(logical=False),
        "logical_cpu_count": psutil.cpu_count(logical=True),
        "torch_threads": torch.get_num_threads(),
        "torch_interop_threads": torch.get_num_interop_threads(),
        "temperature_c": temperature_c,
        "ram_total_mb": virtual_memory.total / (1024.0**2),
        "ram_available_mb": virtual_memory.available / (1024.0**2),
    }


def _worker_paths(args: argparse.Namespace) -> tuple[Path, Path]:
    partial_dir = args.results_dir / ".partials"
    partial_dir.mkdir(parents=True, exist_ok=True)
    key = _worker_key(args.worker_system, args.worker_model)
    return partial_dir / f"{key}_summary.json", partial_dir / f"{key}_timings.csv"


def _worker_key(system: str, model_name: str | None) -> str:
    return f"{system}__{model_name}" if model_name else system


def _checkpoint_for(
    system: str, args: argparse.Namespace, model_name: str | None
) -> Path:
    if system == "digital_16k":
        if model_name is None:
            raise ValueError("A model name is required for digital_16k")
        return args.model_16k_dir / f"{model_name}_best.pt"
    if system == "digital_envelope_400hz":
        if model_name is None:
            raise ValueError("A model name is required for digital_envelope_400hz")
        architecture = f"{model_name}_1d"
        return args.model_400_digital_dir / architecture / "best_validation_checkpoint.pt"
    if system in ("fabricated_400hz_baselines", "fabricated_400hz_baselines_post_envelope"):
        if model_name is None:
            raise ValueError(f"A model name is required for {system}")
        return (
            args.model_fabricated_baselines_dir
            / model_name
            / f"best_fabricated_400hz_5d_{model_name}_classifier.pt"
        )
    return args.model_fabricated_path


def _requested_workers(args: argparse.Namespace) -> list[tuple[str, str | None]]:
    if args.system == "all":
        workers = [
            (system, model_name)
            for model_name in DIGITAL_MODEL_NAMES
            for system in ("digital_16k", "digital_envelope_400hz")
        ]
        workers.append(("fabricated_400hz", None))
        workers.extend(
            ("fabricated_400hz_baselines", model_name)
            for model_name in DIGITAL_MODEL_NAMES
        )
        if args.include_post_envelope:
            workers.append(("fabricated_post_envelope", None))
        return workers
    if args.system in (
        "digital_16k",
        "digital_envelope_400hz",
        "fabricated_400hz_baselines",
        "fabricated_400hz_baselines_post_envelope",
    ):
        return [(args.system, model_name) for model_name in DIGITAL_MODEL_NAMES]
    return [(args.system, None)]


def run_worker(args: argparse.Namespace) -> None:
    configure_torch_runtime()
    worker_start_info = system_information()
    system = args.worker_system
    model_name = args.worker_model
    checkpoint_path = _checkpoint_for(system, args, model_name).resolve()
    loaded = load_model(system, checkpoint_path, model_name)
    preprocess = _preprocessor(loaded)

    if system in (
        "fabricated_400hz",
        "fabricated_post_envelope",
        "fabricated_400hz_baselines",
        "fabricated_400hz_baselines_post_envelope",
    ):
        context_ms = float(loaded.checkpoint_metadata.get("context_ms", 50.0))
        if system in ("fabricated_400hz", "fabricated_400hz_baselines"):
            examples: Sequence[Any] = load_fabricated_raw_test_examples(
                args.fabricated_feature_path.resolve(),
                args.test_split_path.resolve(),
                context_ms,
            )
        elif system == "fabricated_400hz_baselines_post_envelope":
            examples = load_fabricated_baseline_envelope_test_examples(
                args.fabricated_feature_path.resolve(),
                args.test_split_path.resolve(),
                context_ms,
            )
        else:
            examples = load_fabricated_test_examples(
                args.fabricated_feature_path.resolve(),
                args.test_split_path.resolve(),
                context_ms,
            )
    else:
        examples = load_raw_test_examples(
            args.test_split_path.resolve(),
            args.raw_audiomnist_path.resolve(),
            preload_sample_rate=16_000 if system == "digital_16k" else None,
        )

    peak_ram_mb, baseline_rss_mb, absolute_peak_rss_mb = measure_peak_rss_mb(
        loaded.model, examples, preprocess
    )
    raw_rows, rtf = benchmark_pipeline(
        loaded.model, examples, preprocess, args.warmup, args.runs
    )
    cpu_time_ms = measure_cpu_time_ms(loaded.model, examples, preprocess, args.runs)
    accuracy = 100.0 * sum(
        int(row["prediction"]) == int(row["label"]) for row in raw_rows
    ) / len(raw_rows)
    shape_to_macs: dict[tuple[tuple[int, ...], ...], int] = {}
    macs_per_example = []
    for example in examples:
        prepared = _profile_input(loaded, example)
        shape_key = tuple(tuple(int(value) for value in tensor.shape) for tensor in prepared.args)
        if shape_key not in shape_to_macs:
            shape_to_macs[shape_key] = profile_model(loaded.model, prepared)
        macs_per_example.append(shape_to_macs[shape_key])

    observed_source_rates = sorted(
        {float(item.source_rate) for item in examples if hasattr(item, "source_rate")}
    )
    source_sample_rate_hz: int | float = loaded.source_sample_rate_hz
    if observed_source_rates:
        source_sample_rate_hz = statistics.median(observed_source_rates)
        if float(source_sample_rate_hz).is_integer():
            source_sample_rate_hz = int(source_sample_rate_hz)
    frontend = [row["frontend_ms"] for row in raw_rows]
    nn_times = [row["nn_ms"] for row in raw_rows]
    total = [row["total_ms"] for row in raw_rows]
    summary = {
        "system": loaded.name,
        "representation": loaded.representation,
        "source_sample_rate_hz": source_sample_rate_hz,
        "feature_rate_hz": loaded.feature_rate_hz,
        "channels": loaded.channels,
        "accuracy": accuracy,
        "frontend_median_ms": statistics.median(frontend),
        "frontend_p95_ms": percentile(frontend, 95),
        "nn_median_ms": statistics.median(nn_times),
        "nn_p95_ms": percentile(nn_times, 95),
        "total_median_ms": statistics.median(total),
        "total_p95_ms": percentile(total, 95),
        "cpu_time_ms": cpu_time_ms,
        "rtf": rtf,
        "nn_macs": int(statistics.median(macs_per_example)),
        "params": sum(parameter.numel() for parameter in loaded.model.parameters()),
        "peak_ram_mb": peak_ram_mb,
        "details": {
            "accuracy_unit": "percent",
            "architecture": loaded.architecture,
            "checkpoint": str(checkpoint_path),
            "test_examples": len(examples),
            "test_identity_sha256": sample_identity_sha256(examples),
            "sample_ids_auditable": not any(
                item.sample_id.startswith("fabricated_index=") for item in examples
            ),
            "warmup_iterations": args.warmup,
            "measured_runs": args.runs,
            "batch_size": 1,
            "device": "cpu",
            "nn_macs_definition": "median per held-out example; Conv/Linear/Recurrent MAC only",
            "nn_macs_min": int(min(macs_per_example)),
            "nn_macs_p95": int(percentile(macs_per_example, 95)),
            "nn_macs_max": int(max(macs_per_example)),
            "cpu_time_definition": "aggregate process user+system CPU delta divided by measured runs",
            "peak_ram_definition": (
                "peak worker-process RSS during an untimed inference pass minus RSS "
                "after model, frontend, and test inputs were loaded"
            ),
            "post_load_baseline_rss_mb": baseline_rss_mb,
            "absolute_peak_rss_mb": absolute_peak_rss_mb,
            "observed_source_sample_rates_hz": observed_source_rates,
            "measurement_role": (
                "optional post-envelope workload for a future analog envelope detector"
                if system == "fabricated_post_envelope"
                else "baseline-architecture comparison on the current implemented "
                "digital processing path"
                if system == "fabricated_400hz_baselines"
                else "baseline-architecture comparison on the hardware-native envelope "
                "path: the 400 Hz AC-RMS downsampling is assumed done by analog "
                "front-end hardware, so the NN reads saved envelope rows directly "
                "instead of recomputing AC-RMS on-device"
                if system == "fabricated_400hz_baselines_post_envelope"
                else "current implemented digital processing path"
            ),
            "frontend_definition": (
                "digital operations in preprocess function; excludes disk I/O and checkpoint loading"
            ),
            "checkpoint_metadata": loaded.checkpoint_metadata,
            "worker_system_information_before": worker_start_info,
            "worker_system_information_after": system_information(),
        },
    }
    summary_path, timing_path = _worker_paths(args)
    summary_path.write_text(json.dumps(summary, indent=2, default=str) + "\n", encoding="utf-8")
    for row in raw_rows:
        row["system"] = loaded.name
    write_csv(timing_path, raw_rows)
    print(json.dumps({key: value for key, value in summary.items() if key != "details"}, indent=2))


SUMMARY_FIELDS = (
    "system",
    "representation",
    "source_sample_rate_hz",
    "feature_rate_hz",
    "channels",
    "accuracy",
    "frontend_median_ms",
    "frontend_p95_ms",
    "nn_median_ms",
    "nn_p95_ms",
    "total_median_ms",
    "total_p95_ms",
    "cpu_time_ms",
    "rtf",
    "nn_macs",
    "params",
    "peak_ram_mb",
)


def write_csv(path: Path, rows: Sequence[dict[str, Any]], fields: Sequence[str] | None = None) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if fields is None:
        fields = tuple(rows[0].keys()) if rows else ()
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields, extrasaction="ignore")
        writer.writeheader()
        writer.writerows(rows)


def _path_is_placeholder(path: Path) -> bool:
    return str(path).startswith("/path/to/")


def validate_paths(
    args: argparse.Namespace, requested: Sequence[tuple[str, str | None]]
) -> None:
    required = {"test_split_path": args.test_split_path}
    if any(
        system in ("digital_16k", "digital_envelope_400hz")
        for system, _ in requested
    ):
        required["raw_audiomnist_path"] = args.raw_audiomnist_path
    if any(
        system
        in (
            "fabricated_400hz",
            "fabricated_post_envelope",
            "fabricated_400hz_baselines",
            "fabricated_400hz_baselines_post_envelope",
        )
        for system, _ in requested
    ):
        required["fabricated_feature_path"] = args.fabricated_feature_path
    for system, model_name in requested:
        key = _worker_key(system, model_name)
        required[f"checkpoint for {key}"] = _checkpoint_for(system, args, model_name)
    invalid = [f"{name}={path}" for name, path in required.items() if _path_is_placeholder(path) or not path.exists()]
    if invalid:
        raise FileNotFoundError(
            "Configure the following paths at the top of the script or with CLI flags:\n  "
            + "\n  ".join(invalid)
        )


def _child_command(
    args: argparse.Namespace, system: str, model_name: str | None
) -> list[str]:
    command = [
        sys.executable,
        str(Path(__file__).resolve()),
        "--worker-system",
        system,
        "--test-split-path",
        str(args.test_split_path),
        "--raw-audiomnist-path",
        str(args.raw_audiomnist_path),
        "--fabricated-feature-path",
        str(args.fabricated_feature_path),
        "--model-16k-dir",
        str(args.model_16k_dir),
        "--model-400-digital-dir",
        str(args.model_400_digital_dir),
        "--model-fabricated-path",
        str(args.model_fabricated_path),
        "--model-fabricated-baselines-dir",
        str(args.model_fabricated_baselines_dir),
        "--results-dir",
        str(args.results_dir),
        "--warmup",
        str(args.warmup),
        "--runs",
        str(args.runs),
    ]
    if model_name:
        command.extend(("--worker-model", model_name))
    return command


def _print_table(rows: Sequence[dict[str, Any]]) -> None:
    fields = (
        "system",
        "accuracy",
        "frontend_median_ms",
        "nn_median_ms",
        "total_median_ms",
        "total_p95_ms",
        "cpu_time_ms",
        "rtf",
        "nn_macs",
        "params",
        "peak_ram_mb",
    )
    headers = {
        "accuracy": "accuracy_%",
        "frontend_median_ms": "front_ms",
        "nn_median_ms": "nn_ms",
        "total_median_ms": "total_ms",
        "total_p95_ms": "p95_ms",
        "cpu_time_ms": "cpu_ms",
        "peak_ram_mb": "peak_delta_mb",
    }
    formatted = []
    for row in rows:
        formatted.append(
            {
                key: (
                    f"{row[key]:.4f}"
                    if isinstance(row[key], float)
                    else str(row[key])
                )
                for key in fields
            }
        )
    widths = {
        key: max(len(headers.get(key, key)), *(len(row[key]) for row in formatted))
        for key in fields
    }
    print("  ".join(headers.get(key, key).ljust(widths[key]) for key in fields))
    print("  ".join("-" * widths[key] for key in fields))
    for row in formatted:
        print("  ".join(row[key].ljust(widths[key]) for key in fields))


def run_parent(args: argparse.Namespace) -> None:
    configure_torch_runtime()
    requested = _requested_workers(args)
    validate_paths(args, requested)
    args.results_dir.mkdir(parents=True, exist_ok=True)

    info = system_information()
    print("System information:")
    print(json.dumps(info, indent=2))
    (args.results_dir / "system_info.json").write_text(
        json.dumps(info, indent=2) + "\n", encoding="utf-8"
    )
    governors = info.get("cpu_governor", [])
    if platform.system() == "Linux" and governors != ["performance"]:
        print(f"WARNING: expected performance governor, found {governors}", file=sys.stderr)

    summaries = []
    raw_rows = []
    details = {}
    for system, model_name in requested:
        key = _worker_key(system, model_name)
        print(f"\nRunning isolated worker: {key}", flush=True)
        subprocess.run(_child_command(args, system, model_name), check=True)
        summary_path = args.results_dir / ".partials" / f"{key}_summary.json"
        timing_path = args.results_dir / ".partials" / f"{key}_timings.csv"
        result = json.loads(summary_path.read_text(encoding="utf-8"))
        row_name = result["system"]
        details[row_name] = result.pop("details")
        summaries.append(result)
        with timing_path.open("r", encoding="utf-8", newline="") as handle:
            raw_rows.extend(csv.DictReader(handle))

    counts = {details[name]["test_examples"] for name in details}
    if len(counts) != 1:
        raise RuntimeError(f"Systems did not evaluate the same number of test samples: {counts}")
    digital_hashes = {
        details[name]["test_identity_sha256"]
        for name in details
        if name.startswith("digital_16k/")
        or name.startswith("digital_envelope_400hz/")
    }
    if len(digital_hashes) > 1:
        raise RuntimeError("The two digital systems did not use identical test samples")
    fabricated_hash = details.get("fabricated_400hz", {}).get("test_identity_sha256")
    fabricated_auditable = details.get("fabricated_400hz", {}).get("sample_ids_auditable")
    if fabricated_hash and digital_hashes and fabricated_auditable:
        if fabricated_hash not in digital_hashes:
            raise RuntimeError(
                "Fabricated speaker/digit/repetition identities do not match the raw test manifest"
            )
    elif fabricated_hash and digital_hashes:
        print(
            "WARNING: fabricated NPZ has no auditable sample IDs. Add speakers and "
            "repetitions or canonical sample_ids; only its sample count can be checked.",
            file=sys.stderr,
        )
    baseline_names = [
        name
        for name in details
        if name.startswith("fabricated_400hz_baselines/")
        or name.startswith("fabricated_400hz_baselines_post_envelope/")
    ]
    if baseline_names and digital_hashes:
        mismatched = [
            name
            for name in baseline_names
            if details[name]["test_identity_sha256"] not in digital_hashes
        ]
        if mismatched:
            raise RuntimeError(
                "Fabricated-baseline speaker/digit/repetition identities do not match "
                f"the raw test manifest: {mismatched}"
            )

    write_csv(args.results_dir / "benchmark_summary.csv", summaries, SUMMARY_FIELDS)
    write_csv(args.results_dir / "raw_timings.csv", raw_rows)
    metadata = {
        "system_information": info,
        "benchmark_contract": {
            "batch_size": 1,
            "device": "cpu",
            "torch_threads": 1,
            "torch_interop_threads": 1,
            "warmup_iterations": args.warmup,
            "measured_runs": args.runs,
            "timed_sample_schedule": (
                "deterministic evenly spaced held-out indices when runs <= test size; "
                "cyclic indices otherwise"
            ),
            "timing_clock": "time.perf_counter_ns",
            "process_cpu_source": "psutil.Process().cpu_times()",
            "total_ms": "frontend_ms + nn_ms",
            "rtf": "sum(total_processing_seconds) / sum(audio_duration_seconds)",
            "accuracy_unit": "percent",
            "peak_ram": "peak inference RSS minus post-load baseline RSS",
        },
        "systems": details,
    }
    (args.results_dir / "benchmark_metadata.json").write_text(
        json.dumps(metadata, indent=2) + "\n", encoding="utf-8"
    )
    print("\nBenchmark summary:")
    _print_table(summaries)
    print(f"\nSaved {args.results_dir / 'benchmark_summary.csv'}")
    print(f"Saved {args.results_dir / 'raw_timings.csv'}")


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--system", choices=("all", *SYSTEMS), default="all")
    parser.add_argument(
        "--include-post-envelope",
        action="store_true",
        help="also run the optional future analog-envelope workload",
    )
    parser.add_argument("--test-split-path", type=Path, default=TEST_SPLIT_PATH)
    parser.add_argument("--raw-audiomnist-path", type=Path, default=RAW_AUDIOMNIST_PATH)
    parser.add_argument("--fabricated-feature-path", type=Path, default=FABRICATED_FEATURE_PATH)
    parser.add_argument("--model-16k-dir", type=Path, default=MODEL_16K_DIR)
    parser.add_argument("--model-400-digital-dir", type=Path, default=MODEL_400_DIGITAL_DIR)
    parser.add_argument("--model-fabricated-path", type=Path, default=MODEL_FABRICATED_PATH)
    parser.add_argument(
        "--model-fabricated-baselines-dir",
        type=Path,
        default=MODEL_FABRICATED_BASELINES_DIR,
    )
    parser.add_argument("--results-dir", type=Path, default=RESULTS_DIR)
    parser.add_argument("--warmup", type=int, default=50)
    parser.add_argument("--runs", type=int, default=6_000)
    parser.add_argument("--worker-system", choices=SYSTEMS, help=argparse.SUPPRESS)
    parser.add_argument("--worker-model", choices=DIGITAL_MODEL_NAMES, help=argparse.SUPPRESS)
    args = parser.parse_args(argv)
    if args.warmup < 50:
        parser.error("--warmup must be at least 50")
    if args.runs != EXPECTED_TEST_EXAMPLES:
        parser.error(
            f"--runs must be exactly {EXPECTED_TEST_EXAMPLES} so every held-out sample is timed once"
        )
    return args


def main() -> None:
    args = parse_args()
    if args.worker_system:
        run_worker(args)
    else:
        run_parent(args)


if __name__ == "__main__":
    main()
