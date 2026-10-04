#!/usr/bin/env python3
"""One-channel broadband-envelope forms of the three literature-minimal CNN
baselines (cnn_baseline_models.py), for the spectral-channel ablation.

Each class is CnnOneFstride4Fabricated / TinyConvFabricated /
CnnTradFpool3Fabricated with the stem's in_channels set to 1, so only the
input width differs from the five-channel resonator models. The endpoint
follows the six digital_envelope_400hz models (EndpointClassifier400 in
benchmark_audiomnist_raspberry_pi.py): the input is always the fixed
one-second, 400-step broadband RMS envelope, so the decision is the plain
mean of the per-frame logits and forward() takes only the values tensor.

Self-contained (torch only) so the same file is copied into the Raspberry Pi
bundle. CausalConv1d and ConvBNAct1d are exact copies of the blocks in
fabricated_400hz_5d_baseline_models.py and benchmark_audiomnist_raspberry_pi.py.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F

BROADBAND_INPUT_CHANNELS = 1
CLASSES = 10


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


class EndpointClassifierBroadband(nn.Module):
    """Same endpoint as the six digital_envelope_400hz models."""

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return self.encode_frames(values).mean(dim=1)


class CnnOneFstride4Broadband(EndpointClassifierBroadband):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(BROADBAND_INPUT_CHANNELS, 16, kernel_size=8)
        self.classifier = nn.Linear(16, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values.transpose(1, 2))
        return self.classifier(values.transpose(1, 2))


class TinyConvBroadband(EndpointClassifierBroadband):
    total_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = CausalConv1d(
            BROADBAND_INPUT_CHANNELS, 8, kernel_size=10, stride=2, bias=True
        )
        self.classifier = nn.Linear(8, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class CnnTradFpool3Broadband(EndpointClassifierBroadband):
    total_stride = 3

    def __init__(self) -> None:
        super().__init__()
        self.conv1 = ConvBNAct1d(BROADBAND_INPUT_CHANNELS, 16, kernel_size=20, stride=3)
        self.conv2 = ConvBNAct1d(16, 16, kernel_size=10)
        self.classifier = nn.Linear(16, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv2(self.conv1(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


BROADBAND_CNN_FACTORIES = {
    "cnn_one_fstride4": CnnOneFstride4Broadband,
    "tiny_conv": TinyConvBroadband,
    "cnn_trad_fpool3": CnnTradFpool3Broadband,
}
