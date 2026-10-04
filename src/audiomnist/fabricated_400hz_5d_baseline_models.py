#!/usr/bin/env python3
"""Five-channel 400 Hz masked-pooling adaptations of the six matched compact
baseline architectures already frozen for the 16 kHz and one-channel 400 Hz
AudioMNIST benchmarks (BC-ResNet-1/2, EdgeSpeechNet-A, MatchboxNet 3x1x64,
MLPerf-Tiny DS-CNN, TC-ResNet8).

Every building block (CausalConv1d, ConvBNAct1d, TCSConv1d, and each
family's residual block) is an exact copy of the one already frozen in
benchmark_audiomnist_raspberry_pi.py. Only two things change per family
relative to its one-channel *Envelope sibling there:

1. The stem takes 5 input channels (one physical resonator sensor each)
   instead of 1.
2. The endpoint pooling is a length-masked mean over per-frame logits
   instead of an unconditional `.mean(dim=1)`, because batched training
   over variable-length digit segments zero-pads the tail of shorter
   examples and that padding must not be averaged into the prediction.
   Inference-time benchmarking (batch size 1, no padding) makes this
   change a no-op relative to unconditional mean pooling.

This module is imported by the training script. The Raspberry Pi benchmark
script is self-contained by design, so its copy of these same classes is a
literal, independently-pasted duplicate -- consistent with how
FabricatedPooledTcn/Direct400PooledTcnClassifier and the six *Envelope
classes already relate to their training-side counterparts in this repo.
"""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

INPUT_CHANNELS = 5
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


def _ceil_div_lengths(lengths: torch.Tensor, divisor: int) -> torch.Tensor:
    if divisor == 1:
        return lengths
    return torch.div(lengths + divisor - 1, divisor, rounding_mode="floor")


class EndpointClassifierFabricated5ch(nn.Module):
    """Shared length-masked mean endpoint pooling for five-channel input.

    `total_stride` is the architecture's fixed input->output time
    downsampling factor (the product of every strided op in the backbone);
    subclasses set it once as a class attribute. Iterated ceiling division
    by each individual stride equals a single ceiling division by their
    product, so this matches the true per-layer output length exactly.
    """

    total_stride: int = 1

    def forward(self, values: torch.Tensor, lengths: torch.Tensor) -> torch.Tensor:
        frame_logits = self.encode_frames(values)
        output_lengths = _ceil_div_lengths(lengths, self.total_stride)
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


class MatchboxResidualFabricated(nn.Module):
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


class MatchboxNet3x1x64Fabricated(EndpointClassifierFabricated5ch):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.prologue = TCSConv1d(INPUT_CHANNELS, 128, 11)
        self.blocks = nn.Sequential(
            MatchboxResidualFabricated(128, 64, 13),
            MatchboxResidualFabricated(64, 64, 15),
            MatchboxResidualFabricated(64, 64, 17),
        )
        self.epilogue = nn.Sequential(
            TCSConv1d(64, 128, 29, dilation=2),
            ConvBNAct1d(128, 128, 1),
        )
        self.classifier = nn.Linear(128, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.epilogue(self.blocks(self.prologue(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class TCResidualBlockFabricated(nn.Module):
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


class TCResNet8Fabricated(EndpointClassifierFabricated5ch):
    total_stride = 8

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(INPUT_CHANNELS, 16, 3)
        self.blocks = nn.Sequential(
            TCResidualBlockFabricated(16, 24, 2),
            TCResidualBlockFabricated(24, 32, 2),
            TCResidualBlockFabricated(32, 48, 2),
        )
        self.classifier = nn.Linear(48, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class DSConvBlock1dFabricated(nn.Sequential):
    def __init__(self, channels: int) -> None:
        super().__init__(
            CausalConv1d(channels, channels, 3, groups=channels),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
            nn.Conv1d(channels, channels, 1, bias=False),
            nn.BatchNorm1d(channels),
            nn.ReLU(inplace=True),
        )


class MLPerfTinyDSCNNFabricated(EndpointClassifierFabricated5ch):
    total_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(INPUT_CHANNELS, 64, 10, stride=2)
        self.blocks = nn.Sequential(*(DSConvBlock1dFabricated(64) for _ in range(4)))
        self.classifier = nn.Linear(64, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.blocks(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class BCResidual1dFabricated(nn.Module):
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


class BCResNetFabricated(EndpointClassifierFabricated5ch):
    total_stride = 8

    def __init__(self, scale: int = 1) -> None:
        super().__init__()
        channels = [16 * scale, 8 * scale, 12 * scale, 16 * scale, 20 * scale]
        self.stem = ConvBNAct1d(INPUT_CHANNELS, channels[0], 5, stride=2)
        stages = []
        in_channels = channels[0]
        for stage_index, (out_channels, repeats) in enumerate(
            zip(channels[1:], (2, 2, 4, 4))
        ):
            blocks = []
            for block_index in range(repeats):
                stride = 2 if stage_index in (1, 2) and block_index == 0 else 1
                blocks.append(
                    BCResidual1dFabricated(
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
        self.classifier = nn.Linear(32 * scale, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.stages(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


class EdgeResidual1dFabricated(nn.Module):
    def __init__(self, middle_channels: int) -> None:
        super().__init__()
        self.layers = nn.Sequential(
            ConvBNAct1d(39, middle_channels, 3),
            CausalConv1d(middle_channels, 39, 3),
            nn.BatchNorm1d(39),
        )

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        return F.relu(values + self.layers(values), inplace=True)


class EdgeSpeechNetAFabricated(EndpointClassifierFabricated5ch):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(INPUT_CHANNELS, 39, 3)
        self.blocks = nn.Sequential(
            *(EdgeResidual1dFabricated(width) for width in (20, 15, 25, 22, 22, 25))
        )
        self.final = ConvBNAct1d(39, 45, 3)
        self.classifier = nn.Linear(45, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.final(self.blocks(self.stem(values.transpose(1, 2))))
        return self.classifier(values.transpose(1, 2))


FABRICATED_400_FACTORIES = {
    "bc_resnet_1": lambda: BCResNetFabricated(scale=1),
    "bc_resnet_2": lambda: BCResNetFabricated(scale=2),
    "edgespeechnet_a": EdgeSpeechNetAFabricated,
    "matchboxnet_3x1x64": MatchboxNet3x1x64Fabricated,
    "mlperf_tiny_dscnn": MLPerfTinyDSCNNFabricated,
    "tc_resnet8": TCResNet8Fabricated,
}

ARCHITECTURE_IDS = {
    model_name: f"fabricated_400hz_5d_{model_name}_masked_mean_softmax"
    for model_name in FABRICATED_400_FACTORIES
}
