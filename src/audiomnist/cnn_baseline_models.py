#!/usr/bin/env python3
"""Three literature-minimal CNN baselines, each in two forms, to test
whether a genuinely low-capacity convolutional classifier struggles on
16 kHz digital_16k input but holds up on the 5-sensor 400 Hz fabricated
input (way fewer MACs than the six already-frozen compact architectures
in benchmark_audiomnist_raspberry_pi.py / fabricated_400hz_5d_baseline_models.py).

Architectures (not invented -- minimal keyword-spotting CNNs from the
literature, adapted to this project's input shapes):

- cnn-one-fstride4: Sainath & Parada (2015), "Convolutional Neural Networks
  for Small-Footprint Keyword Spotting" -- a single strided conv layer
  (stride 4 across the frequency/sensor axis) feeding one linear layer.
  The explicit low-capacity baseline in that paper.
- tiny_conv: the TensorFlow Lite Micro `micro_speech` reference model --
  one small strided conv layer and one linear layer, the bare-minimum
  microcontroller baseline used in TFLite Micro tutorials.
- cnn-trad-fpool3: Sainath & Parada (2015) again -- two conv layers with a
  frequency-axis max-pool ("fpool3") between them; still basic, but a
  deliberate middle point between the two single-conv baselines above and
  the six efficient architectures already benchmarked.

digital_16k versions classify the whole fixed 1-second, 96-frame x 40-mel
input at once (global average pool over time and frequency, then one
linear head) -- a non-causal, non-streaming design, matching how these
baselines are actually defined and trained in the literature. This is a
deliberate scope difference from the six existing digital_16k baselines in
scripts/create_audiomnist_compact_model_benchmark_colab.py, which are all
causal streaming classifiers; these three are not meant to run on the
Raspberry Pi streaming harness, only to anchor the accuracy/MACs/params
Pareto comparison.

fabricated versions follow the existing fabricated_400hz_5d_baseline_models.py
convention exactly: causal building blocks (CausalConv1d / ConvBNAct1d) and
EndpointClassifierFabricated5ch's length-masked mean pooling over per-frame
logits, so the existing fabricated trainer's machinery (padding/garbling
invariance checks, variable-length batching) applies unchanged.
"""

from __future__ import annotations

import torch
from torch import nn

from fabricated_400hz_5d_baseline_models import (
    CLASSES,
    INPUT_CHANNELS,
    CausalConv1d,
    ConvBNAct1d,
    EndpointClassifierFabricated5ch,
)

N_MELS = 40
COMPACT_FRAMES = 96  # (16_000 samples - 800 n_fft) // 160 hop + 1


# ---------------------------------------------------------------------------
# digital_16k: whole-utterance classifiers over a fixed (96, 40) log-mel grid.
# ---------------------------------------------------------------------------
class CnnOneFstride4(nn.Module):
    """Single conv layer, stride 4 across the mel/frequency axis."""

    def __init__(self, num_classes: int = CLASSES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 16, kernel_size=(8, 8), stride=(4, 1), bias=False)
        self.bn = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2).unsqueeze(1)  # B,T,F -> B,1,F,T
        values = torch.relu(self.bn(self.conv(values)))
        values = self.pool(values).flatten(1)
        return self.classifier(values)


class TinyConv(nn.Module):
    """TensorFlow Lite Micro `micro_speech` reference model: one small
    strided conv layer, one linear head, no batch norm."""

    def __init__(self, num_classes: int = CLASSES) -> None:
        super().__init__()
        self.conv = nn.Conv2d(1, 8, kernel_size=(10, 8), stride=(2, 2), bias=True)
        self.dropout = nn.Dropout(0.2)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(8, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2).unsqueeze(1)  # B,T,F -> B,1,F,T
        values = self.dropout(torch.relu(self.conv(values)))
        values = self.pool(values).flatten(1)
        return self.classifier(values)


class CnnTradFpool3(nn.Module):
    """Two conv layers with a frequency-axis max-pool ("fpool3") between
    them -- the "traditional" CNN from Sainath & Parada (2015), the larger
    of the three basic baselines here."""

    def __init__(self, num_classes: int = CLASSES) -> None:
        super().__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=(8, 20), bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.freq_pool = nn.MaxPool2d(kernel_size=(3, 1))
        self.conv2 = nn.Conv2d(16, 16, kernel_size=(4, 10), bias=False)
        self.bn2 = nn.BatchNorm2d(16)
        self.pool = nn.AdaptiveAvgPool2d(1)
        self.classifier = nn.Linear(16, num_classes)

    def forward(self, values: torch.Tensor) -> torch.Tensor:
        values = values.transpose(1, 2).unsqueeze(1)  # B,T,F -> B,1,F,T
        values = self.freq_pool(torch.relu(self.bn1(self.conv1(values))))
        values = torch.relu(self.bn2(self.conv2(values)))
        values = self.pool(values).flatten(1)
        return self.classifier(values)


DIGITAL16K_CNN_FACTORIES = {
    "cnn_one_fstride4": CnnOneFstride4,
    "tiny_conv": TinyConv,
    "cnn_trad_fpool3": CnnTradFpool3,
}


# ---------------------------------------------------------------------------
# fabricated: causal, length-masked-mean classifiers over 5-channel 400 Hz
# input, matching fabricated_400hz_5d_baseline_models.py's convention.
# ---------------------------------------------------------------------------
class CnnOneFstride4Fabricated(EndpointClassifierFabricated5ch):
    total_stride = 1

    def __init__(self) -> None:
        super().__init__()
        self.stem = ConvBNAct1d(INPUT_CHANNELS, 16, kernel_size=8)
        self.classifier = nn.Linear(16, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.stem(values.transpose(1, 2))
        return self.classifier(values.transpose(1, 2))


class TinyConvFabricated(EndpointClassifierFabricated5ch):
    total_stride = 2

    def __init__(self) -> None:
        super().__init__()
        self.stem = CausalConv1d(INPUT_CHANNELS, 8, kernel_size=10, stride=2, bias=True)
        self.classifier = nn.Linear(8, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = torch.relu(self.stem(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


class CnnTradFpool3Fabricated(EndpointClassifierFabricated5ch):
    total_stride = 3

    def __init__(self) -> None:
        super().__init__()
        # The "fpool3" downsample is folded into conv1's stride, like every
        # other strided block in this codebase, rather than a raw MaxPool1d:
        # a causal strided conv has a precisely bounded receptive field
        # (output step i never reads input past index i*stride), so it stays
        # compatible with EndpointClassifierFabricated5ch's ceil-division
        # length masking. A same-stride MaxPool1d does not have that
        # guarantee at the last partial window and leaks one step of padding.
        self.conv1 = ConvBNAct1d(INPUT_CHANNELS, 16, kernel_size=20, stride=3)
        self.conv2 = ConvBNAct1d(16, 16, kernel_size=10)
        self.classifier = nn.Linear(16, CLASSES)

    def encode_frames(self, values: torch.Tensor) -> torch.Tensor:
        values = self.conv2(self.conv1(values.transpose(1, 2)))
        return self.classifier(values.transpose(1, 2))


FABRICATED_CNN_FACTORIES = {
    "cnn_one_fstride4": CnnOneFstride4Fabricated,
    "tiny_conv": TinyConvFabricated,
    "cnn_trad_fpool3": CnnTradFpool3Fabricated,
}

ARCHITECTURE_IDS = {
    model_name: f"basic_cnn_baseline_{model_name}"
    for model_name in DIGITAL16K_CNN_FACTORIES
}
