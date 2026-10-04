#!/usr/bin/env python3
"""Build audited AudioMNIST accuracy-versus-compute comparison artifacts."""

from __future__ import annotations

import hashlib
import json
import math
import os
import sys
import time
from collections import defaultdict
from dataclasses import dataclass
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
import torch.nn as nn
import torch.nn.functional as F
from matplotlib.lines import Line2D
from matplotlib.ticker import FuncFormatter, MultipleLocator
from torch.nn.utils.rnn import PackedSequence


HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[1]
FIGURES = HERE / "figures"
SOURCE_ROOT = HERE / "benchmark_sources"
LOGMEL_RESULTS = SOURCE_ROOT / "audiomnist_16khz_model_benchmark.csv"
RMS_RESULTS = FIGURES / "audiomnist_400hz_envelope_accuracy_tradeoffs.csv"
LOGMEL_NOTEBOOK = ROOT / "audiomnist_causal_streaming_benchmark_colab.ipynb"
RMS_NOTEBOOK = ROOT / "audiomnist_400hz_digital_compact_models_colab.ipynb"
POOLED_TCN_SUMMARY = (
    HERE / "models/envelope_400hz_5d_pooled_tcn_classifier/heldout_test/heldout_test_summary.json"
)
CONV_GRU_SUMMARY = HERE / "models/causal_conv_gru_400hz/heldout_test/heldout_test_summary.json"
RAW400_TCN_SUMMARY = HERE / "models/raw_waveform_400hz_pooled_tcn/heldout_test_summary.json"

OUTPUT_CSV = FIGURES / "audiomnist_accuracy_energy_and_digital_ops.csv"
OUTPUT_TABLE = FIGURES / "audiomnist_accuracy_energy_and_digital_ops_table.md"
LEGACY_ENERGY_CSV = FIGURES / "audiomnist_accuracy_vs_dynamic_energy_proxy_5s.csv"
LEGACY_ENERGY_TABLE = FIGURES / "audiomnist_accuracy_vs_dynamic_energy_proxy_5s_table.md"
OUTPUT_ENERGY_PNG = FIGURES / "audiomnist_accuracy_vs_dynamic_energy_proxy_5s.png"
OUTPUT_ENERGY_PDF = FIGURES / "audiomnist_accuracy_vs_dynamic_energy_proxy_5s.pdf"
OUTPUT_OPS_PNG = FIGURES / "audiomnist_accuracy_vs_total_digital_ops_per_second.png"
OUTPUT_OPS_PDF = FIGURES / "audiomnist_accuracy_vs_total_digital_ops_per_second.pdf"
OUTPUT_ASSUMPTIONS = FIGURES / "audiomnist_accuracy_energy_and_ops_assumptions.json"
OUTPUT_METHOD = FIGURES / "audiomnist_accuracy_energy_and_ops_methodology.md"
OUTPUT_SENSITIVITY = FIGURES / "audiomnist_dynamic_energy_proxy_sensitivity.csv"

ARCHITECTURES = (
    "matchboxnet_3x1x64",
    "tc_resnet8",
    "mlperf_tiny_dscnn",
    "bc_resnet_1",
    "bc_resnet_2",
)
RMS_MODEL_ID = {name: f"{name}_1d" for name in ARCHITECTURES}
DISPLAY = {
    "matchboxnet_3x1x64": "MatchboxNet 3x1x64",
    "tc_resnet8": "TC-ResNet8",
    "mlperf_tiny_dscnn": "MLPerf Tiny DS-CNN",
    "bc_resnet_1": "BC-ResNet-1",
    "bc_resnet_2": "BC-ResNet-2",
    "fabricated_pooled_tcn": "Fabricated Pooled TCN",
    "fabricated_conv_gru96": "Fabricated Conv-UniGRU-96",
    "raw400_pooled_tcn": "Raw-waveform Pooled TCN",
}
POINT_LABEL = {
    "matchboxnet_3x1x64": "MatchboxNet",
    "tc_resnet8": "TC-ResNet8",
    "mlperf_tiny_dscnn": "DS-CNN",
    "bc_resnet_1": "BC-ResNet-1",
    "bc_resnet_2": "BC-ResNet-2",
    "fabricated_pooled_tcn": "Pooled TCN\n(projected POC)",
    "fabricated_conv_gru96": "Conv-UniGRU-96\n(projected POC)",
    "raw400_pooled_tcn": "Raw waveform TCN\n(no resonator)",
}
COLORS = {
    "matchboxnet_3x1x64": "#00798C",
    "tc_resnet8": "#30638E",
    "mlperf_tiny_dscnn": "#F4A261",
    "bc_resnet_1": "#2A9D8F",
    "bc_resnet_2": "#6A4C93",
    "fabricated_pooled_tcn": "#D1495B",
    "fabricated_conv_gru96": "#B56576",
    "raw400_pooled_tcn": "#5B5F97",
}

ANALYSIS_SECONDS = 5.0
ADC_ENERGY_PJ_PER_SAMPLE = 81.92
ADD_PJ = 0.9
MULTIPLY_PJ = 4.0
MAC_PJ = 4.9
COMPARE_PJ = 0.9
NONLINEAR_PJ = 98.0


@dataclass
class ScalarOps:
    add: float = 0.0
    multiply: float = 0.0
    compare: float = 0.0
    log: float = 0.0
    sqrt: float = 0.0
    exp: float = 0.0
    sigmoid: float = 0.0
    tanh: float = 0.0
    gelu: float = 0.0

    def total(self) -> float:
        return float(sum(vars(self).values()))

    def weighted_pj(self) -> float:
        nonlinear = self.log + self.sqrt + self.exp + self.sigmoid + self.tanh + self.gelu
        return (
            self.add * ADD_PJ
            + self.multiply * MULTIPLY_PJ
            + self.compare * COMPARE_PJ
            + nonlinear * NONLINEAR_PJ
        )


def _scenario_energy(
    row: pd.Series,
    *,
    add_pj: float,
    multiply_pj: float,
    mac_pj: float,
    compare_pj: float,
    nonlinear_pj: float,
    memory_pj_per_second: float = 0.0,
) -> float:
    add = row["front_add_ops_per_second"] + row["nn_aux_add_ops_per_second"]
    multiply = row["front_multiply_ops_per_second"] + row["nn_aux_multiply_ops_per_second"]
    compare = row["front_compare_ops_per_second"] + row["nn_aux_compare_ops_per_second"]
    nonlinear = sum(
        row[f"{prefix}_{kind}_ops_per_second"]
        for prefix in ("front", "nn_aux")
        for kind in ("log", "sqrt", "exp", "sigmoid", "tanh", "gelu")
    )
    energy_pj_per_second = row["input_samples_per_second"] * ADC_ENERGY_PJ_PER_SAMPLE
    energy_pj_per_second += add * add_pj + multiply * multiply_pj
    energy_pj_per_second += compare * compare_pj + nonlinear * nonlinear_pj
    energy_pj_per_second += row["nn_macs_per_second"] * mac_pj + memory_pj_per_second
    return ANALYSIS_SECONDS * energy_pj_per_second / 1e6


def build_sensitivity(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    scenarios = {
        "fp32_45nm_no_memory_primary": lambda row: _scenario_energy(
            row, add_pj=0.9, multiply_pj=4.0, mac_pj=4.9,
            compare_pj=0.9, nonlinear_pj=98.0,
        ),
        "int8_45nm_no_memory": lambda row: _scenario_energy(
            row, add_pj=0.03, multiply_pj=0.20, mac_pj=0.23,
            compare_pj=0.03, nonlinear_pj=4.6,
        ),
        "fp32_weights_once_per_s_from_8kb_sram": lambda row: _scenario_energy(
            row, add_pj=0.9, multiply_pj=4.0, mac_pj=4.9,
            compare_pj=0.9, nonlinear_pj=98.0,
            memory_pj_per_second=row["parameters"] * 5.0,
        ),
        "fp32_one_32bit_sram_weight_read_per_mac": lambda row: _scenario_energy(
            row, add_pj=0.9, multiply_pj=4.0, mac_pj=4.9,
            compare_pj=0.9, nonlinear_pj=98.0,
            memory_pj_per_second=row["nn_macs_per_second"] * 5.0,
        ),
        "fp32_checkpoint_dram_read_once_per_s": lambda row: _scenario_energy(
            row, add_pj=0.9, multiply_pj=4.0, mac_pj=4.9,
            compare_pj=0.9, nonlinear_pj=98.0,
            memory_pj_per_second=math.ceil(row["parameters"] / 2.0) * 1300.0,
        ),
        "stm32u5_active_cycle_no_memory": lambda row: _scenario_energy(
            row, add_pj=64.35, multiply_pj=64.35, mac_pj=64.35,
            compare_pj=64.35, nonlinear_pj=20.0 * 64.35,
        ),
    }
    values = pd.DataFrame(index=data.index)
    records: list[dict[str, object]] = []
    for scenario, function in scenarios.items():
        values[scenario] = data.apply(function, axis=1)
        for rank, index in enumerate(values[scenario].sort_values().index, start=1):
            records.append(
                {
                    "scenario": scenario,
                    "rank": rank,
                    "model": data.loc[index, "model"],
                    "representation": data.loc[index, "representation"],
                    "dynamic_energy_proxy_uj_per_5s": values.loc[index, scenario],
                }
            )
    primary_rank = values["fp32_45nm_no_memory_primary"].rank()
    summary = []
    for scenario in values:
        minimum_index = values[scenario].idxmin()
        summary.append(
            {
                "scenario": scenario,
                "spearman_rank_correlation_vs_primary": primary_rank.corr(
                    values[scenario].rank(), method="spearman"
                ),
                "lowest_energy_model": data.loc[minimum_index, "model"],
                "lowest_energy_representation": data.loc[minimum_index, "representation"],
            }
        )
    return pd.DataFrame(records), pd.DataFrame(summary)


def _hash(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _load_notebook_factories(
    path: Path, marker: str, factory_name: str, cutoff: str, constants: dict[str, object]
) -> dict[str, object]:
    notebook = json.loads(path.read_text(encoding="utf-8"))
    matches = []
    for cell in notebook["cells"]:
        if cell["cell_type"] != "code":
            continue
        source = "".join(cell["source"])
        if marker in source:
            matches.append(source)
    if len(matches) != 1:
        raise RuntimeError(f"Expected one architecture cell containing {marker!r}, found {len(matches)}")
    source = matches[0]
    if cutoff not in source:
        raise RuntimeError(f"Architecture-cell cutoff {cutoff!r} was not found")
    source = source[: source.index(cutoff)]
    namespace: dict[str, object] = {
        "math": math,
        "np": np,
        "torch": torch,
        "nn": nn,
        "F": F,
        **constants,
    }
    exec(compile(source, str(path), "exec"), namespace)
    return namespace[factory_name]


def load_model_factories() -> tuple[dict[str, object], dict[str, object]]:
    logmel = _load_notebook_factories(
        LOGMEL_NOTEBOOK,
        "COMPACT_FACTORIES = {",
        "COMPACT_FACTORIES",
        "dummy = torch.randn",
        {"N_MELS": 40},
    )
    rms = _load_notebook_factories(
        RMS_NOTEBOOK,
        '"matchboxnet_3x1x64_1d": MatchboxNet3x1x64',
        "MODEL_FACTORIES",
        "# Shape and strict past-only audit.",
        {},
    )
    return logmel, rms


def _numel(value: object) -> int:
    if torch.is_tensor(value):
        return value.numel()
    if isinstance(value, (tuple, list)):
        return sum(_numel(item) for item in value)
    return 0


def _tensor(value: object) -> torch.Tensor | None:
    if torch.is_tensor(value):
        return value
    if isinstance(value, (tuple, list)):
        for item in value:
            found = _tensor(item)
            if found is not None:
                return found
    return None


def _add_layer_norm_ops(ops: ScalarOps, output_elements: int, width: int) -> None:
    vectors = output_elements // width
    # Mean, variance, epsilon, normalization, and learned affine transform.
    ops.add += vectors * (5 * width - 1)
    ops.multiply += vectors * (3 * width + 3)
    ops.sqrt += vectors


def _add_top1_decoder(ops: ScalarOps, classes: int = 10) -> None:
    # The evaluated notebooks use argmax on logits; softmax/top-3 is unnecessary
    # for a deployed top-1 decision and is not charged here.
    ops.compare += classes - 1


def profile_endpoint_model(model: nn.Module, sample: torch.Tensor) -> tuple[int, ScalarOps, dict[str, int]]:
    model = model.cpu().eval()
    macs = 0
    auxiliary = ScalarOps()
    details: defaultdict[str, int] = defaultdict(int)
    handles = []
    residual_names = {
        "MatchboxResidualBlock",
        "TCResidualBlock",
        "BCResBlock",
        "MatchboxResidual",
        "TCResidual",
        "BCResidual1d",
    }
    frequency_mean_module = model.blocks if model.__class__.__name__ == "MLPerfTinyDSCNN" else None
    final_frequency_mean_module = getattr(model, "final_features", None)

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        nonlocal macs
        output_elements = _numel(output)
        value = _tensor(output)
        name = module.__class__.__name__
        if isinstance(module, (nn.Conv1d, nn.Conv2d)):
            kernel = int(np.prod(module.kernel_size))
            macs += int(output_elements * kernel * module.in_channels / module.groups)
            if module.bias is not None:
                auxiliary.add += output_elements
        elif isinstance(module, nn.Linear):
            macs += int(output_elements * module.in_features)
            if module.bias is not None:
                auxiliary.add += output_elements
        elif isinstance(module, nn.ReLU):
            auxiliary.compare += output_elements
        elif isinstance(module, nn.SiLU):
            auxiliary.sigmoid += output_elements
            auxiliary.multiply += output_elements
        elif isinstance(module, nn.GELU):
            auxiliary.gelu += output_elements
        elif isinstance(module, nn.LayerNorm):
            width = int(np.prod(module.normalized_shape))
            _add_layer_norm_ops(auxiliary, output_elements, width)
        elif name == "SubSpectralNorm":
            # Its per-subspectrum affine parameters cannot be folded into one
            # channel-wise convolution coefficient.
            auxiliary.multiply += output_elements
            auxiliary.add += output_elements

        if name in residual_names:
            residual_adds = output_elements
            if name == "BCResBlock" and not module.transition:
                residual_adds += output_elements
            auxiliary.add += residual_adds
            auxiliary.compare += output_elements
            details["residual_additions"] += residual_adds
            if name == "BCResBlock" and value is not None:
                frequency = int(value.shape[2])
                reductions = value.numel() // frequency
                auxiliary.add += reductions * (frequency - 1)
                auxiliary.multiply += reductions
                details["frequency_mean_outputs"] += reductions

        if module is frequency_mean_module and value is not None:
            frequency = int(value.shape[-1])
            reductions = value.numel() // frequency
            auxiliary.add += reductions * (frequency - 1)
            auxiliary.multiply += reductions
            details["frequency_mean_outputs"] += reductions
        if module is final_frequency_mean_module and value is not None:
            frequency = int(value.shape[2])
            reductions = value.numel() // frequency
            auxiliary.add += reductions * (frequency - 1)
            auxiliary.multiply += reductions
            details["frequency_mean_outputs"] += reductions

    for module in model.modules():
        if module is not model:
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(sample)
    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        frame_logits = model.encode_frames(sample)
    steps = int(frame_logits.shape[1])
    classes = int(frame_logits.shape[2])
    auxiliary.add += classes * (steps - 1)
    auxiliary.multiply += classes
    _add_top1_decoder(auxiliary, classes)
    details["endpoint_mean_steps"] = steps
    details["output_classes"] = classes
    details["batch_norm_fused"] = 1
    return macs, auxiliary, dict(details)


def profile_pooled_tcn(*, raw_waveform: bool = False) -> tuple[int, ScalarOps, dict[str, int], int]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    if raw_waveform:
        import train_raw_waveform_400hz_pooled_tcn as raw

        model = raw.Raw400PooledTcnClassifier().cpu().eval()
        sample = torch.zeros(1, 400, 1)
    else:
        import train_direct_400hz_5d_pooled_tcn_classifier as pooled

        model = pooled.Direct400PooledTcnClassifier().cpu().eval()
        sample = torch.zeros(1, 400, 5)
    lengths = torch.tensor([400])
    macs = 0
    ops = ScalarOps()
    details: defaultdict[str, int] = defaultdict(int)
    handles = []

    def hook(module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        nonlocal macs
        elements = _numel(output)
        if isinstance(module, nn.Conv1d):
            kernel = int(np.prod(module.kernel_size))
            macs += int(elements * kernel * module.in_channels / module.groups)
            if module.bias is not None:
                ops.add += elements
        elif isinstance(module, nn.Linear):
            macs += int(elements * module.in_features)
            if module.bias is not None:
                ops.add += elements
        elif isinstance(module, nn.LayerNorm):
            _add_layer_norm_ops(ops, elements, int(np.prod(module.normalized_shape)))
        if module.__class__.__name__ == "CausalDepthwiseBlock":
            ops.sigmoid += elements
            ops.multiply += elements
            ops.add += elements
            details["tcn_silu_and_residual_elements"] += elements

    for module in model.modules():
        if module is not model:
            handles.append(module.register_forward_hook(hook))
    # The front-end SiLU is functional and follows this LayerNorm.
    def frontend_activation(_module: nn.Module, _inputs: tuple[object, ...], output: object) -> None:
        elements = _numel(output)
        ops.sigmoid += elements
        ops.multiply += elements

    handles.append(model.frontend_normalization.register_forward_hook(frontend_activation))
    with torch.inference_mode():
        model(sample, lengths)
    for handle in handles:
        handle.remove()

    with torch.inference_mode():
        frames, output_lengths = model.encode_frames(sample, lengths)
    channels = int(frames.shape[1])
    steps = int(output_lengths.item())
    ops.multiply += channels * steps  # validity mask
    ops.add += channels * (steps - 1)  # mean sum
    ops.multiply += channels  # mean division
    ops.compare += channels * (steps - 1)  # maximum
    _add_top1_decoder(ops)
    details["pool_channels"] = channels
    details["pool_steps"] = steps
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return macs, ops, dict(details), parameters


def profile_conv_gru() -> tuple[int, ScalarOps, dict[str, int], int]:
    if str(HERE) not in sys.path:
        sys.path.insert(0, str(HERE))
    from audiomnist_model import CausalDigitClassifier

    model = CausalDigitClassifier().cpu().eval()
    sample = torch.zeros(1, 100, 20)
    lengths = torch.tensor([100])
    macs = 0
    ops = ScalarOps()
    details: defaultdict[str, int] = defaultdict(int)
    handles = []

    def hook(module: nn.Module, inputs: tuple[object, ...], output: object) -> None:
        nonlocal macs
        elements = _numel(output)
        if isinstance(module, nn.Conv1d):
            kernel = int(np.prod(module.kernel_size))
            macs += int(elements * kernel * module.in_channels / module.groups)
            if module.bias is not None:
                ops.add += elements
        elif isinstance(module, nn.Linear):
            macs += int(elements * module.in_features)
            if module.bias is not None:
                ops.add += elements
        elif isinstance(module, nn.GELU):
            ops.gelu += elements
        elif isinstance(module, nn.GRU):
            packed = inputs[0]
            rows = int(packed.data.shape[0]) if isinstance(packed, PackedSequence) else int(np.prod(packed.shape[:2]))
            directions = 2 if module.bidirectional else 1
            layer_input = module.input_size
            for _ in range(module.num_layers):
                macs += rows * directions * 3 * module.hidden_size * (layer_input + module.hidden_size)
                layer_input = directions * module.hidden_size
            cells = rows * directions * module.hidden_size * module.num_layers
            ops.sigmoid += 2 * cells
            ops.tanh += cells
            ops.multiply += 3 * cells
            ops.add += 3 * cells
            if module.bias:
                ops.add += 6 * cells
            details["gru_cell_updates"] += cells

    for module in model.modules():
        if module is not model:
            handles.append(module.register_forward_hook(hook))
    with torch.inference_mode():
        model(sample, lengths)
    for handle in handles:
        handle.remove()
    _add_top1_decoder(ops)
    parameters = sum(parameter.numel() for parameter in model.parameters())
    return macs, ops, dict(details), parameters


def logmel_frontend_ops(source: pd.Series) -> tuple[ScalarOps, dict[str, float]]:
    frames = int(round(float(source["feature_values_per_second"]) / 40.0))
    fft_size = 800
    bins = fft_size // 2 + 1
    mels = 40
    fft_ops = frames * 5.0 * fft_size * math.log2(fft_size)
    ops = ScalarOps(
        add=(
            0.60 * fft_ops
            + frames * bins
            + frames * mels * (bins - 1)
            + frames * mels
        ),
        multiply=(
            frames * fft_size
            + 0.40 * fft_ops
            + frames * 2 * bins
            + frames * mels * bins
            + frames * mels
        ),
        log=frames * mels,
    )
    recorded_arithmetic = float(source["frontend_estimated_arithmetic_ops"])
    reconstructed_arithmetic = ops.add + ops.multiply - 2 * frames * mels
    if not math.isclose(recorded_arithmetic, reconstructed_arithmetic, rel_tol=1e-12):
        raise RuntimeError("The reconstructed log-Mel arithmetic does not match the Colab artifact")
    return ops, {
        "input_samples_per_second": 16_000.0,
        "feature_frames_per_one_second_example": float(frames),
        "fft_size": float(fft_size),
        "window_ms": 50.0,
        "hop_ms": 10.0,
        "normalization_ops_per_second": float(2 * frames * mels),
    }


def rms_frontend_ops() -> tuple[ScalarOps, dict[str, float]]:
    # The AudioMNIST notebook's deployment convention is a streaming running
    # RMS: square/add/remove at 48 kHz, 400 square roots/s, and two frozen
    # normalization operations for each of 400 envelope values.
    ops = ScalarOps(
        add=2 * 48_000 + 400,
        multiply=48_000 + 400,
        sqrt=400,
    )
    if ops.total() != 145_200:
        raise RuntimeError("Unexpected AudioMNIST RMS front-end operation count")
    return ops, {
        "input_samples_per_second": 48_000.0,
        "feature_frames_per_one_second_example": 400.0,
        "fft_size": 0.0,
        "window_ms": 10.0,
        "hop_ms": 2.5,
        "normalization_ops_per_second": 800.0,
    }


def fabricated_frontend_ops() -> tuple[ScalarOps, dict[str, float]]:
    return ScalarOps(), {
        "input_samples_per_second": 2_000.0,
        "feature_frames_per_one_second_example": 400.0,
        "fft_size": 0.0,
        "window_ms": 0.0,
        "hop_ms": 2.5,
        "normalization_ops_per_second": 0.0,
    }


def raw400_frontend_ops() -> tuple[ScalarOps, dict[str, float]]:
    # Deployment interpretation: one already anti-aliased 400 Hz ADC stream.
    # Frozen scalar normalization performs one subtraction and multiply/sample.
    return ScalarOps(add=400, multiply=400), {
        "input_samples_per_second": 400.0,
        "feature_frames_per_one_second_example": 400.0,
        "fft_size": 0.0,
        "window_ms": 0.0,
        "hop_ms": 2.5,
        "normalization_ops_per_second": 800.0,
    }


def _row(
    *,
    model_id: str,
    representation: str,
    accuracy: float,
    parameters: int,
    expected_macs: int,
    profile: tuple[int, ScalarOps, dict[str, int]],
    front: tuple[ScalarOps, dict[str, float]],
    status: str,
    accuracy_source: str,
) -> dict[str, object]:
    macs, auxiliary, profile_details = profile
    front_ops, front_details = front
    if macs != expected_macs:
        raise RuntimeError(f"{model_id}: profiled {macs:,} MAC != artifact {expected_macs:,}")
    total_digital = int(round(front_ops.total())) + macs + int(round(auxiliary.total()))
    adc_pj_per_second = front_details["input_samples_per_second"] * ADC_ENERGY_PJ_PER_SAMPLE
    total_pj = ANALYSIS_SECONDS * (
        adc_pj_per_second
        + front_ops.weighted_pj()
        + macs * MAC_PJ
        + auxiliary.weighted_pj()
    )
    row: dict[str, object] = {
        "model": DISPLAY[model_id],
        "model_id": model_id,
        "representation": representation,
        "accuracy": float(accuracy),
        "front_end_ops_per_second": int(round(front_ops.total())),
        "nn_macs_per_second": int(macs),
        "nn_aux_ops_per_second": int(round(auxiliary.total())),
        "total_digital_ops_per_second": int(total_digital),
        "weighted_dynamic_energy_proxy_uj_per_5s": total_pj / 1e6,
        "adc_proxy_uj_per_second": adc_pj_per_second / 1e6,
        "front_end_proxy_uj_per_second": front_ops.weighted_pj() / 1e6,
        "nn_mac_proxy_uj_per_second": macs * MAC_PJ / 1e6,
        "nn_aux_proxy_uj_per_second": auxiliary.weighted_pj() / 1e6,
        "parameters": int(parameters),
        "measured_projected": status,
        "accuracy_source": accuracy_source,
        **{f"front_{key}_ops_per_second": value for key, value in vars(front_ops).items()},
        **{f"nn_aux_{key}_ops_per_second": value for key, value in vars(auxiliary).items()},
        **front_details,
        **{f"nn_profile_{key}": value for key, value in profile_details.items()},
    }
    return row


def build_table() -> pd.DataFrame:
    required = [
        LOGMEL_RESULTS, RMS_RESULTS, LOGMEL_NOTEBOOK, RMS_NOTEBOOK,
        POOLED_TCN_SUMMARY, CONV_GRU_SUMMARY, RAW400_TCN_SUMMARY,
    ]
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing AudioMNIST artifacts:\n" + "\n".join(missing))

    logmel = pd.read_csv(LOGMEL_RESULTS).set_index("model")
    rms = pd.read_csv(RMS_RESULTS)
    rms = rms.loc[rms["system"].str.startswith("Digital")].set_index("model")
    logmel_factories, rms_factories = load_model_factories()
    rows: list[dict[str, object]] = []

    for model_id in ARCHITECTURES:
        source = logmel.loc[model_id]
        model = logmel_factories[model_id]()
        parameters = sum(parameter.numel() for parameter in model.parameters())
        if parameters != int(source["total_parameters"]):
            raise RuntimeError(f"{model_id}: 16 kHz parameter mismatch")
        rows.append(
            _row(
                model_id=model_id,
                representation="16 kHz 40-bin log-Mel (evaluated)",
                accuracy=float(source["test_accuracy_percent"]),
                parameters=parameters,
                expected_macs=int(source["learned_macs"]),
                profile=profile_endpoint_model(model, torch.zeros(1, 96, 40)),
                front=logmel_frontend_ops(source),
                status="evaluated",
                accuracy_source="speaker-disjoint held-out AudioMNIST test; downloaded Colab artifact",
            )
        )

        display_name = DISPLAY[model_id]
        source_400 = rms.loc[display_name]
        model_400 = rms_factories[RMS_MODEL_ID[model_id]]()
        parameters_400 = sum(parameter.numel() for parameter in model_400.parameters())
        if parameters_400 != int(source_400["parameters"]):
            raise RuntimeError(f"{model_id}: 400 Hz parameter mismatch")
        rows.append(
            _row(
                model_id=model_id,
                representation="400 Hz digital RMS from 48 kHz waveform (evaluated)",
                accuracy=float(source_400["accuracy_percent"]),
                parameters=parameters_400,
                expected_macs=int(source_400["model_macs_per_s"]),
                profile=profile_endpoint_model(model_400, torch.zeros(1, 400, 1)),
                front=rms_frontend_ops(),
                status="evaluated current digital extraction",
                accuracy_source="speaker-disjoint held-out AudioMNIST test; 400 Hz compact-model artifact",
            )
        )

    pooled_summary = json.loads(POOLED_TCN_SUMMARY.read_text(encoding="ascii"))
    pooled_macs, pooled_aux, pooled_details, pooled_parameters = profile_pooled_tcn()
    if pooled_parameters != int(pooled_summary["parameters"]):
        raise RuntimeError("Fabricated pooled-TCN parameter mismatch")
    rows.append(
        _row(
            model_id="fabricated_pooled_tcn",
            representation="fabricated 5-channel 400 Hz RMS envelopes (projected POC)",
            accuracy=100.0 * float(pooled_summary["test_top1_accuracy"]),
            parameters=pooled_parameters,
            expected_macs=int(pooled_summary["learned_macs_per_one_second_utterance"]),
            profile=(pooled_macs, pooled_aux, pooled_details),
            front=fabricated_frontend_ops(),
            status="accuracy evaluated; model-ready physical-envelope front end projected",
            accuracy_source="physical-capture held-out test; software RMS accuracy inherited by projected POC",
        )
    )

    raw_summary = json.loads(RAW400_TCN_SUMMARY.read_text(encoding="ascii"))
    raw_macs, raw_aux, raw_details, raw_parameters = profile_pooled_tcn(raw_waveform=True)
    if raw_parameters != int(raw_summary["parameters"]):
        raise RuntimeError("Raw-waveform pooled-TCN parameter mismatch")
    rows.append(
        _row(
            model_id="raw400_pooled_tcn",
            representation="direct 1-channel 400 Hz waveform (no resonator control)",
            accuracy=float(raw_summary["test_top1_accuracy_percent"]),
            parameters=raw_parameters,
            expected_macs=int(raw_summary["profiled_macs_per_one_second_utterance"]),
            profile=(raw_macs, raw_aux, raw_details),
            front=raw400_frontend_ops(),
            status="accuracy evaluated; direct 400 Hz ADC and analog anti-aliasing projected",
            accuracy_source="speaker-disjoint held-out AudioMNIST test; anti-aliased offline 48-to-0.4 kHz resampling",
        )
    )

    gru_summary = json.loads(CONV_GRU_SUMMARY.read_text(encoding="ascii"))
    gru_macs, gru_aux, gru_details, gru_parameters = profile_conv_gru()
    if gru_parameters != int(gru_summary["model_parameters"]):
        raise RuntimeError("Fabricated Conv-GRU parameter mismatch")
    rows.append(
        _row(
            model_id="fabricated_conv_gru96",
            representation="fabricated 5-channel 400 Hz RMS envelopes (projected POC)",
            accuracy=100.0 * float(gru_summary["accuracy"]),
            parameters=gru_parameters,
            expected_macs=gru_macs,
            profile=(gru_macs, gru_aux, gru_details),
            front=fabricated_frontend_ops(),
            status="accuracy evaluated; model-ready physical-envelope front end projected",
            accuracy_source="physical-capture held-out test; packed 5 x 400 Hz envelopes at 100 model steps/s",
        )
    )
    return pd.DataFrame(rows)


def required_table(data: pd.DataFrame) -> pd.DataFrame:
    return data[
        [
            "model",
            "representation",
            "accuracy",
            "front_end_ops_per_second",
            "nn_macs_per_second",
            "nn_aux_ops_per_second",
            "total_digital_ops_per_second",
            "weighted_dynamic_energy_proxy_uj_per_5s",
            "measured_projected",
        ]
    ].copy()


def write_table(table: pd.DataFrame) -> None:
    shown = table.copy()
    shown["accuracy"] = shown["accuracy"].map(lambda value: f"{value:.2f}")
    for column in (
        "front_end_ops_per_second",
        "nn_macs_per_second",
        "nn_aux_ops_per_second",
        "total_digital_ops_per_second",
    ):
        shown[column] = shown[column].map(lambda value: f"{int(value):,}")
    shown["weighted_dynamic_energy_proxy_uj_per_5s"] = shown[
        "weighted_dynamic_energy_proxy_uj_per_5s"
    ].map(lambda value: f"{value:.3f}")
    lines = [
        "# AudioMNIST accuracy, energy proxy, and digital operations",
        "",
        "| " + " | ".join(shown.columns) + " |",
        "| " + " | ".join("---" for _ in shown.columns) + " |",
    ]
    for row in shown.itertuples(index=False, name=None):
        lines.append("| " + " | ".join(map(str, row)) + " |")
    OUTPUT_TABLE.write_text("\n".join(lines) + "\n", encoding="ascii")


def configure_plot() -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7D8992",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
            "text.color": "#18232C",
            "axes.labelcolor": "#263640",
            "xtick.color": "#42515B",
            "ytick.color": "#42515B",
        }
    )


def _save_figure(figure, png_path: Path, pdf_path: Path) -> None:
    for path, options in (
        (png_path, {"dpi": 320, "bbox_inches": "tight"}),
        (pdf_path, {"bbox_inches": "tight"}),
    ):
        temporary = path.with_name(f".{path.stem}.{os.getpid()}.tmp{path.suffix}")
        figure.savefig(temporary, **options)
        last_error: OSError | None = None
        for _ in range(8):
            try:
                os.replace(temporary, path)
                last_error = None
                break
            except OSError as error:
                last_error = error
                time.sleep(0.25)
        if last_error is not None:
            temporary.unlink(missing_ok=True)
            raise last_error


def _draw_points(axis, data: pd.DataFrame, x_column: str) -> None:
    for model_id in ARCHITECTURES:
        group = data.loc[data["model_id"].eq(model_id)]
        logmel = group.loc[group["representation"].str.startswith("16 kHz")].iloc[0]
        rms = group.loc[group["representation"].str.startswith("400 Hz")].iloc[0]
        axis.plot(
            [logmel[x_column], rms[x_column]],
            [logmel["accuracy"], rms["accuracy"]],
            color=COLORS[model_id],
            linewidth=1.45,
            alpha=0.78,
            zorder=2,
        )
    for row in data.itertuples(index=False):
        fabricated = row.model_id.startswith("fabricated_")
        raw_control = row.model_id == "raw400_pooled_tcn"
        marker = "D" if raw_control else (
            "*" if fabricated else ("o" if row.representation.startswith("16 kHz") else "^")
        )
        axis.scatter(
            getattr(row, x_column),
            row.accuracy,
            s=150 if fabricated else (70 if raw_control else 58),
            marker=marker,
            color=COLORS[row.model_id],
            edgecolor="white",
            linewidth=0.8,
            zorder=4,
        )
        if raw_control:
            label = POINT_LABEL[row.model_id]
            kind = "raw control"
        elif fabricated:
            label = POINT_LABEL[row.model_id]
            kind = "fabricated"
        else:
            kind = "16k" if row.representation.startswith("16 kHz") else "400 Hz"
            label = f"{POINT_LABEL[row.model_id]} {kind}"

        # Display-point offsets remain stable in both vector and raster exports.
        common_offsets = {
            ("matchboxnet_3x1x64", "16k"): (-4, -13, "right"),
            ("matchboxnet_3x1x64", "400 Hz"): (6, 5, "left"),
            ("tc_resnet8", "16k"): (-4, 6, "right"),
            ("tc_resnet8", "400 Hz"): (5, -13, "left"),
            ("mlperf_tiny_dscnn", "16k"): (-5, -13, "right"),
            ("mlperf_tiny_dscnn", "400 Hz"): (5, -13, "left"),
            ("bc_resnet_1", "16k"): (5, -13, "left"),
            ("bc_resnet_1", "400 Hz"): (5, 5, "left"),
            ("bc_resnet_2", "16k"): (5, 6, "left"),
            ("bc_resnet_2", "400 Hz"): (5, 5, "left"),
            ("fabricated_pooled_tcn", "fabricated"): (7, 5, "left"),
            ("fabricated_conv_gru96", "fabricated"): (7, 5, "left"),
            ("raw400_pooled_tcn", "raw control"): (7, 5, "left"),
        }
        dx, dy, alignment = common_offsets[(row.model_id, kind)]
        axis.annotate(
            label,
            (getattr(row, x_column), row.accuracy),
            xytext=(dx, dy),
            textcoords="offset points",
            ha=alignment,
            va="bottom",
            fontsize=6.0,
            color="#263640",
            zorder=5,
        )


def _legend(figure) -> None:
    handles = [
        Line2D([0], [0], marker="o", color="none", markerfacecolor="#596873",
               markeredgecolor="white", markersize=7, label="16 kHz log-Mel evaluated"),
        Line2D([0], [0], marker="^", color="none", markerfacecolor="#596873",
               markeredgecolor="white", markersize=7, label="400 Hz digital RMS evaluated"),
        Line2D([0], [0], marker="*", color="none", markerfacecolor="#D1495B",
               markeredgecolor="white", markersize=11, label="Fabricated envelope model (projected POC)"),
        Line2D([0], [0], marker="D", color="none", markerfacecolor=COLORS["raw400_pooled_tcn"],
               markeredgecolor="white", markersize=6, label="Direct 400 Hz waveform control"),
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.53, 0.895),
                  ncol=4, frameon=False, fontsize=7.3, columnspacing=1.0)


def _ideal(axis) -> None:
    axis.annotate(
        "IDEAL\nhigher accuracy\nlower cost",
        xy=(0.025, 0.96), xytext=(0.17, 0.74),
        xycoords="axes fraction", textcoords="axes fraction",
        arrowprops={"arrowstyle": "->", "color": "#30745A", "linewidth": 1.15},
        ha="center", va="center", fontsize=8.4, fontweight="semibold", color="#245E49",
    )


def plot_ops(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 6.5))
    figure.subplots_adjust(left=0.105, right=0.975, top=0.82, bottom=0.19)
    _draw_points(axis, data, "total_digital_ops_per_second")
    maximum = float(data["total_digital_ops_per_second"].max())
    axis.set_xlim(0, maximum * 1.08)
    axis.set_ylim(50, 102)
    axis.xaxis.set_major_locator(MultipleLocator(5e6))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 1e6:.0f}"))
    axis.yaxis.set_major_locator(MultipleLocator(5))
    axis.grid(True, color="#D4D8DB", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlabel("Total digital operations/s (millions, linear scale)", fontsize=10.5)
    axis.set_ylabel("Held-out top-1 accuracy (%)", fontsize=10.5)
    _ideal(axis)
    _legend(figure)
    figure.suptitle("AudioMNIST Accuracy vs Total Digital Operations", fontsize=16,
                    fontweight="semibold", y=0.975)
    figure.text(0.5, 0.925,
                "Lines connect each compact architecture's evaluated 16 kHz and 400 Hz front ends",
                ha="center", fontsize=9.2, color="#46545F")
    figure.text(0.5, 0.052,
                "Total = front-end scalar ops/s + NN MAC/s + NN auxiliary scalar ops/s. ADC conversions excluded.",
                ha="center", fontsize=8.4, color="#566570")
    _save_figure(figure, OUTPUT_OPS_PNG, OUTPUT_OPS_PDF)
    plt.close(figure)


def plot_energy(data: pd.DataFrame) -> None:
    figure, axis = plt.subplots(figsize=(10.2, 6.5))
    figure.subplots_adjust(left=0.105, right=0.975, top=0.82, bottom=0.19)
    column = "weighted_dynamic_energy_proxy_uj_per_5s"
    _draw_points(axis, data, column)
    maximum = float(data[column].max())
    axis.set_xlim(0, maximum * 1.08)
    axis.set_ylim(50, 102)
    axis.yaxis.set_major_locator(MultipleLocator(5))
    axis.xaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value:,.0f}"))
    axis.grid(True, color="#D4D8DB", linewidth=0.65, alpha=0.75)
    axis.set_axisbelow(True)
    axis.set_xlabel("Weighted dynamic energy proxy (uJ per 5 s audio, linear scale)", fontsize=10.5)
    axis.set_ylabel("Held-out top-1 accuracy (%)", fontsize=10.5)
    _ideal(axis)
    _legend(figure)
    figure.suptitle("AudioMNIST Accuracy vs Dynamic Energy Proxy", fontsize=16,
                    fontweight="semibold", y=0.975)
    figure.text(0.5, 0.925,
                "Five-second analytical ADC + FP32 compute proxy; not measured hardware energy",
                ha="center", fontsize=9.2, color="#46545F")
    figure.text(0.5, 0.052,
                "Includes ADC conversions and arithmetic; excludes memory, static power, and unknown analog-envelope energy.",
                ha="center", fontsize=8.4, color="#566570")
    _save_figure(figure, OUTPUT_ENERGY_PNG, OUTPUT_ENERGY_PDF)
    plt.close(figure)


def write_documentation(data: pd.DataFrame, sensitivity_summary: pd.DataFrame) -> None:
    assumptions = {
        "task": "AudioMNIST isolated one-second digit classification",
        "analysis_duration_seconds": ANALYSIS_SECONDS,
        "operation_equation": "front-end scalar ops/s + NN MAC/s + NN auxiliary scalar ops/s",
        "energy_equation": "5 * (ADC + weighted front end + weighted NN MAC + weighted NN auxiliary)",
        "coefficients_pj": {
            "adc_sample": ADC_ENERGY_PJ_PER_SAMPLE,
            "fp32_add": ADD_PJ,
            "fp32_multiply": MULTIPLY_PJ,
            "fp32_mac": MAC_PJ,
            "comparison": COMPARE_PJ,
            "nonlinear_call": NONLINEAR_PJ,
        },
        "coefficient_sources_and_assumptions": {
            "arithmetic": {
                "source": "Mark Horowitz, Computing's Energy Problem, ISSCC 2014",
                "technology": "rough 45 nm estimates",
                "doi": "https://doi.org/10.1109/ISSCC.2014.6757323",
                "fp32_add_pj": 0.9,
                "fp32_multiply_pj": 4.0,
                "fp32_mac_pj": 4.9,
                "int8_add_pj": 0.03,
                "int8_multiply_pj": 0.20,
                "int8_mac_pj": 0.23,
                "8kb_sram_64bit_access_pj": 10.0,
                "dram_64bit_access_pj_lower_bound": 1300.0,
            },
            "nonlinear": "20 MAC equivalents per call; explicit assumption, not a measured coefficient",
            "comparison": "set equal to one add; explicit assumption",
            "adc": {
                "equation": "E_sample = 2^ENOB * Walden FoM",
                "enob_bits": 12,
                "walden_fom_fj_per_conversion_step": 20.0,
                "energy_per_sample_pj": ADC_ENERGY_PJ_PER_SAMPLE,
            },
            "stm32u5": {
                "source": "STM32U585 datasheet",
                "url": "https://www.st.com/resource/en/datasheet/stm32u585ai.pdf",
                "run_current_ua_per_mhz": 19.5,
                "voltage_v": 3.3,
                "derived_energy_per_cycle_pj": 64.35,
                "warning": "one cycle per scalar/MAC and 20 cycles per nonlinear is illustrative, not an instruction trace",
            },
        },
        "logmel_frontend": {
            "input_hz": 16_000,
            "frames_per_fixed_one_second_example": 96,
            "window_ms": 50,
            "hop_ms": 10,
            "fft_size": 800,
            "normalization": "40-bin train-only mean/std; 2 scalar ops/value",
        },
        "digital_rms_frontend": {
            "source_hz": 48_000,
            "outputs_per_second": 400,
            "window_ms": 10,
            "hop_ms": 2.5,
            "scalar_ops_per_second": 145_200,
            "convention": "3 running-RMS arithmetic ops/source sample + 400 sqrt + 800 normalization ops",
        },
        "fabricated_projection": {
            "adc_channels": 5,
            "sample_rate_hz_per_channel": 400,
            "digital_frontend_ops_per_second": 0,
            "warning": "accuracy evaluated from digitally extracted physical-capture envelopes; model-ready analog envelope front end is projected",
        },
        "raw_400hz_control": {
            "adc_channels": 1,
            "sample_rate_hz": 400,
            "digital_normalization_ops_per_second": 800,
            "accuracy_generation": "offline scipy.signal.resample_poly from 48 kHz with anti-aliasing",
            "deployment_interpretation": "direct 400 Hz ADC after an analog anti-alias filter",
            "warning": "analog anti-alias-filter energy is unknown and excluded",
        },
        "nn_profile": {
            "mac_layers": "Conv1d, Conv2d, Linear, and GRU matrix products",
            "auxiliary": "bias, activation, residual, reduction, non-foldable normalization, LayerNorm, GRU gates/state updates, and top-1 argmax",
            "batch_norm": "folded into preceding convolution at inference except SubSpectralNorm",
            "top1_decoder": "argmax comparisons only; top-3 and softmax excluded",
        },
        "excluded": [
            "dataset file I/O, crop/pad memory copies, and offline waveform resampling/peak conditioning",
            "parameter memory traffic and activation memory traffic",
            "static/leakage/platform power",
            "unknown analog resonator and envelope-extractor energy",
        ],
        "source_sha256": {
            str(LOGMEL_RESULTS.relative_to(ROOT)): _hash(LOGMEL_RESULTS),
            str(RMS_RESULTS.relative_to(ROOT)): _hash(RMS_RESULTS),
            str(POOLED_TCN_SUMMARY.relative_to(ROOT)): _hash(POOLED_TCN_SUMMARY),
            str(CONV_GRU_SUMMARY.relative_to(ROOT)): _hash(CONV_GRU_SUMMARY),
            str(RAW400_TCN_SUMMARY.relative_to(ROOT)): _hash(RAW400_TCN_SUMMARY),
        },
        "sensitivity_summary": sensitivity_summary.to_dict(orient="records"),
    }
    OUTPUT_ASSUMPTIONS.write_text(json.dumps(assumptions, indent=2) + "\n", encoding="ascii")

    lines = [
        "# AudioMNIST accuracy-energy methodology",
        "",
        "## Scope",
        "",
        "Accuracy is speaker-disjoint held-out top-1 accuracy for one-second isolated-digit examples. The operation rate is the work for one fixed one-second input. The energy proxy integrates that rate over five seconds to match the companion alphabet analysis.",
        "",
        "## Equations",
        "",
        "```text",
        "total digital ops/s = front-end scalar ops/s + NN MAC/s + NN auxiliary scalar ops/s",
        "E_5s = 5 * (ADC energy/s + front-end energy/s + NN MAC energy/s + NN auxiliary energy/s)",
        "```",
        "",
        "The total-digital-operations plot excludes ADC conversions. The energy plot includes the analytical ADC term.",
        "",
        "## Front ends",
        "",
        "- The 16 kHz implementation uses 96 frames for a fixed one-second, center=False input: 50 ms windows, 10 ms hop, 800-point FFT, 401 power bins, 40 Mel filters, logarithm, and frozen feature normalization.",
        "- The evaluated digital RMS implementation uses a 48 kHz waveform, a causal 10 ms running-RMS window, 2.5 ms hop, 400 square roots/s, and frozen scalar normalization. Its notebook-defined deployment count is 145,200 scalar ops/s.",
        "- The fabricated stars assume five model-ready physical envelope channels sampled at 400 Hz. Their zero digital front-end count is projected; the held-out accuracies were evaluated using digitally extracted envelopes from physical captures.",
        "- The no-resonator control was evaluated by anti-aliased offline resampling of one signed AudioMNIST waveform channel from 48 kHz to 400 Hz. Its plotted deployment point assumes a direct 400 Hz ADC plus 800 scalar normalization ops/s; required analog anti-alias-filter energy is unknown and excluded.",
        "",
        "## Double-counting audit",
        "",
        "Every compact model is instantiated from the exact generated AudioMNIST notebook source. Conv/Linear/GRU matrix products are counted once as MACs and asserted against stored artifact values. Auxiliary counts contain only work omitted by that MAC profiler. BatchNorm is folded at inference except SubSpectralNorm; recurrent matrix operations are not repeated in auxiliary operations.",
        "",
        "## Coefficients and limits",
        "",
        "FP32 add (0.9 pJ), multiply (4.0 pJ), MAC (4.9 pJ), 8 KB SRAM (10 pJ per 64-bit access), and DRAM (1.3 nJ per 64-bit access lower bound) are rough 45 nm estimates from Horowitz, ISSCC 2014. The nonlinear cost of 20 MAC equivalents, comparison-equals-add rule, and 12-bit 20 fJ/conversion-step ADC are explicit analytical assumptions. These are not measured AudioMNIST hardware values.",
        "",
        "Parameter count is excluded from the primary proxy. Parameter and per-MAC memory reads appear only in named sensitivity scenarios because actual traffic depends on precision, cache, tiling, and dataflow.",
        "",
        "## Sensitivity",
        "",
        *[
            f"- `{row.scenario}`: rank correlation {row.spearman_rank_correlation_vs_primary:.3f}; lowest point {row.lowest_energy_model} ({row.lowest_energy_representation})."
            for row in sensitivity_summary.itertuples(index=False)
        ],
        "",
        "See `audiomnist_accuracy_energy_and_ops_assumptions.json`, `audiomnist_dynamic_energy_proxy_sensitivity.csv`, and the main CSV for coefficients, hashes, component counts, and exclusions.",
    ]
    OUTPUT_METHOD.write_text("\n".join(lines) + "\n", encoding="ascii")


def main() -> None:
    FIGURES.mkdir(parents=True, exist_ok=True)
    data = build_table()
    table = required_table(data)
    sensitivity, sensitivity_summary = build_sensitivity(data)
    data.to_csv(OUTPUT_CSV, index=False)
    data.to_csv(LEGACY_ENERGY_CSV, index=False)
    sensitivity.to_csv(OUTPUT_SENSITIVITY, index=False)
    write_table(table)
    LEGACY_ENERGY_TABLE.write_text(OUTPUT_TABLE.read_text(encoding="ascii"), encoding="ascii")
    write_documentation(data, sensitivity_summary)
    print(table.to_string(index=False))
    configure_plot()
    plot_energy(data)
    plot_ops(data)
    print(f"\nCSV:        {OUTPUT_CSV}")
    print(f"Table:      {OUTPUT_TABLE}")
    print(f"Energy PNG: {OUTPUT_ENERGY_PNG}")
    print(f"Energy PDF: {OUTPUT_ENERGY_PDF}")
    print(f"Ops PNG:    {OUTPUT_OPS_PNG}")
    print(f"Ops PDF:    {OUTPUT_OPS_PDF}")


if __name__ == "__main__":
    main()
