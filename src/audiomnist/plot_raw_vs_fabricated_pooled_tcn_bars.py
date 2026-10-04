#!/usr/bin/env python3
"""Compare the direct-waveform and fabricated-envelope AudioMNIST Pooled TCNs."""

from __future__ import annotations

import argparse
import json
import os
import platform
import time
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import torch
from matplotlib.lines import Line2D

import train_direct_400hz_5d_pooled_tcn_classifier as fabricated_training
import train_raw_waveform_400hz_pooled_tcn as raw_training


HERE = Path(__file__).resolve().parent
FIGURES = HERE / "figures"
RAW_ROOT = HERE / "models" / "raw_waveform_400hz_pooled_tcn"
FABRICATED_ROOT = HERE / "models" / "envelope_400hz_5d_pooled_tcn_classifier"
RAW_SUMMARY = RAW_ROOT / "heldout_test_summary.json"
FABRICATED_SUMMARY = FABRICATED_ROOT / "heldout_test" / "heldout_test_summary.json"
RAW_CHECKPOINT = RAW_ROOT / "best_raw_waveform_400hz_pooled_tcn.pt"
FABRICATED_CHECKPOINT = FABRICATED_ROOT / "best_envelope_400hz_5d_pooled_tcn_classifier.pt"

OUTPUT_CSV = FIGURES / "audiomnist_raw_vs_fabricated_pooled_tcn_comparison.csv"
OUTPUT_JSON = FIGURES / "audiomnist_raw_vs_fabricated_pooled_tcn_latency_protocol.json"
OUTPUT_PNG = FIGURES / "audiomnist_raw_vs_fabricated_pooled_tcn_bars.png"
OUTPUT_PDF = FIGURES / "audiomnist_raw_vs_fabricated_pooled_tcn_bars.pdf"

COLORS = {"Direct waveform": "#5B5F97", "Fabricated envelopes": "#D1495B"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--warmup", type=int, default=100)
    parser.add_argument("--repeats", type=int, default=1000)
    return parser.parse_args()


def load_model(checkpoint_path: Path, *, fabricated: bool) -> torch.nn.Module:
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    dropout = float(checkpoint.get("model_config", {}).get("dropout", 0.10))
    if fabricated:
        model = fabricated_training.Direct400PooledTcnClassifier(dropout=dropout)
    else:
        model = raw_training.Raw400PooledTcnClassifier(dropout=dropout)
    model.load_state_dict(checkpoint["model_state"])
    return model.cpu().eval()


@torch.inference_mode()
def paired_latency(
    raw_model: torch.nn.Module,
    fabricated_model: torch.nn.Module,
    *,
    warmup: int,
    repeats: int,
) -> dict[str, np.ndarray]:
    if warmup < 0 or repeats <= 0:
        raise ValueError("Warmup must be nonnegative and repeats must be positive")
    previous_threads = torch.get_num_threads()
    torch.set_num_threads(1)
    inputs = {
        "Direct waveform": (raw_model, torch.zeros(1, 400, 1), torch.tensor([400])),
        "Fabricated envelopes": (
            fabricated_model,
            torch.zeros(1, 400, 5),
            torch.tensor([400]),
        ),
    }
    try:
        for _ in range(warmup):
            for model, values, lengths in inputs.values():
                model(values, lengths)
        samples = {name: np.empty(repeats, dtype=np.float64) for name in inputs}
        for index in range(repeats):
            # Alternate order to reduce drift or thermal bias between the models.
            order = list(inputs) if index % 2 == 0 else list(reversed(inputs))
            for name in order:
                model, values, lengths = inputs[name]
                started = time.perf_counter_ns()
                model(values, lengths)
                samples[name][index] = (time.perf_counter_ns() - started) / 1e6
    finally:
        torch.set_num_threads(previous_threads)
    return samples


def build_table(warmup: int, repeats: int) -> pd.DataFrame:
    required = (RAW_SUMMARY, FABRICATED_SUMMARY, RAW_CHECKPOINT, FABRICATED_CHECKPOINT)
    missing = [str(path) for path in required if not path.is_file()]
    if missing:
        raise FileNotFoundError("Missing comparison artifacts:\n" + "\n".join(missing))

    raw_summary = json.loads(RAW_SUMMARY.read_text(encoding="ascii"))
    fabricated_summary = json.loads(FABRICATED_SUMMARY.read_text(encoding="ascii"))
    raw_model = load_model(RAW_CHECKPOINT, fabricated=False)
    fabricated_model = load_model(FABRICATED_CHECKPOINT, fabricated=True)
    samples = paired_latency(
        raw_model, fabricated_model, warmup=warmup, repeats=repeats
    )

    rows = [
        {
            "system": "Direct waveform",
            "representation": "one signed anti-aliased 400 Hz waveform channel",
            "accuracy_percent": float(raw_summary["test_top1_accuracy_percent"]),
            "latency_median_ms": float(np.median(samples["Direct waveform"])),
            "latency_p95_ms": float(np.percentile(samples["Direct waveform"], 95)),
            "parameters": sum(parameter.numel() for parameter in raw_model.parameters()),
            "nn_macs_per_one_second": int(
                raw_summary["profiled_macs_per_one_second_utterance"]
            ),
            "front_end_dsp_ops_per_second": 800,
            "front_end_dsp_definition": "400 train-frozen subtracts + 400 multiplies/s",
            "accuracy_status": "evaluated held-out test",
            "deployment_status": "direct 400 Hz ADC and analog anti-alias filter projected",
        },
        {
            "system": "Fabricated envelopes",
            "representation": "five model-ready physical 400 Hz RMS-envelope channels",
            "accuracy_percent": 100.0 * float(fabricated_summary["test_top1_accuracy"]),
            "latency_median_ms": float(np.median(samples["Fabricated envelopes"])),
            "latency_p95_ms": float(np.percentile(samples["Fabricated envelopes"], 95)),
            "parameters": sum(
                parameter.numel() for parameter in fabricated_model.parameters()
            ),
            "nn_macs_per_one_second": int(
                fabricated_summary["learned_macs_per_one_second_utterance"]
            ),
            "front_end_dsp_ops_per_second": 0,
            "front_end_dsp_definition": "model-ready physical envelopes; no digital extraction",
            "accuracy_status": "evaluated held-out physical-capture test",
            "deployment_status": "model-ready analog envelope front end projected POC",
        },
    ]
    table = pd.DataFrame(rows)
    expected_parameters = {"Direct waveform": 9_547, "Fabricated envelopes": 9_807}
    expected_macs = {"Direct waveform": 791_040, "Fabricated envelopes": 816_640}
    for row in table.itertuples(index=False):
        if row.parameters != expected_parameters[row.system]:
            raise RuntimeError(f"{row.system}: parameter audit failed")
        if row.nn_macs_per_one_second != expected_macs[row.system]:
            raise RuntimeError(f"{row.system}: MAC audit failed")
    return table


def _value_label(axis, bars, values, formatter) -> None:
    maximum = max(values) if max(values) > 0 else 1.0
    for bar, value in zip(bars, values):
        axis.text(
            bar.get_x() + bar.get_width() / 2,
            value + 0.035 * maximum,
            formatter(value),
            ha="center",
            va="bottom",
            fontsize=8.2,
            fontweight="semibold",
            color="#263640",
        )


def plot(table: pd.DataFrame) -> None:
    plt.rcParams.update(
        {
            "font.family": "DejaVu Sans",
            "axes.spines.top": False,
            "axes.spines.right": False,
            "axes.edgecolor": "#7D8992",
            "axes.labelcolor": "#263640",
            "xtick.color": "#42515B",
            "ytick.color": "#42515B",
            "figure.facecolor": "white",
            "axes.facecolor": "white",
        }
    )
    figure, axes = plt.subplots(2, 3, figsize=(10.5, 6.8))
    figure.subplots_adjust(left=0.075, right=0.975, top=0.79, bottom=0.145,
                           wspace=0.34, hspace=0.48)
    systems = table["system"].tolist()
    colors = [COLORS[name] for name in systems]
    x = np.arange(2)
    specifications = [
        ("accuracy_percent", "Held-out accuracy", "Accuracy (%)", lambda value: f"{value:.2f}%"),
        ("latency_median_ms", "Model latency", "Median latency (ms)", lambda value: f"{value:.2f}"),
        ("parameters", "Model parameters", "Parameters", lambda value: f"{int(value):,}"),
        ("nn_macs_per_one_second", "Neural-network MACs", "MACs / 1 s input", lambda value: f"{int(value):,}"),
        ("front_end_dsp_ops_per_second", "Front-end DSP cost", "Scalar ops/s", lambda value: f"{int(value):,}"),
    ]
    for axis, (column, title, ylabel, formatter) in zip(axes.flat, specifications):
        values = table[column].to_numpy(np.float64)
        bars = axis.bar(x, values, width=0.58, color=colors, edgecolor="white", linewidth=0.9)
        if column == "latency_median_ms":
            upper = table["latency_p95_ms"].to_numpy(np.float64) - values
            axis.errorbar(x, values, yerr=np.vstack([np.zeros(2), upper]), fmt="none",
                          ecolor="#263640", elinewidth=1.0, capsize=3)
        axis.set_title(title, fontsize=10.3, fontweight="semibold")
        axis.set_ylabel(ylabel, fontsize=8.6)
        axis.set_xticks([])
        upper_limit = (
            table["latency_p95_ms"].max() * 1.12
            if column == "latency_median_ms"
            else values.max() * 1.24
        )
        axis.set_ylim(0, max(upper_limit, 1.0))
        axis.grid(axis="y", color="#D4D8DB", linewidth=0.65, alpha=0.75)
        axis.set_axisbelow(True)
        axis.tick_params(labelsize=8)
        _value_label(axis, bars, values, formatter)

    summary_axis = axes.flat[-1]
    summary_axis.axis("off")
    raw = table.set_index("system").loc["Direct waveform"]
    fabricated = table.set_index("system").loc["Fabricated envelopes"]
    accuracy_gain = fabricated.accuracy_percent - raw.accuracy_percent
    digital_change = 100.0 * (
        (fabricated.nn_macs_per_one_second + fabricated.front_end_dsp_ops_per_second)
        / (raw.nn_macs_per_one_second + raw.front_end_dsp_ops_per_second)
        - 1.0
    )
    summary_axis.text(0.5, 0.68, f"+{accuracy_gain:.2f} pp", ha="center", va="center",
                      fontsize=19, fontweight="semibold", color="#245E49")
    summary_axis.text(0.5, 0.49, "accuracy with resonator envelopes", ha="center",
                      fontsize=8.6, color="#46545F")
    summary_axis.text(0.5, 0.27, f"{digital_change:+.2f}%", ha="center", va="center",
                      fontsize=15, fontweight="semibold", color="#263640")
    summary_axis.text(0.5, 0.11, "NN MAC + front-end DSP change", ha="center",
                      fontsize=8.2, color="#46545F")

    handles = [
        Line2D([0], [0], marker="s", color="none", markerfacecolor=COLORS[name],
               markeredgecolor="white", markersize=8, label=name)
        for name in systems
    ]
    figure.legend(handles=handles, loc="upper center", bbox_to_anchor=(0.5, 0.88),
                  ncol=2, frameon=False, fontsize=9)
    figure.suptitle("Same Pooled TCN: Direct 400 Hz Waveform vs Fabricated Envelopes",
                    fontsize=15, fontweight="semibold", y=0.975)
    figure.text(0.5, 0.925,
                "Speaker-disjoint AudioMNIST test; paired batch-one CPU timing with one PyTorch thread",
                ha="center", fontsize=9, color="#46545F")
    figure.text(0.5, 0.045,
                "Latency bars show median with p95 whiskers. DSP cost excludes ADC, memory, and unknown analog filter/envelope energy.",
                ha="center", fontsize=8.1, color="#566570")
    figure.savefig(OUTPUT_PNG, dpi=320, bbox_inches="tight")
    figure.savefig(OUTPUT_PDF, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    FIGURES.mkdir(parents=True, exist_ok=True)
    table = build_table(args.warmup, args.repeats)
    table.to_csv(OUTPUT_CSV, index=False)
    protocol = {
        "latency_device": platform.processor() or platform.machine(),
        "torch_version": torch.__version__,
        "torch_threads": 1,
        "batch_size": 1,
        "input_duration_seconds": 1.0,
        "warmup": args.warmup,
        "repeats": args.repeats,
        "paired_same_process": True,
        "alternating_execution_order": True,
        "file_io_and_front_end_excluded": True,
        "latency_statistic": "median; p95 retained in CSV and shown as upper whisker",
    }
    OUTPUT_JSON.write_text(json.dumps(protocol, indent=2) + "\n", encoding="ascii")
    plot(table)
    print(table.to_string(index=False))
    print(f"\nCSV:  {OUTPUT_CSV}")
    print(f"PNG:  {OUTPUT_PNG}")
    print(f"PDF:  {OUTPUT_PDF}")
    print(f"JSON: {OUTPUT_JSON}")


if __name__ == "__main__":
    main()
