"""Plot one measured AudioMNIST sample and its five 100 Hz RMS envelopes."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.lines import Line2D

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE = (
    REPO_ROOT
    / "raspberry_pi_audiomnist_benchmark_bundle"
    / "data"
    / "fabricated"
    / "captures"
    / "test"
    / "20260905_225221_audiomnist190_repeat01"
)
OUT_DIR = REPO_ROOT / "final"

DIGIT = 9
SPEAKER_ID = 26
REPETITION = 38
CONTEXT_S = 0.12

# Acquisition order (AI0..AI4) rearranged into ascending resonator frequency,
# matching make_icassp_two_speaker_two_digit_sensor_plot.py. The first field is
# the DAQ channel index, which still indexes the raw capture columns. Colors are
# the per-frequency assignment used by the pipeline diagram, so a given
# resonator is the same color in every figure.
CHANNELS = (
    (0, "R300", "#0072B2"),
    (3, "R1800", "#D55E00"),
    (1, "R3500", "#009E73"),
    (4, "R4500", "#CC79A7"),
    (2, "R7500", "#E69F00"),
)


def nice_tick(limit: float) -> float:
    """Largest value of the form {1,2,5} x 10^k that still fits under limit.

    Using one rounding rule for every panel keeps the tick labels a short,
    predictable set instead of five independently chosen scales.
    """
    exponent = np.floor(np.log10(limit))
    for mantissa in (5.0, 2.0, 1.0):
        candidate = mantissa * 10.0**exponent
        if candidate <= limit:
            return candidate
    return 10.0**exponent


def minmax_trace(
    times: np.ndarray, values: np.ndarray, max_bins: int = 4500
) -> tuple[np.ndarray, np.ndarray]:
    """Reduce a dense trace while retaining each display bin's extrema."""
    if len(values) <= 2 * max_bins:
        return times, values

    bin_size = int(np.ceil(len(values) / max_bins))
    padded_length = int(np.ceil(len(values) / bin_size) * bin_size)
    values_padded = np.pad(values, (0, padded_length - len(values)), constant_values=np.nan)
    times_padded = np.pad(times, (0, padded_length - len(times)), constant_values=np.nan)
    value_bins = values_padded.reshape(-1, bin_size)
    time_bins = times_padded.reshape(-1, bin_size)

    minima = np.nanargmin(value_bins, axis=1)
    maxima = np.nanargmax(value_bins, axis=1)
    rows = np.arange(len(value_bins))
    first_is_minimum = minima <= maxima
    first = np.where(first_is_minimum, minima, maxima)
    second = np.where(first_is_minimum, maxima, minima)
    reduced_times = np.column_stack((time_bins[rows, first], time_bins[rows, second])).ravel()
    reduced_values = np.column_stack(
        (value_bins[rows, first], value_bins[rows, second])
    ).ravel()
    valid = np.isfinite(reduced_times) & np.isfinite(reduced_values)
    return reduced_times[valid], reduced_values[valid]


def main() -> None:
    with np.load(CAPTURE / "raw_capture.npz", allow_pickle=False) as archive:
        volts = np.asarray(archive["volts_v"], dtype=np.float64)
        sample_rate_hz = float(archive["sample_rate_hz"])

    # 100 Hz envelopes with real-DAQ-rate window-center timestamps, the
    # classifier input; columns are in acquisition order AI0..AI4.
    with np.load(CAPTURE / "envelopes_100Hz_ablation_capture.npz", allow_pickle=False) as archive:
        if float(archive["sample_rate_hz"]) != 100.0:
            raise RuntimeError("expected 100 Hz envelopes")
        envelope_all_times = np.asarray(archive["time_s"], dtype=np.float64)
        envelope_all_v = np.asarray(archive["envelope_v"], dtype=np.float64)
    events = pd.read_csv(CAPTURE / "aligned_events.csv")
    speaker_ids = pd.to_numeric(events["source_speaker_id"], errors="coerce")
    repetitions = pd.to_numeric(events["source_repetition"], errors="coerce")
    event = events.loc[
        events["event_type"].eq("letter")
        & events["label"].astype(str).eq(str(DIGIT))
        & speaker_ids.eq(float(SPEAKER_ID))
        & repetitions.eq(float(REPETITION))
    ].iloc[0]

    event_start_s = float(event["start_s"])
    event_end_s = float(event["end_s"])
    plot_start_s = max(0.0, event_start_s - CONTEXT_S)
    plot_end_s = min(len(volts) / sample_rate_hz, event_end_s + CONTEXT_S)

    first_sample = int(np.floor(plot_start_s * sample_rate_hz))
    last_sample = int(np.ceil(plot_end_s * sample_rate_hz))
    raw_times = np.arange(first_sample, last_sample, dtype=np.float64) / sample_rate_hz
    raw_times -= plot_start_s
    segment = volts[first_sample:last_sample]
    segment -= np.median(segment, axis=0, keepdims=True)

    envelope_mask = (envelope_all_times >= plot_start_s) & (envelope_all_times <= plot_end_s)
    envelope_times = envelope_all_times[envelope_mask]
    envelope_times -= plot_start_s
    event_start_relative_s = event_start_s - plot_start_s
    event_end_relative_s = event_end_s - plot_start_s
    plotted_duration_s = plot_end_s - plot_start_s

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )
    # Sized for a single ICASSP column (3.39 in) so it is placed at native
    # scale and the type stays at the size set above.
    fig, axes = plt.subplots(5, 1, figsize=(3.39, 2.75), sharex=True)
    fig.subplots_adjust(left=0.155, right=0.985, bottom=0.185, top=0.99, hspace=0.16)

    for axis, (channel, label, color) in zip(axes, CHANNELS, strict=True):
        trace_time, trace_mv = minmax_trace(raw_times, 1000.0 * segment[:, channel])
        rms_mv = 1000.0 * envelope_all_v[envelope_mask, channel]

        axis.axvspan(event_start_relative_s, event_end_relative_s, color="#EFF2F5", zorder=0)
        axis.axvline(event_start_relative_s, color="#BFC5CC", linewidth=0.7, zorder=1)
        axis.axvline(event_end_relative_s, color="#BFC5CC", linewidth=0.7, zorder=1)
        axis.plot(
            trace_time,
            trace_mv,
            color="#525C66",
            linewidth=0.34,
            alpha=0.38,
            rasterized=True,
        )
        axis.fill_between(envelope_times, -rms_mv, rms_mv, color=color, alpha=0.19, linewidth=0)
        axis.plot(envelope_times, rms_mv, color=color, linewidth=1.15)
        axis.plot(envelope_times, -rms_mv, color=color, linewidth=1.15)
        axis.axhline(0.0, color="#6D7680", linewidth=0.55, alpha=0.7)

        limit = max(
            float(np.quantile(np.abs(trace_mv), 0.995)),
            float(np.quantile(rms_mv, 0.995)),
            1.0,
        )
        axis.set_ylim(-1.18 * limit, 1.18 * limit)

        # Label only 0 and the positive peak. The trace is symmetric about 0,
        # so the negative tick adds no information, and dropping it means a
        # panel's lowest label is its centered 0 -- which cannot collide with
        # the upper label of the panel below.
        tick = nice_tick(limit)
        axis.set_yticks([0.0, tick])
        axis.set_yticklabels(["0", f"{tick:g}"])
        axis.tick_params(axis="y", length=2, pad=1.5)

        # The channel name sits inside the panel, over the leading silence, so
        # the left margin carries only the shared amplitude axis.
        axis.text(
            0.011,
            0.94,
            label,
            transform=axis.transAxes,
            ha="left",
            va="top",
            fontsize=5.8,
            weight="bold",
            color=color,
            zorder=5,
        )
        axis.grid(axis="y", color="#D9DDE2", linewidth=0.5, alpha=0.7)
        axis.spines[["top", "right"]].set_visible(False)

    axes[-1].set_xlim(0.0, plotted_duration_s)
    axes[-1].set_xlabel("Time (s)", labelpad=2)
    fig.supylabel("Amplitude (mV)", x=0.013, fontsize=7)

    legend = [
        Line2D([0], [0], color="#525C66", linewidth=0.8, alpha=0.65, label="Measured waveform"),
        Line2D(
            [0],
            [0],
            color=CHANNELS[0][2],
            linewidth=1.5,
            label="10 ms RMS envelope (100 Hz)",
        ),
    ]
    fig.legend(
        handles=legend,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.0),
        ncol=2,
        frameon=False,
        fontsize=5.8,
        handlelength=1.6,
        columnspacing=1.2,
    )

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "icassp_audiomnist_digit9_resonator_envelopes.png"
    pdf_path = OUT_DIR / "icassp_audiomnist_digit9_resonator_envelopes.pdf"
    fig.savefig(png_path, dpi=300, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
