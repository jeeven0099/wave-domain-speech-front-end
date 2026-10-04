"""ICASSP-format side-by-side comparison figure: one audio sample (digit 9,
speaker 26, repetition 38) rendered through the three signal paths compared
in the paper.

  (a) 16 kHz log-Mel spectrogram          -- exact LogMelFrontend math
  (b) 1-channel digital 400 Hz RMS envelope -- exact preprocess_400hz_envelope math
  (c) 5-channel fabricated 100 Hz AC-RMS envelope -- _frame_ac_rms_400hz math with a
      10 ms hop (the classifier input of the paper's main system),
      sliced from the real analog-resonator DAQ capture

All preprocessing here is copied verbatim (not reimplemented/approximated)
from raspberry_pi_audiomnist_benchmark_bundle/benchmark_audiomnist_raspberry_pi.py
so the figure is trustworthy for publication.
"""

from __future__ import annotations

import csv
import json
import math
import os
from pathlib import Path

os.environ.setdefault("KMP_DUPLICATE_LIB_OK", "TRUE")

import numpy as np
import torch
import torchaudio
from scipy import signal
from scipy.io import wavfile

REPO_ROOT = Path(__file__).resolve().parents[2]
BUNDLE = REPO_ROOT / "raspberry_pi_audiomnist_benchmark_bundle"
# Digit 7 rather than 9: in "seven" the fricative and the voiced vowel excite
# different resonators at different times, so all five channels of (c) carry
# visible structure, where the "nine" lit up R1800 alone.
FAB_RATE_HZ = 100.0
DIGIT = 7
SPEAKER = 26
REPETITION = 24
WAV_PATH = BUNDLE / "data" / "AudioMNIST" / "data" / str(SPEAKER) / f"{DIGIT}_{SPEAKER}_{REPETITION}.wav"
CAPTURE_DIR = (
    BUNDLE
    / "data"
    / "fabricated"
    / "captures"
    / "test"
    / "20260905_225453_audiomnist191_repeat01"
)
OUT_DIR = REPO_ROOT / "fabrication" / "audiomnist_daq_experiment" / "figures"
OUT_DIR.mkdir(parents=True, exist_ok=True)

TARGET_PEAK_DBFS = -9.0
WINDOW_S = 1.0

# Capture column order. Panel (c) re-sorts the channels by frequency.
RESONATOR_LABELS = ("R300", "R3500", "R7500", "R1800", "R4500")
RESONATOR_FREQUENCIES_HZ = np.asarray((300, 3500, 7500, 1800, 4500), dtype=np.float64)
ENVELOPE_COLUMNS = (
    "AI0_R300_rms_v",
    "AI1_R3500_rms_v",
    "AI2_R7500_rms_v",
    "AI3_R1800_rms_v",
    "AI4_R4500_rms_v",
)


# ---------------------------------------------------------------------------
# Verbatim helper functions copied from benchmark_audiomnist_raspberry_pi.py
# ---------------------------------------------------------------------------
def _resample_poly(values: np.ndarray, source_rate: int, target_rate: int) -> np.ndarray:
    if source_rate == target_rate:
        return values.astype(np.float32, copy=False)
    divisor = math.gcd(source_rate, target_rate)
    return signal.resample_poly(
        values, target_rate // divisor, source_rate // divisor
    ).astype(np.float32)


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


def _frame_ac_rms_400hz(
    values: np.ndarray, sample_rate_hz: float, rate_hz: float = 400.0
) -> np.ndarray:
    frame_length = max(1, int(round(sample_rate_hz * 10.0 / 1_000.0)))
    hop_length = max(1, int(round(sample_rate_hz / rate_hz)))
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


# ---------------------------------------------------------------------------
# (a) 16 kHz log-Mel spectrogram
# ---------------------------------------------------------------------------
def compute_logmel(wav_path: Path) -> tuple[np.ndarray, float, float]:
    source_rate, pcm = wavfile.read(wav_path)
    waveform = pcm.astype(np.float32) / 32768.0
    resampled = _resample_poly(waveform, int(source_rate), 16_000)
    resampled = resampled.astype(np.float16).astype(np.float32)
    fixed = _crop_or_pad(_peak_normalize(resampled), 16_000)
    mel = torchaudio.transforms.MelSpectrogram(
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
    tensor = torch.from_numpy(np.ascontiguousarray(fixed)).unsqueeze(0)
    log_mel = torch.log(mel(tensor).clamp_min(1e-8)).squeeze(0).numpy()  # [40, frames]
    frame_duration_s = 160 / 16_000
    total_duration_s = 16_000 / 16_000
    return log_mel, frame_duration_s, total_duration_s


def htk_mel_bin_edges_hz(
    n_mels: int, f_min_hz: float, f_max_hz: float
) -> np.ndarray:
    """Return display edges centered on the HTK Mel filterbank frequencies."""
    hz_to_mel = lambda hz: 2595.0 * np.log10(1.0 + hz / 700.0)
    mel_to_hz = lambda mel: 700.0 * (10.0 ** (mel / 2595.0) - 1.0)
    mel_points = np.linspace(hz_to_mel(f_min_hz), hz_to_mel(f_max_hz), n_mels + 2)
    centers_hz = mel_to_hz(mel_points)[1:-1]
    edges_hz = np.empty(n_mels + 1, dtype=np.float64)
    edges_hz[1:-1] = 0.5 * (centers_hz[:-1] + centers_hz[1:])
    edges_hz[0] = 0.0
    edges_hz[-1] = f_max_hz
    return edges_hz


# ---------------------------------------------------------------------------
# (b) 1-channel digital 400 Hz RMS envelope
# ---------------------------------------------------------------------------
def compute_digital_400hz_envelope(wav_path: Path) -> np.ndarray:
    source_rate, pcm = wavfile.read(wav_path)
    if int(source_rate) != 48_000:
        raise ValueError(f"expected 48 kHz source, got {source_rate}")
    waveform = pcm.astype(np.float32) / 32768.0
    fixed = _crop_or_pad(_peak_normalize(waveform), 48_000)
    window_samples = 480
    end_indices = np.rint((np.arange(400, dtype=np.float64) + 1.0) * 48_000 / 400).astype(np.int64)
    end_indices = np.clip(end_indices, 1, len(fixed))
    start_indices = np.maximum(0, end_indices - window_samples)
    squared = np.square(fixed, dtype=np.float64)
    cumulative = np.concatenate(([0.0], np.cumsum(squared)))
    energy = cumulative[end_indices] - cumulative[start_indices]
    envelope = np.sqrt(np.maximum(energy, 0.0) / window_samples).astype(np.float32)
    return envelope


# ---------------------------------------------------------------------------
# (c) 5-channel fabricated 100 Hz AC-RMS envelope, sliced from the real capture
# ---------------------------------------------------------------------------
def compute_fabricated_envelope(capture_dir: Path) -> tuple[np.ndarray, float, float]:
    metadata = json.loads((capture_dir / "metadata.json").read_text(encoding="ascii"))
    speaker = int(metadata["dataset_speaker_id"])

    with np.load(capture_dir / "raw_capture.npz", allow_pickle=False) as archive:
        volts = np.asarray(archive["volts_v"], dtype=np.float32)
        source_rate = float(archive["sample_rate_hz"])

    frame_length = int(round(source_rate * 0.010))
    hop_length = int(round(source_rate / FAB_RATE_HZ))

    # Real-DAQ-rate window centers of the saved 100 Hz envelopes (the
    # acquisition CSV's time_s assumes exactly 10 ms spacing and drifts).
    with np.load(capture_dir / "envelopes_100Hz_ablation_capture.npz", allow_pickle=False) as archive:
        center_times = np.asarray(archive["time_s"], dtype=np.float64)
        saved_envelope = np.asarray(archive["envelope_v"], dtype=np.float32)
    availability_times = center_times + 0.005

    with (capture_dir / "aligned_events.csv").open("r", encoding="utf-8", newline="") as handle:
        events = [row for row in csv.DictReader(handle) if row["event_type"] == "letter"]
    target = next(
        row
        for row in events
        if int(row["label"]) == DIGIT and int(float(row["source_repetition"])) == REPETITION
    )
    # A full 1 s window, placed the way _crop_or_pad centers the clip in panels
    # (a) and (b): the event is the playback of the whole wav, so its start
    # lands at the same (1 - duration) / 2 offset. The margins are the capture's
    # own inter-digit silence -- the next digit starts 0.15 s after this one
    # ends, and the window's last 70 ms of it reads at the ~0.01 V noise floor.
    event_duration_s = float(target["end_s"]) - float(target["start_s"])
    start_s = float(target["start_s"]) - 0.5 * (WINDOW_S - event_duration_s)
    first = int(np.searchsorted(availability_times, start_s, side="left"))
    last = first + int(round(WINDOW_S * FAB_RATE_HZ))
    if first < 0 or last > len(availability_times):
        raise ValueError("1 s window runs past the capture")
    raw_first = first * hop_length
    raw_last = (last - 1) * hop_length + frame_length
    segment = np.ascontiguousarray(volts[raw_first:raw_last], dtype=np.float32)

    envelope = _frame_ac_rms_400hz(segment, source_rate, FAB_RATE_HZ)
    if not np.allclose(envelope, saved_envelope[first:last], rtol=1e-4, atol=1e-7):
        raise RuntimeError("recomputed envelope does not match the saved 100 Hz envelope")
    duration_s = (last - first) / FAB_RATE_HZ
    print(
        f"fabricated segment: speaker={speaker} digit={DIGIT} repetition={REPETITION} "
        f"frames={len(envelope)} duration_s={duration_s:.4f} window_start_s={start_s:.4f}"
    )
    return envelope, 1.0 / FAB_RATE_HZ, duration_s


def main() -> None:
    log_mel, mel_frame_s, mel_total_s = compute_logmel(WAV_PATH)
    digital_envelope = compute_digital_400hz_envelope(WAV_PATH)
    fabricated_envelope, fab_frame_s, fab_duration_s = compute_fabricated_envelope(CAPTURE_DIR)

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib import ticker

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.labelsize": 7,
            "xtick.labelsize": 6,
            "ytick.labelsize": 6,
            "figure.dpi": 300,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # One ICASSP column. The three panels stack top to bottom on a shared time
    # axis, so only the bottom panel spends height on time ticks.
    fig_width_in = 3.39
    fig_height_in = 3.2
    fig, axes = plt.subplots(
        3,
        1,
        figsize=(fig_width_in, fig_height_in),
        sharex=True,
        gridspec_kw={"height_ratios": (1.15, 0.8, 1.05)},
    )
    # Colorbars are drawn as inset axes rather than with fraction=, which would
    # take the space out of the parent axes and leave (b) wider than (a) and (c).
    fig.subplots_adjust(
        left=0.155, right=0.865, bottom=0.105, top=0.965, hspace=0.035
    )
    common_duration_s = 1.0
    common_time_ticks = np.arange(0.0, common_duration_s + 0.001, 0.25)

    # Different colormaps for the two heatmaps: their units differ (log energy
    # vs volts), and a distinct map marks (c) as measured hardware output rather
    # than a software feature on the same scale as (a).
    HEATMAP_CMAP = "magma"
    HARDWARE_CMAP = "viridis"
    LINE_COLOR = "#0072B2"

    def add_colorbar(axis, mappable, label):
        cax = axis.inset_axes((1.025, 0.0, 0.035, 1.0))
        bar = fig.colorbar(mappable, cax=cax)
        bar.set_label(label, fontsize=6, labelpad=2)
        bar.locator = ticker.MaxNLocator(4)
        bar.update_ticks()
        bar.ax.tick_params(labelsize=5.5, length=1.8, width=0.5, pad=1.2)
        bar.outline.set_linewidth(0.5)
        return bar

    # (a) 16 kHz log-Mel spectrogram
    ax = axes[0]
    n_mels, n_mel_frames = log_mel.shape
    # The y axis is the 40 Mel channels the 16 kHz models consume, one row each,
    # rather than a Hz axis that stretches the high channels into tall bands.
    # The clip below is display only. torch.log(...clamp_min(1e-8)) puts the
    # zero-padded silence at log 1e-8 = -18.4, roughly 25 log units under the
    # speech, so the full range spends most of the colormap on padding and
    # flattens the formants. 12 log units is about 52 dB of visible range.
    mel_top = float(np.max(log_mel))
    im_a = ax.imshow(
        log_mel,
        origin="lower",
        aspect="auto",
        extent=(0.0, n_mel_frames * mel_frame_s, 0.5, n_mels + 0.5),
        cmap=HEATMAP_CMAP,
        vmin=mel_top - 12.0,
        vmax=mel_top,
        interpolation="nearest",
        rasterized=True,
    )
    # With center=False the 96 frames end at 0.96 s. The last 40 ms is
    # zero padding past the clip, so it gets the colormap floor rather than a
    # white strip that reads as missing data.
    ax.set_facecolor(plt.get_cmap(HEATMAP_CMAP)(0.0))
    ax.set_ylabel("Mel channel")
    ax.set_ylim(0.5, n_mels + 0.5)
    ax.set_yticks([1, 10, 20, 30, 40])
    add_colorbar(ax, im_a, "log energy")

    # (b) 1-channel digital 400 Hz RMS envelope. This one stays a line rather
    # than a band: it carries no frequency information at all, and drawing it
    # across the 0-8 kHz axis would imply it resolves that range.
    ax = axes[1]
    time_axis_b = (np.arange(len(digital_envelope)) + 1) / 400.0
    ax.fill_between(time_axis_b, digital_envelope, color=LINE_COLOR, alpha=0.16, linewidth=0)
    ax.plot(time_axis_b, digital_envelope, color=LINE_COLOR, linewidth=0.8)
    ax.set_ylabel("RMS amplitude")
    ax.set_ylim(bottom=0.0)
    ax.yaxis.set_major_locator(ticker.MaxNLocator(3))
    ax.grid(axis="y", color="#DCE0E5", linewidth=0.5, alpha=0.9)
    ax.set_axisbelow(True)

    # (c) 5-channel fabricated 100 Hz AC-RMS envelope
    ax = axes[2]
    # The y axis is the five resonator channels the 100 Hz models consume, as
    # adjacent rows in frequency order, so the panel reads as one 5 x 100
    # feature map rather than bands floating on a kHz axis.
    order = np.argsort(RESONATOR_FREQUENCIES_HZ)
    fab_duration_plot_s = len(fabricated_envelope) * fab_frame_s
    im_c = ax.imshow(
        fabricated_envelope[:, order].T,
        origin="lower",
        aspect="auto",
        extent=(0.0, fab_duration_plot_s, 0.5, len(order) + 0.5),
        cmap=HARDWARE_CMAP,
        interpolation="nearest",
        rasterized=True,
    )
    ax.set_ylabel("Resonator channel")
    ax.set_ylim(0.5, len(order) + 0.5)
    ax.set_yticks(np.arange(1, len(order) + 1))
    ax.set_yticklabels([RESONATOR_LABELS[index] for index in order])
    add_colorbar(ax, im_c, "RMS (V)")

    # No panel titles: the caption names (a), (b) and (c), top to bottom.
    for ax in axes:
        ax.set_xlim(0.0, common_duration_s)
        ax.set_xticks(common_time_ticks)
        ax.set_xticklabels(["0", "0.25", "0.5", "0.75", "1"])
        ax.tick_params(length=2.0, width=0.5, pad=1.5)
        for spine in ax.spines.values():
            spine.set_linewidth(0.5)
            spine.set_color("#9AA1A8")
    axes[-1].set_xlabel("Time (s)", labelpad=1.5)
    fig.align_ylabels(axes)

    # No bbox_inches="tight": the margins above are set so the figure is
    # exactly the 3.39 in column width, and cropping would rescale it in LaTeX.
    name = f"icassp_frontend_comparison_digit{DIGIT}_speaker{SPEAKER}_rep{REPETITION}"
    final_dir = REPO_ROOT / "final"
    final_dir.mkdir(parents=True, exist_ok=True)
    for directory in (OUT_DIR, final_dir):
        png_path = directory / f"{name}.png"
        pdf_path = directory / f"{name}.pdf"
        fig.savefig(png_path, dpi=400, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
