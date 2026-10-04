"""Plot cross-speaker AudioMNIST five-sensor class templates in a 2x2 grid."""

from __future__ import annotations

import json
from itertools import combinations
from pathlib import Path

import matplotlib
import numpy as np
import pandas as pd

matplotlib.use("Agg")
import matplotlib.pyplot as plt

REPO_ROOT = Path(__file__).resolve().parents[2]
CAPTURE_ROOT = (
    REPO_ROOT
    / "raspberry_pi_audiomnist_benchmark_bundle"
    / "data"
    / "fabricated"
    / "captures"
    / "test"
)
OUT_DIR = REPO_ROOT / "final"

FEATURE_RATE_HZ = 100.0
# 100 Hz envelopes with real-DAQ-rate window-center timestamps (see
# ablation_envelope_rate.py --stage export-pi); the acquisition CSV's time_s
# assumes exactly 10 ms spacing and drifts ~19 ms over a run.
ENVELOPE_NPZ = "envelopes_100Hz_ablation_capture.npz"
NPZ_CHANNEL_ORDER = ("R300", "R3500", "R7500", "R1800", "R4500")
BASELINE_CONTEXT_S = 0.050
TEMPLATE_FRAMES = 200
EXPECTED_REPETITIONS = 50

# Acquisition order rearranged into ascending resonator frequency.
CHANNEL_COLUMNS = (
    "AI0_R300_rms_v",
    "AI3_R1800_rms_v",
    "AI1_R3500_rms_v",
    "AI4_R4500_rms_v",
    "AI2_R7500_rms_v",
)
CHANNEL_LABELS = ("R300", "R1800", "R3500", "R4500", "R7500")

# The pair reported in the paper. select_comparison() below is what chose it;
# it still runs so the figure cannot drift away from the stated criterion
# without the disagreement being printed.
SPEAKERS = (18, 22)
DIGITS = (4, 8)


def interpolate_time(values: np.ndarray, target_frames: int) -> np.ndarray:
    source_x = np.linspace(0.0, 1.0, values.shape[1])
    target_x = np.linspace(0.0, 1.0, target_frames)
    return np.vstack([np.interp(target_x, source_x, channel) for channel in values])


def discover_captures() -> dict[int, tuple[str, ...]]:
    captures: dict[int, list[str]] = {}
    for metadata_path in sorted(CAPTURE_ROOT.glob("*/metadata.json")):
        metadata = json.loads(metadata_path.read_text(encoding="ascii"))
        speaker = int(metadata["dataset_speaker_id"])
        captures.setdefault(speaker, []).append(metadata_path.parent.name)
    return {speaker: tuple(names) for speaker, names in captures.items()}


def load_speaker_templates(
    speaker: int, capture_names: tuple[str, ...]
) -> dict[int, np.ndarray]:
    by_digit: dict[int, list[np.ndarray]] = {digit: [] for digit in range(10)}
    for capture_name in capture_names:
        capture = CAPTURE_ROOT / capture_name
        with np.load(capture / ENVELOPE_NPZ, allow_pickle=False) as archive:
            if float(archive["sample_rate_hz"]) != FEATURE_RATE_HZ:
                raise RuntimeError(f"{capture_name}: envelope rate is not {FEATURE_RATE_HZ} Hz")
            times = np.asarray(archive["time_s"], dtype=np.float64) + 0.005
            order = [NPZ_CHANNEL_ORDER.index(label) for label in CHANNEL_LABELS]
            envelope_values = np.asarray(archive["envelope_v"], dtype=np.float32)[:, order]
        events = pd.read_csv(capture / "aligned_events.csv")
        event_rows = events.loc[events["event_type"].eq("letter")]

        for event in event_rows.itertuples(index=False):
            digit = int(event.label)
            start_s = float(event.start_s)
            end_s = float(event.end_s)
            first = int(np.searchsorted(times, start_s, side="left"))
            last = int(np.searchsorted(times, end_s, side="left"))
            if last - first < 2:
                continue

            before_first = int(
                np.searchsorted(times, start_s - BASELINE_CONTEXT_S, side="left")
            )
            after_last = int(
                np.searchsorted(times, end_s + BASELINE_CONTEXT_S, side="left")
            )
            baseline_rows = np.concatenate(
                (envelope_values[before_first:first], envelope_values[last:after_last]),
                axis=0,
            )
            baseline = np.median(baseline_rows, axis=0) if len(baseline_rows) else 0.0
            signal = np.maximum(envelope_values[first:last] - baseline, 0.0).T

            # Remove recording-level loudness while retaining relative sensor activity.
            loudness = float(np.quantile(signal, 0.95))
            if loudness <= np.finfo(np.float32).eps:
                continue
            signal /= loudness
            by_digit[digit].append(interpolate_time(signal, TEMPLATE_FRAMES))

    templates: dict[int, np.ndarray] = {}
    for digit, repetitions in by_digit.items():
        if len(repetitions) != EXPECTED_REPETITIONS:
            raise RuntimeError(
                f"Speaker {speaker}, digit {digit}: expected {EXPECTED_REPETITIONS} "
                f"repetitions, found {len(repetitions)}"
            )
        templates[digit] = np.mean(np.stack(repetitions), axis=0)
    return templates


def correlation(first: np.ndarray, second: np.ndarray) -> float:
    first_flat = first.ravel() - float(np.mean(first))
    second_flat = second.ravel() - float(np.mean(second))
    denominator = float(np.linalg.norm(first_flat) * np.linalg.norm(second_flat))
    return float(np.dot(first_flat, second_flat) / denominator) if denominator else 0.0


def select_comparison(
    templates: dict[int, dict[int, np.ndarray]],
) -> tuple[tuple[int, int], tuple[int, int], float]:
    """Select the clearest speaker/digit combination using template correlations."""
    best_speakers = (0, 0)
    best_digits = (0, 1)
    best_margin = -np.inf
    found_high_similarity_pair = False
    for speakers in combinations(sorted(templates), 2):
        for digits in combinations(range(10), 2):
            first_speaker, second_speaker = speakers
            first_digit, second_digit = digits
            same_digit = min(
                correlation(templates[first_speaker][first_digit], templates[second_speaker][first_digit]),
                correlation(templates[first_speaker][second_digit], templates[second_speaker][second_digit]),
            )
            different_digit = max(
                correlation(templates[first_speaker][first_digit], templates[second_speaker][second_digit]),
                correlation(templates[first_speaker][second_digit], templates[second_speaker][first_digit]),
                correlation(templates[first_speaker][first_digit], templates[first_speaker][second_digit]),
                correlation(templates[second_speaker][first_digit], templates[second_speaker][second_digit]),
            )
            margin = same_digit - different_digit
            is_high_similarity_pair = same_digit >= 0.90
            if is_high_similarity_pair and (
                not found_high_similarity_pair or margin > best_margin
            ):
                best_speakers = speakers
                best_digits = digits
                best_margin = margin
                found_high_similarity_pair = True
            elif not found_high_similarity_pair and margin > best_margin:
                best_speakers = speakers
                best_digits = digits
                best_margin = margin
    return best_speakers, best_digits, float(best_margin)


def main() -> None:
    captures = discover_captures()
    templates = {
        speaker: load_speaker_templates(speaker, capture_names)
        for speaker, capture_names in captures.items()
    }
    speakers, digits = SPEAKERS, DIGITS
    chosen_speakers, chosen_digits, margin = select_comparison(templates)
    if (chosen_speakers, chosen_digits) != (speakers, digits):
        print(
            f"WARNING: pinned speakers={speakers} digits={digits} no longer match "
            f"the selection criterion, which now picks speakers={chosen_speakers} "
            f"digits={chosen_digits}"
        )

    selected = {
        (speaker, digit): templates[speaker][digit]
        for speaker in speakers
        for digit in digits
    }
    # Equalize fixed resonator gains for display, with one scale shared by all panels.
    channel_maxima = np.maximum.reduce(
        [np.max(template, axis=1) for template in selected.values()]
    ).reshape(5, 1)
    channel_maxima = np.maximum(channel_maxima, np.finfo(np.float32).eps)
    displayed = {
        key: np.clip(template / channel_maxima, 0.0, 1.0)
        for key, template in selected.items()
    }

    same_correlations = {
        digit: correlation(templates[speakers[0]][digit], templates[speakers[1]][digit])
        for digit in digits
    }
    # The same-digit figures only show that a digit survives a change of
    # speaker. What shows discrimination is how far below them every
    # different-digit comparison of these same four templates sits, so report
    # those too rather than leaving them implicit in the margin.
    first_digit, second_digit = digits
    different_correlations = {
        f"s{first_speaker}d{first_digit}_vs_s{second_speaker}d{second_digit}": correlation(
            templates[first_speaker][first_digit], templates[second_speaker][second_digit]
        )
        for first_speaker in speakers
        for second_speaker in speakers
    }
    print(
        f"selected speakers={speakers}; digits={digits}; "
        f"same-digit correlations={same_correlations}; "
        f"different-digit correlations={different_correlations}; "
        f"max different-digit={max(different_correlations.values()):.4f}; "
        f"selection margin={margin:.4f}"
    )

    plt.rcParams.update(
        {
            "font.family": "serif",
            "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
            "font.size": 7,
            "axes.titlesize": 7,
            "axes.labelsize": 6.5,
            "xtick.labelsize": 6,
            "ytick.labelsize": 5.5,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
        }
    )

    # Sized for a single ICASSP column (3.39 in) so it is placed at native scale.
    fig, axes = plt.subplots(2, 2, figsize=(3.39, 2.45), sharex=True, sharey=True)
    fig.subplots_adjust(left=0.205, right=0.855, bottom=0.175, top=0.90, wspace=0.10, hspace=0.20)
    panel_labels = (("(a)", "(b)"), ("(c)", "(d)"))
    image = None

    for row, speaker in enumerate(speakers):
        for column, digit in enumerate(digits):
            axis = axes[row, column]
            image = axis.imshow(
                displayed[(speaker, digit)],
                origin="lower",
                aspect="auto",
                extent=(0.0, 1.0, -0.5, 4.5),
                interpolation="bilinear",
                cmap="viridis",
                vmin=0.0,
                vmax=1.0,
            )
            axis.text(
                0.018,
                0.94,
                panel_labels[row][column],
                transform=axis.transAxes,
                ha="left",
                va="top",
                fontsize=6,
                fontweight="bold",
                color="white",
                bbox={"facecolor": "#222222", "edgecolor": "none", "pad": 1.0},
            )
            axis.set_yticks(range(5))
            if column == 0:
                axis.set_yticklabels(CHANNEL_LABELS)
            else:
                axis.tick_params(axis="y", labelleft=False)
            axis.tick_params(length=2.0, width=0.5, pad=1.5)
            for spine in axis.spines.values():
                spine.set_linewidth(0.6)
            if row == 0:
                axis.set_title(f"Digit {digit}", pad=3)
            if row == 1:
                # Three short labels rather than the default five: at one
                # column, "1.00" on the left panel and "0.00" on the right
                # collide across the gap between them. The axes are shared, so
                # this applies to both columns however many times it is set.
                axis.set_xticks([0.0, 0.5, 1.0])
                axis.set_xticklabels(["0", "0.5", "1"])
            if column == 0:
                axis.set_ylabel(f"Speaker {speaker}", labelpad=4)

    fig.supxlabel("Normalized utterance time", x=0.53, y=0.035, fontsize=6.5)

    colorbar_axis = fig.add_axes((0.875, 0.175, 0.026, 0.725))
    colorbar = fig.colorbar(image, cax=colorbar_axis)
    colorbar.set_label("Relative RMS", fontsize=6, labelpad=3)
    colorbar.set_ticks([0.0, 0.5, 1.0])
    colorbar.ax.tick_params(labelsize=5.5, length=2.0, pad=1.5)
    colorbar.outline.set_linewidth(0.6)

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    png_path = OUT_DIR / "icassp_audiomnist_2x2_five_sensor_comparison.png"
    pdf_path = OUT_DIR / "icassp_audiomnist_2x2_five_sensor_comparison.pdf"
    fig.savefig(png_path, dpi=600, facecolor="white")
    fig.savefig(pdf_path, facecolor="white")
    plt.close(fig)
    print(png_path.resolve())
    print(pdf_path.resolve())


if __name__ == "__main__":
    main()
