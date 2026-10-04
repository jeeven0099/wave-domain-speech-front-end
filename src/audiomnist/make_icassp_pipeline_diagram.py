"""Generate the ICASSP hardware resonator pipeline diagram."""

from __future__ import annotations

from pathlib import Path

import matplotlib
import numpy as np

matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

REPO_ROOT = Path(__file__).resolve().parents[2]
# The paper pulls its figures from final/, so write there too rather than
# leaving a stale copy of this diagram next to the Pareto figure.
OUT_DIRS = (
    REPO_ROOT / "fabrication" / "audiomnist_daq_experiment" / "figures",
    REPO_ROOT / "final",
)
for _directory in OUT_DIRS:
    _directory.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.family": "serif",
        "font.serif": ["Times New Roman", "Times", "DejaVu Serif"],
        "font.size": 5.6,
        "figure.dpi": 300,
        "pdf.fonttype": 42,
        "ps.fonttype": 42,
    }
)

# One ICASSP column. Data units are inches (the axes fill the figure with an
# equal aspect), so every coordinate below is its size on the page.
FIG_WIDTH_IN = 3.39
FIG_HEIGHT_IN = 1.0

RESONATOR_FREQUENCIES_HZ = np.asarray([300, 1800, 3500, 4500, 7500], dtype=float)
RESONATOR_COLORS = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00"]

COLOR_INPUT = "#eeeeee"
COLOR_RESONATOR = "#cfe2f3"
COLOR_CLASSIFIER = "#d9ead3"
COLOR_OUTPUT = "#fff2cc"
EDGE_COLOR = "#333333"


def add_box(ax, x0, y0, width, height, text, facecolor, fontsize=5.6, weight="normal"):
    box = FancyBboxPatch(
        (x0, y0),
        width,
        height,
        boxstyle="round,pad=0.01,rounding_size=0.04",
        linewidth=0.6,
        edgecolor=EDGE_COLOR,
        facecolor=facecolor,
        zorder=3,
    )
    ax.add_patch(box)
    if text:
        ax.text(
            x0 + width / 2,
            y0 + height / 2,
            text,
            ha="center",
            va="center",
            fontsize=fontsize,
            weight=weight,
            linespacing=1.15,
            zorder=4,
        )
    return box


def add_arrow(ax, start, end):
    ax.add_patch(
        FancyArrowPatch(
            start,
            end,
            arrowstyle="-|>",
            mutation_scale=5,
            linewidth=0.6,
            color=EDGE_COLOR,
            zorder=2,
            shrinkA=0,
            shrinkB=0,
        )
    )


def add_waveform_icon(ax, center_x, center_y, width, height):
    phase = np.linspace(0, 4 * np.pi, 200)
    envelope = np.exp(-((phase - 2 * np.pi) ** 2) / (2 * (1.6 * np.pi) ** 2))
    waveform = envelope * np.sin(2.2 * phase)
    x = center_x - width / 2 + phase / (4 * np.pi) * width
    y = center_y + waveform * height / 2
    ax.plot(x, y, color="#555555", linewidth=0.6, zorder=4)


def add_resonator_bank_icon(ax, x0, y0, width, height, y_lo=0.16, y_hi=0.84):
    """Draw five color-coded resonator responses in one unlabeled block.

    y_lo/y_hi place the curves within the box as fractions of its height, so
    the same icon can sit above a caption inside a shared box.
    """
    icon_x0, icon_x1 = x0 + 0.08 * width, x0 + 0.92 * width
    icon_y0, icon_y1 = y0 + y_lo * height, y0 + y_hi * height
    normalized_frequency = np.linspace(0.0, 1.0, 500)
    quality = 0.035

    for frequency_hz, color in zip(RESONATOR_FREQUENCIES_HZ, RESONATOR_COLORS):
        peak = frequency_hz / 8_000.0
        response = np.exp(-((normalized_frequency - peak) ** 2) / (2 * quality**2))
        x = icon_x0 + normalized_frequency * (icon_x1 - icon_x0)
        y = icon_y0 + response * (icon_y1 - icon_y0)
        ax.plot(x, y, color=color, linewidth=0.8, zorder=4)

    ax.plot(
        [icon_x0, icon_x1],
        [icon_y0, icon_y0],
        color="#777777",
        linewidth=0.35,
        zorder=4,
    )


def add_group(ax, x0, y0, width, height, title):
    ax.add_patch(
        Rectangle(
            (x0, y0),
            width,
            height,
            linewidth=0.55,
            linestyle=(0, (3, 2)),
            edgecolor="#777777",
            facecolor="none",
            zorder=1,
        )
    )
    ax.text(
        x0 + width / 2,
        y0 + height + 0.07,
        title,
        ha="center",
        va="center",
        fontsize=5.8,
        style="italic",
    )


def main() -> None:
    fig = plt.figure(figsize=(FIG_WIDTH_IN, FIG_HEIGHT_IN))
    ax = fig.add_axes((0.0, 0.0, 1.0, 1.0))
    ax.set_xlim(0.0, FIG_WIDTH_IN)
    ax.set_ylim(0.0, FIG_HEIGHT_IN)
    ax.set_aspect("equal")
    ax.axis("off")

    center_y = 0.44

    input_x, input_y, input_w, input_h = 0.03, 0.22, 0.50, 0.44
    add_box(ax, input_x, input_y, input_w, input_h, "", COLOR_INPUT)
    add_waveform_icon(ax, input_x + input_w / 2, center_y + 0.06, input_w * 0.74, 0.17)
    ax.text(input_x + input_w / 2, input_y + 0.09, "Sound", ha="center", va="center")

    # The resonator bank and the feature stream it emits share one box: the
    # 5-channel 100 Hz stream is what the hardware front end produces, so it
    # belongs inside the front-end boundary rather than downstream of it.
    add_group(ax, 0.63, 0.08, 1.20, 0.72, "Hardware resonator front end")
    bank_x, bank_y, bank_w, bank_h = 0.71, 0.15, 1.04, 0.58
    add_box(ax, bank_x, bank_y, bank_w, bank_h, "", COLOR_RESONATOR)
    add_resonator_bank_icon(ax, bank_x, bank_y, bank_w, bank_h, y_lo=0.46, y_hi=0.92)
    ax.text(
        bank_x + bank_w / 2,
        bank_y + 0.22 * bank_h,
        "5-channel 100 Hz\nfeature stream",
        ha="center",
        va="center",
        fontsize=5.3,
        weight="bold",
        linespacing=1.1,
        zorder=4,
    )
    add_arrow(ax, (input_x + input_w, center_y), (bank_x, center_y))

    add_group(ax, 1.93, 0.08, 1.43, 0.72, "On-device classifier")
    classifier_x, classifier_y, classifier_w, classifier_h = 2.01, 0.19, 0.74, 0.50
    add_box(
        ax,
        classifier_x,
        classifier_y,
        classifier_w,
        classifier_h,
        # "causal-convolution" rather than "causal": every conv in the
        # classifier is causal, but the endpoint pools over the whole
        # utterance, so the system is not streaming.
        "Lightweight\ncausal-conv.\nclassifier",
        COLOR_CLASSIFIER,
    )
    add_arrow(ax, (bank_x + bank_w, center_y), (classifier_x, center_y))

    output_x, output_y, output_w, output_h = 2.85, 0.25, 0.46, 0.38
    add_box(
        ax,
        output_x,
        output_y,
        output_w,
        output_h,
        "Predicted\nclass",
        COLOR_OUTPUT,
        weight="bold",
    )
    add_arrow(ax, (classifier_x + classifier_w, center_y), (output_x, center_y))

    # No bbox_inches="tight": the figure is authored at the exact column width,
    # and cropping would make LaTeX rescale it.
    for directory in OUT_DIRS:
        png_path = directory / "icassp_resonator_pipeline_diagram.png"
        pdf_path = directory / "icassp_resonator_pipeline_diagram.pdf"
        fig.savefig(png_path, dpi=600, facecolor="white")
        fig.savefig(pdf_path, facecolor="white")
        print(f"saved {png_path}")
        print(f"saved {pdf_path}")


if __name__ == "__main__":
    main()
