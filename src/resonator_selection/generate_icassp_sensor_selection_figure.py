"""Generate the ICASSP figure documenting five-resonator subset selection."""

from __future__ import annotations

import csv
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
import numpy as np


ROOT = Path(__file__).resolve().parents[1]
SOURCE_DIR = ROOT / "outputs" / "five_sensor_subset_selection_100hz"
SOURCE_CSV = SOURCE_DIR / "best_result_by_five_sensor_subset.csv"
OUTPUT_DIR = ROOT / "final" / "icassp_sensor_selection"

SELECTED = "300;1800;3500;4500;7500"
BLUE = "#0072B2"
GRAY = "#858B91"
LIGHT_GRAY = "#D8DCE0"
DARK = "#20252A"


def read_results() -> list[dict[str, object]]:
    with SOURCE_CSV.open(newline="", encoding="utf-8-sig") as handle:
        rows = []
        for raw in csv.DictReader(handle):
            rows.append(
                {
                    "frequencies": raw["target_frequencies_hz"],
                    "accuracy": 100.0 * float(raw["balanced_accuracy"]),
                    "std": 100.0 * float(raw["fold_balanced_accuracy_std"]),
                    "minimum": 100.0 * float(raw["fold_balanced_accuracy_min"]),
                    "macro_f1": 100.0 * float(raw["macro_f1"]),
                }
            )
    rows.sort(
        key=lambda row: (
            float(row["accuracy"]),
            float(row["macro_f1"]),
            -float(row["std"]),
        ),
        reverse=True,
    )
    if len(rows) != 56:
        raise ValueError(f"Expected all 56 five-of-eight subsets, found {len(rows)}")
    if rows[0]["frequencies"] != SELECTED:
        raise ValueError(
            f"Expected selected bank {SELECTED}, found {rows[0]['frequencies']}"
        )
    return rows


def display_frequencies(value: object) -> str:
    return ", ".join(str(value).split(";"))


def draw_rank_curve(axis: plt.Axes, rows: list[dict[str, object]]) -> None:
    rank = np.arange(1, len(rows) + 1)
    accuracy = np.asarray([float(row["accuracy"]) for row in rows])

    axis.plot(rank, accuracy, color=GRAY, linewidth=1.15, zorder=1)
    axis.scatter(
        rank[1:],
        accuracy[1:],
        s=12,
        facecolor="white",
        edgecolor=GRAY,
        linewidth=0.7,
        zorder=2,
    )
    axis.scatter(
        rank[0],
        accuracy[0],
        s=72,
        marker="*",
        facecolor=BLUE,
        edgecolor=DARK,
        linewidth=0.6,
        zorder=4,
        label="Selected bank",
    )
    axis.annotate(
        "Selected",
        xy=(rank[0], accuracy[0]),
        xytext=(10.0, 10.0),
        textcoords="offset points",
        fontsize=6.7,
        color=DARK,
        ha="left",
        va="bottom",
        fontweight="bold",
    )
    axis.set_xlim(0, 57)
    axis.set_ylim(30, 72)
    axis.set_xticks([1, 10, 20, 30, 40, 50, 56])
    axis.set_xlabel("Rank among 56 five-resonator subsets")
    axis.set_ylabel("Balanced accuracy (%)")
    axis.grid(axis="y", color=LIGHT_GRAY, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.text(
        -0.17,
        1.04,
        "(a)",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=DARK,
    )
    axis.set_title("Exhaustive five-of-eight subset ranking", fontsize=8.1, pad=7)


def draw_top_subsets(axis: plt.Axes, rows: list[dict[str, object]]) -> None:
    top = rows[:8]
    y = np.arange(len(top))
    means = np.asarray([float(row["accuracy"]) for row in top])
    standard_deviation = np.asarray([float(row["std"]) for row in top])
    colors = [BLUE if row["frequencies"] == SELECTED else GRAY for row in top]
    sizes = [48 if row["frequencies"] == SELECTED else 27 for row in top]

    for index, (mean, spread, color, size) in enumerate(
        zip(means, standard_deviation, colors, sizes, strict=True)
    ):
        axis.errorbar(
            mean,
            index,
            xerr=spread,
            fmt="none",
            ecolor=color,
            elinewidth=1.15,
            capsize=2.2,
            capthick=0.9,
            alpha=0.92,
            zorder=2,
        )
        marker = "*" if index == 0 else "o"
        axis.scatter(
            mean,
            index,
            s=size,
            marker=marker,
            facecolor=color,
            edgecolor=DARK if index == 0 else "white",
            linewidth=0.55,
            zorder=3,
        )
        axis.text(
            mean + 1.0,
            index - 0.16,
            f"{mean:.1f}",
            fontsize=6.3,
            color=DARK,
            ha="left",
            va="center",
        )

    axis.set_yticks(y)
    axis.set_yticklabels([display_frequencies(row["frequencies"]) for row in top])
    axis.invert_yaxis()
    axis.set_xlim(30, 88)
    axis.set_xlabel("Mean balanced accuracy across seven held-out recordings (%)")
    axis.grid(axis="x", color=LIGHT_GRAY, linewidth=0.6)
    axis.set_axisbelow(True)
    axis.text(
        -0.25,
        1.04,
        "(b)",
        transform=axis.transAxes,
        fontsize=8.5,
        fontweight="bold",
        color=DARK,
    )
    axis.set_title("Top subsets; whiskers show 1 fold SD", fontsize=8.1, pad=7)


def write_readme(rows: list[dict[str, object]]) -> None:
    winner = rows[0]
    runner_up = rows[1]
    text = f"""# ICASSP five-resonator selection figure

## Suggested caption

**Selection of the five-resonator bank from eight COMSOL target responses.
All 56 five-of-eight subsets were evaluated using 182 labeled A-Z segments and
seven leave-one-recording-out folds at a 100 Hz trajectory rate. (a) Ranked
balanced accuracy for every subset. (b) The eight highest-mean subsets;
whiskers indicate +/-1 standard deviation across held-out recordings. The
selected 300, 1800, 3500, 4500, and 7500 Hz bank obtained {float(winner['accuracy']):.1f}%
mean balanced accuracy. Overlapping variability and the {float(runner_up['accuracy']):.1f}% runner-up
show that this is a classification-driven fabrication choice rather than proof
of a uniquely optimal frequency set. Frequencies denote COMSOL design targets.**

## Files

- `five_resonator_selection_icassp.pdf`: preferred vector paper asset
- `five_resonator_selection_icassp.svg`: editable vector asset
- `five_resonator_selection_icassp_600dpi.png`: high-resolution raster
- figure size: 7.16 x 3.25 inches, intended for full two-column width

## Source and protocol

- source table: `{SOURCE_CSV.relative_to(ROOT)}`
- candidate bank: 110, 180, 300, 1800, 3500, 4500, 6000, 7500 Hz
- combinations evaluated: 56
- examples: 182 A-Z segments from seven recordings
- validation: leave one complete recording out per fold
- selected bank: 300, 1800, 3500, 4500, 7500 Hz
- selected mean balanced accuracy: {float(winner['accuracy']):.4f}%
- selected fold standard deviation: {float(winner['std']):.4f} percentage points
- selected worst fold: {float(winner['minimum']):.4f}%

Regenerate from the repository root with:

```powershell
python .\\final\\generate_icassp_sensor_selection_figure.py
```
"""
    (OUTPUT_DIR / "README.md").write_text(text, encoding="ascii")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    rows = read_results()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.labelsize": 7.2,
            "xtick.labelsize": 6.5,
            "ytick.labelsize": 6.2,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure, axes = plt.subplots(
        1,
        2,
        figsize=(7.16, 3.25),
        gridspec_kw={"width_ratios": [0.90, 1.35], "wspace": 0.50},
    )
    draw_rank_curve(axes[0], rows)
    draw_top_subsets(axes[1], rows)
    figure.subplots_adjust(left=0.085, right=0.985, bottom=0.17, top=0.88)

    figure.savefig(
        OUTPUT_DIR / "five_resonator_selection_icassp.pdf",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "five_resonator_selection_icassp.svg",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "five_resonator_selection_icassp_600dpi.png",
        dpi=600,
        facecolor="white",
    )
    plt.close(figure)
    write_readme(rows)

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
