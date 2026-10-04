"""Generate the publication cross-section figure for the five final resonators."""

from __future__ import annotations

import csv
from dataclasses import dataclass
from pathlib import Path

import matplotlib as mpl
import matplotlib.pyplot as plt
from matplotlib.patches import Patch, Polygon, Rectangle


ROOT = Path(__file__).resolve().parent
OUTPUT_DIR = ROOT / "icassp_resonator_cross_sections"

BODY_FACE = "#D9DDE1"
BODY_EDGE = "#20262B"
AIR_FACE = "#39A6B8"
AIR_EDGE = "#075B68"
TEXT = "#16191C"


@dataclass(frozen=True)
class Resonator:
    panel: str
    channel: str
    target_hz: int
    revision: str
    spec_path: Path
    body_width_mm: float
    body_height_mm: float
    neck_diameter_mm: float
    neck_length_mm: float
    cavity_width_mm: float
    cavity_depth_mm: float
    taper_height_mm: float
    tap_width_mm: float
    tap_length_mm: float
    volume_ml: float
    boss_width_mm: float | None = None
    shoulder_height_mm: float | None = None
    drilled_tap: bool = False


def read_row(path: Path) -> dict[str, str]:
    with path.open(newline="", encoding="ascii") as handle:
        rows = list(csv.DictReader(handle))
    if len(rows) != 1:
        raise ValueError(f"Expected one specification row in {path}, found {len(rows)}")
    return rows[0]


def number(row: dict[str, str], *names: str) -> float:
    for name in names:
        value = row.get(name)
        if value not in (None, ""):
            return float(value)
    raise KeyError(f"None of {names!r} exists in specification row")


def load_resonators() -> list[Resonator]:
    definitions = [
        {
            "panel": "a",
            "channel": "AI0",
            "target_hz": 300,
            "revision": "V2P3",
            "folder": "single_helmholtz_300hz_v2p3_raised_sensor_boss",
            "spec": "300Hz_V2P3_raised_boss_spec.csv",
            "taper": True,
            "boss": True,
        },
        {
            "panel": "b",
            "channel": "AI3",
            "target_hz": 1800,
            "revision": "V7",
            "folder": "single_helmholtz_1800hz_v7_calibrated_from_2150",
            "spec": "AI3_1800Hz_V7_spec.csv",
            "taper": True,
        },
        {
            "panel": "c",
            "channel": "AI1",
            "target_hz": 3500,
            "revision": "V6",
            "folder": "single_helmholtz_3500hz_v6_reduced_bandwidth",
            "spec": "v6_reduced_bandwidth_spec.csv",
            "taper": False,
        },
        {
            "panel": "d",
            "channel": "AI4",
            "target_hz": 4500,
            "revision": "V1",
            "folder": "single_helmholtz_4500hz_v1_support_free_taper",
            "spec": "4500hz_v1_spec.csv",
            "taper": True,
        },
        {
            "panel": "e",
            "channel": "AI2",
            "target_hz": 7500,
            "revision": "V1P2",
            "folder": "single_helmholtz_7500hz_v1p2_calibrated_from_6800",
            "spec": "7500hz_v1p2_spec.csv",
            "taper": False,
            "drilled_tap": True,
        },
    ]

    resonators: list[Resonator] = []
    for definition in definitions:
        path = ROOT / str(definition["folder"]) / str(definition["spec"])
        row = read_row(path)
        taper_height = (
            number(row, "support_free_taper_height_mm")
            if definition["taper"]
            else 0.0
        )
        resonators.append(
            Resonator(
                panel=str(definition["panel"]),
                channel=str(definition["channel"]),
                target_hz=int(definition["target_hz"]),
                revision=str(definition["revision"]),
                spec_path=path,
                body_width_mm=number(row, "body_x_mm"),
                body_height_mm=number(
                    row, "total_body_height_mm", "body_height_mm"
                ),
                neck_diameter_mm=number(row, "neck_diameter_mm"),
                neck_length_mm=number(row, "physical_neck_length_mm"),
                cavity_width_mm=number(
                    row, "main_cavity_x_mm", "printed_cavity_x_mm"
                ),
                cavity_depth_mm=number(
                    row, "main_cavity_depth_mm", "printed_cavity_depth_mm"
                ),
                taper_height_mm=taper_height,
                tap_width_mm=number(
                    row, "microphone_port_mm", "microphone_tap_mm"
                ),
                tap_length_mm=number(
                    row,
                    "microphone_port_total_length_mm",
                    "microphone_tap_length_mm",
                ),
                volume_ml=number(row, "total_acoustic_volume_ml"),
                boss_width_mm=(
                    number(row, "raised_boss_side_mm")
                    if definition.get("boss")
                    else None
                ),
                shoulder_height_mm=(
                    number(row, "body_shoulder_height_mm")
                    if definition.get("boss")
                    else None
                ),
                drilled_tap=bool(definition.get("drilled_tap", False)),
            )
        )
    return resonators


def add_body(axis: plt.Axes, item: Resonator) -> None:
    if item.boss_width_mm is None or item.shoulder_height_mm is None:
        patch = Rectangle(
            (0.0, 0.0),
            item.body_width_mm,
            item.body_height_mm,
            facecolor=BODY_FACE,
            edgecolor=BODY_EDGE,
            linewidth=0.8,
            hatch="////",
            zorder=1,
        )
    else:
        center = item.body_width_mm / 2.0
        boss_left = center - item.boss_width_mm / 2.0
        boss_right = center + item.boss_width_mm / 2.0
        patch = Polygon(
            [
                (0.0, 0.0),
                (item.body_width_mm, 0.0),
                (item.body_width_mm, item.shoulder_height_mm),
                (boss_right, item.shoulder_height_mm),
                (boss_right, item.body_height_mm),
                (boss_left, item.body_height_mm),
                (boss_left, item.shoulder_height_mm),
                (0.0, item.shoulder_height_mm),
            ],
            closed=True,
            facecolor=BODY_FACE,
            edgecolor=BODY_EDGE,
            linewidth=0.8,
            hatch="////",
            zorder=1,
        )
    axis.add_patch(patch)


def add_air_path(axis: plt.Axes, item: Resonator) -> None:
    center = item.body_width_mm / 2.0
    cavity_left = center - item.cavity_width_mm / 2.0
    tap_left = center - item.tap_width_mm / 2.0
    cavity_bottom = item.neck_length_mm
    cavity_top = cavity_bottom + item.cavity_depth_mm
    tap_bottom = cavity_top + item.taper_height_mm

    common = {
        "facecolor": AIR_FACE,
        "edgecolor": AIR_EDGE,
        "linewidth": 0.75,
        "zorder": 3,
    }
    axis.add_patch(
        Rectangle(
            (center - item.neck_diameter_mm / 2.0, 0.0),
            item.neck_diameter_mm,
            item.neck_length_mm,
            **common,
        )
    )
    axis.add_patch(
        Rectangle(
            (cavity_left, cavity_bottom),
            item.cavity_width_mm,
            item.cavity_depth_mm,
            **common,
        )
    )
    if item.taper_height_mm > 0:
        axis.add_patch(
            Polygon(
                [
                    (cavity_left, cavity_top),
                    (cavity_left + item.cavity_width_mm, cavity_top),
                    (tap_left + item.tap_width_mm, tap_bottom),
                    (tap_left, tap_bottom),
                ],
                closed=True,
                **common,
            )
        )
    axis.add_patch(
        Rectangle(
            (tap_left, tap_bottom),
            item.tap_width_mm,
            min(item.tap_length_mm, item.body_height_mm - tap_bottom),
            **common,
        )
    )

    if item.drilled_tap:
        # V1P2 is shown in its operated state after the required 1 mm hand drill.
        axis.plot(
            [tap_left - 0.55, tap_left],
            [cavity_top, cavity_top],
            color=BODY_EDGE,
            linewidth=1.0,
            zorder=5,
        )
        axis.plot(
            [tap_left + item.tap_width_mm, tap_left + item.tap_width_mm + 0.55],
            [cavity_top, cavity_top],
            color=BODY_EDGE,
            linewidth=1.0,
            zorder=5,
        )


def add_scale_bar(axis: plt.Axes, item: Resonator) -> None:
    bar_mm = 5.0
    x0 = 0.06 * item.body_width_mm
    y = -0.095 * item.body_height_mm
    axis.plot(
        [x0, x0 + bar_mm],
        [y, y],
        color=TEXT,
        linewidth=1.5,
        solid_capstyle="butt",
        clip_on=False,
    )
    tick = 0.018 * item.body_height_mm
    axis.plot([x0, x0], [y - tick, y + tick], color=TEXT, linewidth=0.8, clip_on=False)
    axis.plot(
        [x0 + bar_mm, x0 + bar_mm],
        [y - tick, y + tick],
        color=TEXT,
        linewidth=0.8,
        clip_on=False,
    )
    axis.text(
        x0 + bar_mm / 2.0,
        y - 0.035 * item.body_height_mm,
        "5 mm",
        ha="center",
        va="top",
        fontsize=6.2,
        color=TEXT,
        clip_on=False,
    )


def draw_resonator(axis: plt.Axes, item: Resonator) -> None:
    add_body(axis, item)
    add_air_path(axis, item)
    add_scale_bar(axis, item)

    center = item.body_width_mm / 2.0
    arrow_length = max(0.06 * item.body_height_mm, 0.8)
    axis.annotate(
        "",
        xy=(center, 0.05 * item.body_height_mm),
        xytext=(center, -arrow_length),
        arrowprops={"arrowstyle": "->", "color": TEXT, "linewidth": 0.8},
        annotation_clip=False,
    )
    axis.annotate(
        "",
        xy=(center, item.body_height_mm - 0.04 * item.body_height_mm),
        xytext=(center, item.body_height_mm + arrow_length),
        arrowprops={"arrowstyle": "->", "color": TEXT, "linewidth": 0.8},
        annotation_clip=False,
    )

    axis.text(
        0.01,
        0.99,
        f"({item.panel})",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=8.0,
        fontweight="bold",
        color=TEXT,
    )
    axis.set_title(
        f"{item.channel} | {item.target_hz} Hz target\n"
        f"{item.revision}: d x L = {item.neck_diameter_mm:.2f} x "
        f"{item.neck_length_mm:.2f} mm; V = {item.volume_ml:.3f} mL",
        fontsize=7.2,
        color=TEXT,
        pad=5.0,
        linespacing=1.18,
    )
    axis.set_aspect("equal", adjustable="box")
    axis.set_xlim(-0.12 * item.body_width_mm, 1.12 * item.body_width_mm)
    axis.set_ylim(-0.17 * item.body_height_mm, 1.14 * item.body_height_mm)
    axis.set_xticks([])
    axis.set_yticks([])
    for spine in axis.spines.values():
        spine.set_visible(False)


def draw_legend(axis: plt.Axes) -> None:
    axis.axis("off")
    handles = [
        Patch(
            facecolor=BODY_FACE,
            edgecolor=BODY_EDGE,
            hatch="////",
            linewidth=0.8,
            label="Printed PLA",
        ),
        Patch(
            facecolor=AIR_FACE,
            edgecolor=AIR_EDGE,
            linewidth=0.8,
            label="Acoustic air path",
        ),
    ]
    axis.legend(
        handles=handles,
        loc="upper left",
        bbox_to_anchor=(0.05, 0.89),
        frameon=False,
        fontsize=7.2,
        handlelength=1.7,
        handleheight=1.0,
        borderpad=0.0,
        labelspacing=0.7,
    )
    axis.text(
        0.05,
        0.58,
        "Center-plane cutaways\n"
        "Bottom arrow: acoustic inlet\n"
        "Top arrow: sealed microphone port\n\n"
        "Panels are independently scaled.\n"
        "Frequency labels are design targets.",
        transform=axis.transAxes,
        ha="left",
        va="top",
        fontsize=7.0,
        color=TEXT,
        linespacing=1.35,
    )


def write_caption(resonators: list[Resonator]) -> None:
    source_lines = "\n".join(
        f"- ({item.panel}) {item.target_hz} Hz: `{item.spec_path.relative_to(ROOT)}`"
        for item in resonators
    )
    caption = f"""# ICASSP resonator cross-section figure

## Suggested caption

**Center-plane cross-sections of the five monolithic Helmholtz resonators used
in the fabricated speech front end:** (a) AI0, 300 Hz V2P3; (b) AI3, 1800 Hz
V7; (c) AI1, 3500 Hz V6; (d) AI4, 4500 Hz V1; and (e) AI2, 7500 Hz V1P2.
Gray hatched regions denote printed PLA and blue regions denote the connected
acoustic air path from the source inlet to the sealed microphone port. Panels
are independently scaled for legibility; each includes a 5 mm scale bar.
Frequencies are design targets rather than measured-response claims. The 7500
Hz section is shown in its operated state after opening the specified 1 mm
sacrificial bridge.**

## Export format

- `five_resonator_cross_sections_icassp_v2.pdf`: preferred vector paper asset
- `five_resonator_cross_sections_icassp_v2.svg`: editable vector asset
- `five_resonator_cross_sections_icassp_v2_600dpi.png`: high-resolution raster
- figure size: 7.16 x 4.55 inches, intended for full two-column width
- fonts embedded by Matplotlib PDF output; minimum designed label size 6.2 pt

## Geometry sources

{source_lines}

The figure is generated from the checked-in CSV specifications, not by tracing
the older preview PNG files. Regenerate it with:

```powershell
python .\\final\\generate_icassp_resonator_cross_sections.py
```
"""
    (OUTPUT_DIR / "README.md").write_text(caption, encoding="ascii")


def main() -> None:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    resonators = load_resonators()

    mpl.rcParams.update(
        {
            "font.family": "sans-serif",
            "font.sans-serif": ["Arial", "DejaVu Sans"],
            "font.size": 7.0,
            "axes.linewidth": 0.7,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "svg.fonttype": "none",
        }
    )

    figure, axes = plt.subplots(
        2,
        3,
        figsize=(7.16, 4.55),
        gridspec_kw={"wspace": 0.20, "hspace": 0.32},
    )
    for axis, item in zip(axes.flat[:5], resonators, strict=True):
        draw_resonator(axis, item)
    draw_legend(axes.flat[5])

    # Reserve enough headroom for the two-line panel titles in every backend.
    figure.subplots_adjust(left=0.025, right=0.99, bottom=0.055, top=0.88)
    figure.savefig(
        OUTPUT_DIR / "five_resonator_cross_sections_icassp_v2.pdf",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "five_resonator_cross_sections_icassp_v2.svg",
        facecolor="white",
    )
    figure.savefig(
        OUTPUT_DIR / "five_resonator_cross_sections_icassp_v2_600dpi.png",
        dpi=600,
        facecolor="white",
    )
    plt.close(figure)
    write_caption(resonators)

    for path in sorted(OUTPUT_DIR.iterdir()):
        print(path)


if __name__ == "__main__":
    main()
