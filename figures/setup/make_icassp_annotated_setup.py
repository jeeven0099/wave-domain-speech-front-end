"""Create publication-ready annotations for the resonator experiment photo.

The source JPEG is never modified. A rotated/cropped derivative (audio source
on the left, sound travelling left to right) is built in-memory at figure-build
time. Labels and sound waves remain vector objects in the PDF exports; PNG
copies are rendered at 600 dpi for convenient previewing.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from pathlib import Path

import matplotlib.patheffects as path_effects
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Arc, Circle
from PIL import Image

HERE = Path(__file__).resolve().parent
SOURCE_JPEG = HERE / "top_view.jpeg"

BLUE = "#0072B2"
INK = "#111111"
WHITE = "#FFFFFF"

# Rotation that puts the audio source (phone) on the left and makes the
# phone's emitting edge vertical, so sound waves travel purely left to right.
# Calibrated against the source photo's actual phone orientation.
ROTATION_DEG = 73.02533560549571
EDGE_PAD = 350  # edge-extended padding so expand=True has no flat/white fill
TOP_TRIM_PX = 125  # extra rows trimmed off the top (empty tabletop/ceiling) to shorten the figure
BOTTOM_TRIM_PX = 25  # extra rows trimmed off the bottom (empty floor margin)


@dataclass(frozen=True)
class Callout:
    text: str
    xy: tuple[float, float]
    xytext: tuple[float, float]
    align: str = "left"


@dataclass(frozen=True)
class SoundRings:
    """Concentric arcs radiating from a point source, opening toward the
    direction of travel (classic "sound wave" glyph), rather than a line
    with an arrowhead."""

    source: tuple[float, float]
    radii: tuple[float, ...]
    theta1: float
    theta2: float
    center_radius: float


def build_rotated_top_view() -> tuple[Image.Image, tuple[float, float, float, float], tuple[int, int]]:
    """Return the rotated+cropped top-view image and the crop box used.

    The source photo is edge-padded before rotation so PIL's expand=True
    corners are filled with extended tabletop texture instead of a flat
    fill color, then rotated and cropped to the region that contains every
    piece of equipment (phone, resonator bank, sensor, DAQ, breadboard)
    with no synthetic padding inside the crop.
    """
    image = Image.open(SOURCE_JPEG).convert("RGB")
    width, height = image.size
    array = np.array(image)

    padded = np.pad(array, ((EDGE_PAD, EDGE_PAD), (EDGE_PAD, EDGE_PAD), (0, 0)), mode="edge")
    padded_image = Image.fromarray(padded)
    rotated = padded_image.rotate(
        ROTATION_DEG, expand=True, fillcolor=(140, 135, 125), resample=Image.BICUBIC
    )

    # Content box in the original (unpadded) photo's pixel coordinates that
    # bounds the phone, resonator bank / sensor cluster, DAQ, and breadboard.
    x0, y0, x1, y1 = 122, 20, 768, 805
    corners = [(x0, y0), (x1, y0), (x0, y1), (x1, y1)]
    rotated_corners = [_rotate_point(px, py, width, height, rotated.size) for px, py in corners]
    xs = [p[0] for p in rotated_corners]
    ys = [p[1] for p in rotated_corners]
    crop_box = (min(xs), min(ys), max(xs), max(ys))
    cropped = rotated.crop(tuple(int(round(v)) for v in crop_box))

    cropped = cropped.crop((0, TOP_TRIM_PX, cropped.width, cropped.height - BOTTOM_TRIM_PX))
    crop_box = (
        crop_box[0],
        crop_box[1] + TOP_TRIM_PX,
        crop_box[2],
        crop_box[3] - BOTTOM_TRIM_PX,
    )
    return cropped, crop_box, rotated.size


def _rotate_point(
    px: float, py: float, width: int, height: int, rotated_size: tuple[int, int]
) -> tuple[float, float]:
    padded_center = (width / 2 + EDGE_PAD, height / 2 + EDGE_PAD)
    rad = math.radians(ROTATION_DEG)
    dx = (px + EDGE_PAD) - padded_center[0]
    dy = (py + EDGE_PAD) - padded_center[1]
    xr = dx * math.cos(rad) + dy * math.sin(rad)
    yr = -dx * math.sin(rad) + dy * math.cos(rad)
    rcx, rcy = rotated_size[0] / 2, rotated_size[1] / 2
    return rcx + xr, rcy + yr


CALLOUTS: tuple[Callout, ...] = (
    Callout("Audio source", (0.2013, 0.3783), (0.06, 0.1049)),
    Callout("Resonator bank", (0.7112, 0.3116), (0.42, 0.1471)),
    Callout("Sensor\n(rear port)", (0.7076, 0.4985), (0.50, 0.7142)),
    Callout("DAQ", (0.7917, 0.7173), (0.68, 0.8822)),
    Callout("Breadboard", (0.8680, 0.3166), (0.95, 0.0630), "right"),
)

# Midpoint of the phone's emitting edge, in the original (unrotated) photo's
# pixel coordinates -- this is the point the concentric wave rings radiate
# from, resolved to crop-pixel data coordinates at draw time.
SOUND_SOURCE_RAW = (280.0, 310.0)
SOUND_RING_RADII = (35.0, 90.0, 150.0, 215.0, 285.0)
SOUND_RING_CENTER_RADIUS = 10.0
SOUND_RING_THETA1 = -48.0
SOUND_RING_THETA2 = 48.0


def configure_style() -> None:
    plt.rcParams.update(
        {
            "font.family": "Arial",
            "font.size": 9.0,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.facecolor": "white",
        }
    )


def draw_sound_rings(ax: plt.Axes, item: SoundRings, linewidth: float) -> None:
    """Draw concentric arcs opening toward the direction of travel, in data
    (pixel) coordinates so the rings are true circles regardless of the
    figure's aspect ratio -- the classic radiating "sound wave" glyph."""
    halo = [path_effects.Stroke(linewidth=linewidth + 1.3, foreground=WHITE), path_effects.Normal()]

    center = Circle(
        item.source,
        item.center_radius,
        transform=ax.transData,
        facecolor=WHITE,
        edgecolor=BLUE,
        linewidth=linewidth,
        zorder=5,
    )
    center.set_path_effects(halo)
    ax.add_patch(center)

    for radius in item.radii:
        arc = Arc(
            item.source,
            width=2 * radius,
            height=2 * radius,
            theta1=item.theta1,
            theta2=item.theta2,
            transform=ax.transData,
            edgecolor=BLUE,
            linewidth=linewidth * 1.8,
            zorder=4,
        )
        arc.set_path_effects(halo)
        ax.add_patch(arc)


def draw_view(
    ax: plt.Axes,
    image: Image.Image,
    crop_box: tuple[float, float, float, float],
    rotated_size: tuple[int, int],
    *,
    fontsize: float,
    linewidth: float,
) -> None:
    ax.imshow(image)
    ax.set_axis_off()

    width, height = Image.open(SOURCE_JPEG).size
    source_x, source_y = _rotate_point(*SOUND_SOURCE_RAW, width, height, rotated_size)
    cx0, cy0, _, _ = crop_box
    rings = SoundRings(
        source=(source_x - cx0, source_y - cy0),
        radii=SOUND_RING_RADII,
        theta1=SOUND_RING_THETA1,
        theta2=SOUND_RING_THETA2,
        center_radius=SOUND_RING_CENTER_RADIUS,
    )
    draw_sound_rings(ax, rings, linewidth)

    for item in CALLOUTS:
        annotation = ax.annotate(
            item.text,
            xy=item.xy,
            xytext=item.xytext,
            xycoords="axes fraction",
            textcoords="axes fraction",
            ha=item.align,
            va="center",
            fontsize=fontsize,
            fontweight="bold",
            color=INK,
            linespacing=0.95,
            bbox={
                "boxstyle": "round,pad=0.18",
                "facecolor": WHITE,
                "edgecolor": INK,
                "linewidth": 0.55,
                "alpha": 0.92,
            },
            arrowprops={
                "arrowstyle": "-|>",
                "color": INK,
                "linewidth": linewidth,
                "mutation_scale": 7.0,
                "shrinkA": 2,
                "shrinkB": 1,
            },
            zorder=6,
        )
        annotation.set_clip_on(False)


def save_figure(fig: plt.Figure, stem: str) -> None:
    fig.savefig(HERE / f"{stem}.pdf", bbox_inches="tight", pad_inches=0.01)
    fig.savefig(HERE / f"{stem}.png", dpi=600, bbox_inches="tight", pad_inches=0.01)
    plt.close(fig)


def make_figure(
    stem: str,
    figure_width: float,
    image: Image.Image,
    crop_box: tuple[float, float, float, float],
    rotated_size: tuple[int, int],
    *,
    fontsize: float,
    linewidth: float,
) -> None:
    width, height = image.size
    figure_height = figure_width * height / width
    fig, ax = plt.subplots(figsize=(figure_width, figure_height), constrained_layout=True)
    draw_view(ax, image, crop_box, rotated_size, fontsize=fontsize, linewidth=linewidth)
    save_figure(fig, stem)


def main() -> None:
    configure_style()
    image, crop_box, rotated_size = build_rotated_top_view()
    # ICASSP 2027 specifies an 86 mm (3.39 in) single-column width and
    # permits a 178 mm (7 in) full-width image area.
    make_figure("icassp_top_view_annotated", 3.39, image, crop_box, rotated_size, fontsize=8.0, linewidth=0.85)
    make_figure(
        "icassp_resonator_experiment_setup", 7.00, image, crop_box, rotated_size, fontsize=8.0, linewidth=1.0
    )
    print(f"Wrote annotated setup figures to {HERE}")


if __name__ == "__main__":
    main()
