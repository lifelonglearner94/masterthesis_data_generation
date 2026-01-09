#!/usr/bin/env python3
"""Generate a transparent PNG with a perfect black grid.

Requires: Pillow (PIL)
  pip install pillow

The generated image has an RGBA canvas (fully transparent background) and
black grid lines snapped to pixel boundaries for crisp edges.
"""

from __future__ import annotations

import argparse
from dataclasses import dataclass
from pathlib import Path

from PIL import Image, ImageDraw


# ---------------------------
# User-adjustable parameters
# ---------------------------

IMAGE_WIDTH_PX = 256
IMAGE_HEIGHT_PX = 256

# Distance between grid lines (in pixels).
GRID_SPACING_PX = 32

# Grid line thickness (in pixels). Use an integer >= 1.
LINE_WIDTH_PX = 4

# Optional margin from the image border before the first interior grid line.
# Set to 0 to have grid lines start at the outer border.
MARGIN_PX = 0

# RGBA colors
LINE_RGBA = (0, 0, 0, 255)  # pure black, fully opaque
BACKGROUND_RGBA = (255, 255, 255, 255)  # solid white background

# If True, draw border lines at x=0/x=width-1 and y=0/y=height-1.
DRAW_BORDER = True


@dataclass(frozen=True)
class GridSpec:
    width_px: int
    height_px: int
    spacing_px: int
    line_width_px: int
    margin_px: int
    line_rgba: tuple[int, int, int, int]
    background_rgba: tuple[int, int, int, int]
    draw_border: bool


def _validate(spec: GridSpec) -> None:
    if spec.width_px <= 0 or spec.height_px <= 0:
        raise ValueError("Image dimensions must be positive.")
    if spec.spacing_px <= 0:
        raise ValueError("GRID_SPACING_PX must be positive.")
    if spec.line_width_px <= 0:
        raise ValueError("LINE_WIDTH_PX must be >= 1.")
    if spec.margin_px < 0:
        raise ValueError("MARGIN_PX must be >= 0.")


def generate_grid_image(spec: GridSpec) -> Image.Image:
    """Create an RGBA image with transparent background and black grid lines."""
    _validate(spec)

    img = Image.new("RGBA", (spec.width_px, spec.height_px), spec.background_rgba)
    draw = ImageDraw.Draw(img)

    w, h = spec.width_px, spec.height_px

    # Draw border (snapped to pixel grid).
    if spec.draw_border:
        # Top
        draw.rectangle([0, 0, w - 1, spec.line_width_px - 1], fill=spec.line_rgba)
        # Bottom
        draw.rectangle([0, h - spec.line_width_px, w - 1, h - 1], fill=spec.line_rgba)
        # Left
        draw.rectangle([0, 0, spec.line_width_px - 1, h - 1], fill=spec.line_rgba)
        # Right
        draw.rectangle([w - spec.line_width_px, 0, w - 1, h - 1], fill=spec.line_rgba)

    # Interior vertical lines
    x = spec.margin_px
    while x < w:
        draw.rectangle(
            [x, 0, min(x + spec.line_width_px - 1, w - 1), h - 1],
            fill=spec.line_rgba,
        )
        x += spec.spacing_px

    # Interior horizontal lines
    y = spec.margin_px
    while y < h:
        draw.rectangle(
            [0, y, w - 1, min(y + spec.line_width_px - 1, h - 1)],
            fill=spec.line_rgba,
        )
        y += spec.spacing_px

    return img


def main() -> int:
    parser = argparse.ArgumentParser(description="Generate a transparent PNG with a black grid.")
    parser.add_argument(
        "--out",
        type=Path,
        default=Path("black_grid.png"),
        help="Output PNG path (default: black_grid.png)",
    )
    args = parser.parse_args()

    spec = GridSpec(
        width_px=IMAGE_WIDTH_PX,
        height_px=IMAGE_HEIGHT_PX,
        spacing_px=GRID_SPACING_PX,
        line_width_px=LINE_WIDTH_PX,
        margin_px=MARGIN_PX,
        line_rgba=LINE_RGBA,
        background_rgba=BACKGROUND_RGBA,
        draw_border=DRAW_BORDER,
    )

    img = generate_grid_image(spec)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    img.save(args.out, format="PNG")
    print(f"Wrote: {args.out.resolve()}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
