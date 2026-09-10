"""Shared matplotlib style for every figure in this project: one validated
categorical palette (see the dataviz reference), a single sequential hue
for magnitude, no default matplotlib chrome, and deliberately no bar
charts anywhere -- every comparison here is a dot plot, a line, a small
multiple, or a direct visual, per this project's own standing convention.
"""
from __future__ import annotations

import matplotlib.pyplot as plt
import numpy as np

BLUE = "#2a78d6"
ORANGE = "#eb6834"
AQUA = "#1baf7a"
YELLOW = "#eda100"
MAGENTA = "#e87ba4"
GREEN = "#008300"
VIOLET = "#4a3aa7"
RED = "#e34948"
CATEGORICAL = [BLUE, ORANGE, AQUA, YELLOW, MAGENTA, GREEN, VIOLET, RED]

INK = "#1a1a19"
INK_SECONDARY = "#52514e"
INK_MUTED = "#8a8980"
SURFACE = "#fcfcfb"
GRID = "#e4e3dd"

SEQUENTIAL_BLUE = ["#eaf2fb", "#c3ddf5", "#8dbdea", "#548fd0", "#2a78d6", "#194c8a"]


def apply_style() -> None:
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.labelcolor": INK_SECONDARY,
            "text.color": INK,
            "xtick.color": INK_SECONDARY,
            "ytick.color": INK_SECONDARY,
            "axes.grid": False,
            "font.family": "sans-serif",
            "font.sans-serif": ["Segoe UI", "Arial", "DejaVu Sans"],
            "font.size": 11,
            "axes.titlesize": 13,
            "axes.titleweight": "medium",
            "axes.spines.top": False,
            "axes.spines.right": False,
        }
    )


def sequential_cmap():
    from matplotlib.colors import LinearSegmentedColormap

    return LinearSegmentedColormap.from_list("ballast_blue", SEQUENTIAL_BLUE)


def save(fig, path: str, dpi: int = 200) -> None:
    fig.savefig(path, dpi=dpi, bbox_inches="tight", facecolor=SURFACE)
    plt.close(fig)


def clean_axes(ax) -> None:
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.tick_params(length=0)
