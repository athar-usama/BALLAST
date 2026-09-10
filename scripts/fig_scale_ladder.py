"""Figure: the scale ladder. The same 10-number bottleneck applies whether
the object is a hand tool on a table or a planet in an orbit fit --
Theorems 1 and 2 do not know or care about scale. A log-axis dot plot
positions real objects this project actually touches (Rattlebox slugs,
XDen-1K hand tools, Bennu, the terrestrial planets) on one mass axis.
Numbered markers (rather than leader-line text) because four planets sit
within two decades of each other -- fewer than two of twenty-eight decades
of axis width -- so no vertical stagger gives text room to breathe; a
compact side legend does instead.
"""
from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, GREEN, INK, INK_MUTED, ORANGE, SURFACE, apply_style, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.data.bennu import BENNU_MASS_KG  # noqa: E402
from ballast.data.bodies import EARTH, MARS, MERCURY, MOON  # noqa: E402

ENTRIES = [
    ("Rattlebox slug object", 0.35, BLUE, "synthetic, this project"),
    ("XDen-1K hand tool (median)", 0.18, ORANGE, "real, X-ray interior"),
    ("Bennu", BENNU_MASS_KG, GREEN, "real, spacecraft-tracked"),
    ("Moon", MOON.mass_kg, GREEN, "real, seismically constrained"),
    ("Mercury", MERCURY.mass_kg, GREEN, "real, spacecraft-tracked"),
    ("Mars", MARS.mass_kg, GREEN, "real, spacecraft-tracked"),
    ("Earth", EARTH.mass_kg, GREEN, "real, seismically constrained"),
]


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(11.0, 4.4))

    for i, (name, mass_kg, color, tag) in enumerate(ENTRIES, start=1):
        ax.scatter([mass_kg], [0], s=420, color=color, zorder=3, edgecolors=SURFACE, linewidths=1.6)
        ax.annotate(
            str(i),
            xy=(mass_kg, 0),
            ha="center",
            va="center",
            fontsize=9.5,
            color=SURFACE,
            fontweight="bold",
            zorder=4,
        )

    ax.set_xscale("log")
    ax.axhline(0, color=INK_MUTED, lw=1.2, zorder=2)
    ax.set_yticks([])
    ax.set_ylim(-1.0, 1.0)
    ax.set_xlim(1e-2, 1e26)
    ax.set_xlabel("mass (kg), log scale", color=INK_MUTED, fontsize=10)
    for spine in ["top", "right", "left"]:
        ax.spines[spine].set_visible(False)
    ax.spines["bottom"].set_color(INK_MUTED)
    ax.tick_params(length=0)

    ax.set_title(
        "The same ten numbers, twenty-seven orders of magnitude apart",
        fontsize=13.5,
        color=INK,
        pad=14,
    )

    legend_lines = [f"{i}.  {name}   {mass_kg:.2e} kg   ({tag})" for i, (name, mass_kg, _, tag) in enumerate(ENTRIES, start=1)]
    fig.text(
        0.08,
        -0.06,
        "\n".join(legend_lines),
        ha="left",
        va="top",
        fontsize=9.3,
        color=INK,
        family="monospace",
        linespacing=1.9,
    )

    save(fig, str(out_dir / "scale_ladder.png"))
    print("done")


if __name__ == "__main__":
    main()
