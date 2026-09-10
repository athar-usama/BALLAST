"""Figure: Bennu, a real asteroid with an independently published
heterogeneous interior (Scheeres et al. 2020), used to show that
principal-axis spin alone cannot distinguish it from an otherwise
identical uniform body. Left: the two interiors' density profile along
the spin axis. Right: their inertia ratios, which is all spin dynamics
can ever see, plotted against each other.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, INK, INK_MUTED, ORANGE, apply_style, clean_axes, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.data.bennu import (  # noqa: E402
    BENNU_SEMI_AXES_M,
    heterogeneous_bennu_model,
    spin_axis_alignment,
    uniform_bennu_model,
)


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    uniform = uniform_bennu_model()
    hetero = heterogeneous_bennu_model()
    spin_axis_world = np.array([0.0, 0.0, 1.0])
    align_uniform = spin_axis_alignment(uniform, spin_axis_world)
    align_hetero = spin_axis_alignment(hetero, spin_axis_world)

    print("uniform inertia diag:", np.diag(uniform.inertia_kgm2))
    print("hetero  inertia diag:", np.diag(hetero.inertia_kgm2))
    print("spin-axis alignment (uniform, hetero):", align_uniform, align_hetero)

    z = uniform.grid.centers()[:, 2]
    z_bins = np.linspace(-BENNU_SEMI_AXES_M[2], BENNU_SEMI_AXES_M[2], 24)
    bin_idx = np.digitize(z, z_bins)

    def profile(model):
        occ = model.rho > 0
        prof = np.full(len(z_bins) + 1, np.nan)
        for b in range(len(z_bins) + 1):
            sel = occ & (bin_idx == b)
            if sel.sum() > 5:
                prof[b] = model.rho[sel].mean()
        return prof

    prof_uniform = profile(uniform)
    prof_hetero = profile(hetero)
    z_centers_km = np.concatenate([[z_bins[0]], (z_bins[:-1] + z_bins[1:]) / 2, [z_bins[-1]]]) / 1000.0

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(11.0, 4.4))

    ax = axes[0]
    ax.plot(z_centers_km, prof_uniform, color=BLUE, lw=2.2, label="uniform interior")
    ax.plot(z_centers_km, prof_hetero, color=ORANGE, lw=2.2, label="heterogeneous (Scheeres et al.)")
    ax.set_xlabel("height along spin axis (km)", color=INK_MUTED, fontsize=10)
    ax.set_ylabel("mean density (kg/m$^3$)", color=INK_MUTED, fontsize=10)
    ax.set_title("Two different interiors,\nsame mass, shape, and spin axis", fontsize=12, color=INK)
    ax.legend(loc="lower center", frameon=False, fontsize=9)
    clean_axes(ax)

    ax2 = axes[1]
    labels = ["$I_{xx}/I_{zz}$", "$I_{yy}/I_{zz}$"]
    uniform_ratios = [uniform.inertia_kgm2[0, 0] / uniform.inertia_kgm2[2, 2],
                       uniform.inertia_kgm2[1, 1] / uniform.inertia_kgm2[2, 2]]
    hetero_ratios = [hetero.inertia_kgm2[0, 0] / hetero.inertia_kgm2[2, 2],
                      hetero.inertia_kgm2[1, 1] / hetero.inertia_kgm2[2, 2]]
    y_pos = np.arange(len(labels))
    ax2.scatter(uniform_ratios, y_pos - 0.08, s=200, color=BLUE, zorder=3, label="uniform")
    ax2.scatter(hetero_ratios, y_pos + 0.08, s=200, color=ORANGE, zorder=3, label="heterogeneous")
    for yi, (u, h) in zip(y_pos, zip(uniform_ratios, hetero_ratios, strict=True), strict=True):
        ax2.plot([u, h], [yi - 0.08, yi + 0.08], color=INK_MUTED, lw=1.0, zorder=1)
    ax2.set_yticks(y_pos)
    ax2.set_yticklabels(labels, fontsize=11)
    ax2.set_xlabel("inertia ratio", color=INK_MUTED, fontsize=10)
    ax2.set_title(
        f"Different ratios, but both fit the\nobserved spin axis equally well ({align_uniform:.4f} vs {align_hetero:.4f})",
        fontsize=12,
        color=INK,
    )
    ax2.legend(loc="lower right", frameon=False, fontsize=9)
    ax2.set_ylim(-0.6, 1.6)
    clean_axes(ax2)
    ax2.spines["left"].set_visible(False)
    ax2.tick_params(left=False)

    fig.suptitle(
        "Bennu: a real body known to be non-uniform, indistinguishable from a uniform twin by spin",
        fontsize=13,
        color=INK,
        y=1.04,
    )
    save(fig, str(out_dir / "bennu.png"))
    print("done")


if __name__ == "__main__":
    main()
