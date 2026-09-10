"""Figure: T1, the sharp-bound theorem, made concrete. Given only the 10
measured moments of some true (hidden) interior, the tightest possible
answer to a linear query -- here, mass on one side of a plane -- is a pair
of extremal interiors that are bang-bang (rho_max or 0) almost everywhere,
with a QUADRIC interface, exactly as LP duality predicts. This renders the
actual extremal interiors `sharp_query_bound` returns, not a cartoon.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, INK, INK_MUTED, ORANGE, SURFACE, apply_style, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.moments.bounds import half_space_query, sharp_query_bound  # noqa: E402
from ballast.moments.operator import build_moment_operator  # noqa: E402
from ballast.moments.voxelgrid import VoxelGrid  # noqa: E402

N = 17
RHO_MAX = 8000.0


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    grid = VoxelGrid(shape=(N, N, N), voxel_size_m=0.01, origin_m=np.array([-0.08, -0.08, -0.08]))
    centers = grid.centers()

    # a ground-truth interior nobody observing motion could ever see
    # directly: an off-center dense lump inside a lighter shell
    lump_center = np.array([0.02, -0.015, 0.01])
    dist = np.linalg.norm(centers - lump_center[None, :], axis=1)
    rho_true = np.where(dist < 0.035, RHO_MAX, 0.35 * RHO_MAX)
    r_outer = np.linalg.norm(grid.geometric_center_m() - centers, axis=1).reshape(grid.shape)
    outer_mask = (r_outer <= 0.075).ravel()
    rho_true = np.where(outer_mask, rho_true, 0.0)

    a = build_moment_operator(grid)
    b = a @ rho_true

    query = half_space_query(grid, normal=np.array([1.0, 0.0, 0.0]), offset_m=0.0)
    true_value = float(query @ rho_true)

    bound = sharp_query_bound(query, a, b, rho_max=RHO_MAX)
    # a genuinely uninformed bound: no motion at all, only "the interior is
    # somewhere inside the known outer shape with density between 0 and
    # rho_max" -- every voxel independently free, no moment constraint.
    no_motion_min, no_motion_max = 0.0, float(query.sum() * RHO_MAX)
    no_motion_width = no_motion_max - no_motion_min

    print(f"true value:            {true_value:.6f}")
    print(f"sharp bound (from 10 moments): [{bound.query_min:.6f}, {bound.query_max:.6f}]")
    print(f"no-motion bound (shape only):  [{no_motion_min:.6f}, {no_motion_max:.6f}]")
    print(f"width from 10 moments: {bound.width:.6f}  ({bound.width / no_motion_width:.1%} of no-motion width)")
    assert bound.contains(true_value)

    mid_slice = N // 2
    shape = grid.shape

    def sl(field):
        return field.reshape(shape)[:, :, mid_slice].T

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(11.5, 4.4))
    panels = [
        (sl(rho_true), "the true interior\n(never directly observed)", INK),
        (sl(bound.rho_min), "extremal interior:\nsmallest consistent query value", BLUE),
        (sl(bound.rho_max), "extremal interior:\nlargest consistent query value", ORANGE),
    ]
    for ax, (field, title, color) in zip(axes, panels, strict=True):
        ax.imshow(field, cmap="Greys", vmin=0, vmax=RHO_MAX, origin="lower")
        ax.contour(outer_mask.reshape(shape)[:, :, mid_slice].T.astype(float), levels=[0.5],
                   colors=[INK_MUTED], linewidths=1.0)
        ax.set_title(title, fontsize=10.5, color=color)
        ax.axis("off")

    fig.suptitle(
        "Same 10 moments, two provably extreme interiors -- both bang-bang, split by a flat interface",
        fontsize=12.5,
        color=INK,
        y=1.03,
    )

    save(fig, str(out_dir / "sharp_bound_quadric.png"))

    fig2, ax2 = plt.subplots(figsize=(8.5, 2.6))
    span = no_motion_width
    ax2.hlines(0.35, no_motion_min, no_motion_max, color=INK_MUTED, lw=6.0, alpha=0.25, zorder=1)
    ax2.hlines(-0.05, bound.query_min, bound.query_max, color=BLUE, lw=6.0, zorder=1)
    ax2.scatter([bound.query_min, bound.query_max], [-0.05, -0.05], s=70, color=BLUE, zorder=3)
    ax2.scatter([true_value], [-0.05], s=150, color=ORANGE, zorder=4, marker="*")
    ax2.annotate("no motion at all (shape only)", xy=(no_motion_max, 0.35), xytext=(-6, 0),
                 textcoords="offset points", ha="right", va="center", fontsize=9.5, color=INK_MUTED)
    ax2.annotate("from 10 measured moments", xy=(bound.query_max, -0.05), xytext=(8, 14),
                 textcoords="offset points", ha="left", va="center", fontsize=9.5, color=BLUE)
    ax2.annotate("true value", xy=(true_value, -0.05), xytext=(0, -26), textcoords="offset points",
                 ha="center", fontsize=9.5, color=ORANGE)
    ax2.set_xlim(-0.02 * span, 1.02 * span)
    ax2.set_ylim(-0.6, 0.6)
    ax2.axis("off")
    ax2.set_title("Certified interval on mass to one side of a plane, from 10 numbers alone", fontsize=11.5,
                  color=INK)
    save(fig2, str(out_dir / "sharp_bound_interval.png"))
    print("done")


if __name__ == "__main__":
    main()
