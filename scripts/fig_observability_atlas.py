"""Figure: the observability atlas -- which motion regime informs which of
the 10 inertial parameters, straight from the Fisher information matrices
this project's identification modules compute. A dot matrix (never a bar
chart): dot area is the per-parameter Fisher information, on a shared log
scale within each row, with the "exactly null" cells shown as open rings
rather than omitted -- the null cells are the point of two of this
project's own theorems.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, INK, INK_MUTED, SURFACE, apply_style, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.identify.observability import (  # noqa: E402
    fisher_information,
    numeric_jacobian,
    pack_theta,
    rocking_observable,
    rolling_observable,
    tap_observable,
    toss_drag_observable,
    tumble_observable,
)

M_TRUE = 0.5
C_TRUE = np.array([0.012, -0.018, 0.009])
I_TRUE = np.diag([0.0020, 0.0032, 0.0045])
THETA = pack_theta(M_TRUE, C_TRUE, I_TRUE)

PARAM_LABELS = ["mass", "$c_x$", "$c_y$", "$c_z$", "$I_{xx}$", "$I_{yy}$", "$I_{zz}$", "$I_{xy}$", "$I_{xz}$", "$I_{yz}$"]


def regime_fim(name: str) -> np.ndarray:
    if name == "tumble":
        omega0 = np.array([0.3, 4.0, 0.1])

        def obs(theta):
            return np.concatenate([tumble_observable(theta, omega0, t_eval_s=t) for t in (0.02, 0.05, 0.09)])

        return fisher_information(numeric_jacobian(obs, THETA), np.eye(9))
    if name == "toss + drag":
        v0, omega0 = np.array([1.2, 0.0, 2.5]), np.array([0.3, 4.0, 0.1])

        def obs(theta):
            return toss_drag_observable(theta, v0, omega0, drag_coeff=0.03, t_flight_s=0.15)

        return fisher_information(numeric_jacobian(obs, THETA), np.eye(3) * 1e-6)
    if name == "rocking":
        pivot = np.array([0.0, 0.0, -0.05])

        def obs(theta):
            return rocking_observable(theta, pivot_point_body=pivot)

        return fisher_information(numeric_jacobian(obs, THETA), np.eye(1) * 1e-4)
    if name == "rolling":

        def obs(theta):
            return rolling_observable(theta, roll_axis=2, radius_m=0.03, incline_rad=np.radians(15))

        return fisher_information(numeric_jacobian(obs, THETA), np.eye(1) * 1e-4)
    if name == "known tap":
        r_app, impulse = np.array([0.03, 0.0, 0.02]), np.array([0.0, 0.4, 0.05])

        def obs(theta):
            return tap_observable(theta, r_app_body=r_app, impulse_body=impulse)

        return fisher_information(numeric_jacobian(obs, THETA), np.eye(6) * 1e-6)
    raise ValueError(name)


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    regimes = ["tumble", "toss + drag", "rocking", "rolling", "known tap"]
    grid = np.zeros((len(regimes), 10))
    for i, name in enumerate(regimes):
        fim = regime_fim(name)
        diag = np.abs(np.diag(fim))
        grid[i] = diag / (diag.max() + 1e-300)  # normalize per row for visual comparability

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.5, 4.3))
    for i in range(len(regimes)):
        for j in range(10):
            v = grid[i, j]
            null = v < 1e-6
            size = 40 if null else 60 + 900 * v
            if null:
                ax.scatter(j, i, s=140, facecolors="none", edgecolors=INK_MUTED, linewidths=1.2, zorder=3)
            else:
                ax.scatter(j, i, s=size, color=BLUE, alpha=0.35 + 0.65 * v, zorder=3, edgecolors="none")

    ax.set_xticks(range(10))
    ax.set_xticklabels(PARAM_LABELS, fontsize=11)
    ax.set_yticks(range(len(regimes)))
    ax.set_yticklabels(regimes, fontsize=11)
    ax.invert_yaxis()
    ax.set_xlim(-0.7, 9.7)
    ax.set_ylim(len(regimes) - 0.5, -0.7)
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.tick_params(length=0)
    ax.set_facecolor(SURFACE)

    ax.scatter([], [], s=400, color=BLUE, alpha=0.85, label="informative (dot area = information)")
    ax.scatter([], [], s=140, facecolors="none", edgecolors=INK_MUTED, linewidths=1.2, label="exactly null")
    ax.legend(loc="upper center", bbox_to_anchor=(0.5, -0.18), ncol=2, frameon=False, fontsize=10)

    ax.set_title("Which motion tells you which of the ten numbers", fontsize=13.5, color=INK, pad=14)
    save(fig, str(out_dir / "observability_atlas.png"))
    print("done")


if __name__ == "__main__":
    main()
