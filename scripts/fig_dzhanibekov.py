"""Figure: the intermediate-axis instability, straight from the rigid-body
integrator that the whole project stands on. A perturbed spin about each
of the three principal axes; only the intermediate one flips.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, GREEN, INK, INK_MUTED, ORANGE, apply_style, clean_axes, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.physics.rigid import RigidBodyState, integrate  # noqa: E402

I_BODY = np.diag([1.0, 2.0, 3.0])  # minor, intermediate, major


def tumble(omega0, n_steps=40000, dt=0.001):
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.array(omega0),
    )
    traj = integrate(state0, m=1.0, i_body=I_BODY, dt=dt, n_steps=n_steps)
    return traj.t_s, traj.omega_body_rad_s


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    cases = [
        ("spin about the MINOR axis", [5.0, 0.05, 0.02], "stable"),
        ("spin about the INTERMEDIATE axis", [0.05, 5.0, 0.02], "flips"),
        ("spin about the MAJOR axis", [0.05, 0.02, 5.0], "stable"),
    ]

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 3, figsize=(12.5, 3.6), sharey=True)
    for ax, (title, omega0, status) in zip(axes, cases, strict=True):
        t, w = tumble(omega0)
        ax.plot(t, w[:, 0], color=BLUE, lw=1.3, label="$\\omega_x$")
        ax.plot(t, w[:, 1], color=ORANGE, lw=1.3, label="$\\omega_y$")
        ax.plot(t, w[:, 2], color=GREEN, lw=1.3, label="$\\omega_z$")
        ax.set_title(title, fontsize=11.5, color=INK)
        ax.set_xlabel("time (s)", color=INK_MUTED, fontsize=10)
        badge_color = "#e34948" if status == "flips" else "#1baf7a"
        ax.text(
            0.97,
            0.04,
            status,
            transform=ax.transAxes,
            ha="right",
            va="bottom",
            fontsize=10,
            color=badge_color,
            fontweight="bold",
        )
        clean_axes(ax)
    axes[0].set_ylabel("angular velocity (rad/s)", color=INK_MUTED, fontsize=10)
    axes[0].legend(loc="upper left", frameon=False, fontsize=9)
    fig.suptitle(
        "The same rigid body, spun about each of its three principal axes",
        fontsize=13,
        color=INK,
        y=1.03,
    )
    save(fig, str(out_dir / "dzhanibekov.png"))
    print("done")


if __name__ == "__main__":
    main()
