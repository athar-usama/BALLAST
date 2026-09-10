"""Figure: the active-interrogation saturation curve. A handful of
greedily chosen, diverse motions should approach the achievable posterior
variance far faster than random draws from a redundancy-heavy pool --
because theta has only 10 dimensions, so no policy can extract more than
10 informative directions no matter how many actions it spends. Uses the
exact candidate-pool construction the test suite already validates this
claim against (`tests/test_goal_oed.py`), not a synthetic toy case.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import BLUE, INK, INK_MUTED, ORANGE, apply_style, clean_axes, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.active.goal_oed import greedy_select, posterior_query_variance  # noqa: E402
from ballast.identify.observability import (  # noqa: E402
    fisher_information,
    numeric_jacobian,
    pack_theta,
    rocking_observable,
    tap_observable,
    tumble_observable,
)

M_TRUE = 0.5
C_TRUE = np.array([0.012, -0.018, 0.009])
I_TRUE = np.diag([0.0020, 0.0032, 0.0045])
THETA = pack_theta(M_TRUE, C_TRUE, I_TRUE)


def weak_prior_cov() -> np.ndarray:
    scale = np.abs(THETA) + 1e-6
    return np.diag((100.0 * scale) ** 2)


def build_candidate_pool(rng, n_rocking=40, n_tumble=3, n_tap=2):
    candidates: dict[str, np.ndarray] = {}
    for i in range(n_rocking):
        pivot = rng.uniform(-0.05, 0.05, size=3)
        pivot[2] -= 0.05

        def observable(theta, pivot=pivot):
            return rocking_observable(theta, pivot_point_body=pivot)

        candidates[f"rocking_{i}"] = fisher_information(numeric_jacobian(observable, THETA), np.eye(1) * 1e-4)

    for i in range(n_tumble):
        omega0 = rng.uniform(0.1, 4.0, size=3)

        def observable(theta, omega0=omega0):
            return tumble_observable(theta, omega0, t_eval_s=0.05)

        candidates[f"tumble_{i}"] = fisher_information(numeric_jacobian(observable, THETA), np.eye(3) * 1e-3)

    for i in range(n_tap):
        r_app = rng.uniform(-0.04, 0.04, size=3)
        impulse = rng.uniform(-0.5, 0.5, size=3)

        def observable(theta, r_app=r_app, impulse=impulse):
            return tap_observable(theta, r_app_body=r_app, impulse_body=impulse)

        candidates[f"tap_{i}"] = fisher_information(numeric_jacobian(observable, THETA), np.eye(6) * 1e-6)

    return candidates


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    rng = np.random.default_rng(2)
    candidates = build_candidate_pool(rng)
    prior_cov = weak_prior_cov()
    query = np.zeros(10)
    query[4:10] = 1.0  # sensitive to the full inertia-tensor structure

    budget = 12
    greedy_result = greedy_select(prior_cov, candidates, query, budget=budget)
    greedy_curve = greedy_result.variance_after_each

    names = list(candidates.keys())
    n_random_trials = 200
    random_curve = []
    for n_actions in range(1, budget + 1):
        vs = []
        for _ in range(n_random_trials):
            picked = rng.choice(names, size=n_actions, replace=False)
            fims = [candidates[name] for name in picked]
            vs.append(posterior_query_variance(prior_cov, fims, query))
        random_curve.append(np.mean(vs))

    prior_only_variance = posterior_query_variance(prior_cov, [], query)
    print("prior-only variance:", prior_only_variance)
    print("greedy curve:", greedy_curve)
    print("random curve:", random_curve)

    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(8.5, 5.0))
    actions = np.arange(1, budget + 1)
    ax.plot(actions, greedy_curve, color=BLUE, lw=2.4, marker="o", markersize=5, label="greedy, diverse regimes")
    ax.plot(actions, random_curve, color=ORANGE, lw=2.4, marker="o", markersize=5, label="random from the pool")
    ax.axvline(10, color=INK_MUTED, lw=1.2, linestyle=(0, (4, 3)))
    ax.text(10.25, 2e-8, "rank(A) = 10\nceiling", fontsize=9, color=INK_MUTED, va="center")
    ax.set_yscale("log")
    ax.set_xlabel("number of actions taken", color=INK_MUTED, fontsize=10.5)
    ax.set_ylabel("posterior variance of the query (log scale)", color=INK_MUTED, fontsize=10.5)
    ax.set_title(
        "A few chosen actions beat many random ones,\nand nothing beats ten",
        fontsize=13.5,
        color=INK,
    )
    ax.legend(loc="center right", frameon=False, fontsize=10, bbox_to_anchor=(1.0, 0.55))
    clean_axes(ax)

    save(fig, str(out_dir / "active_interrogation.png"))
    print("done")


if __name__ == "__main__":
    main()
