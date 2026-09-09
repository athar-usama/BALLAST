"""Tests for the active interrogation module: built on the SAME Fisher-
information machinery as the observability atlas, applied to real regime
candidates -- not synthetic toy matrices -- to check the plan's actual
target claim: a few well-chosen, diverse motions should beat many
redundant random ones.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.active.goal_oed import greedy_select, posterior_query_variance, random_select_variance
from ballast.identify.observability import (
    fisher_information,
    inertia_scale_direction,
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


def _weak_prior_cov() -> np.ndarray:
    """A weak prior, but scaled to each parameter's own natural magnitude
    (mass ~0.5, com ~0.01 m, inertia ~0.003 kg m^2) rather than one flat
    isotropic variance across all 10 -- mixing a huge FLAT variance (e.g.
    1e6 for every component) with Fisher information matrices that span
    many orders of magnitude across differently-scaled parameters produces
    a precision matrix with a condition number at or beyond double-
    precision's limit, and `np.linalg.inv` genuinely raises "Singular
    matrix" on it (found directly). A per-component scale avoids that."""
    scale = np.abs(THETA) + 1e-6
    return np.diag((100.0 * scale) ** 2)


def _build_candidate_pool(rng: np.random.Generator, n_rocking: int = 15, n_tumble: int = 3, n_tap: int = 2):
    """A realistic, redundancy-heavy pool: many rocking pivots (each gives
    only ~1 dimension of information, and largely the SAME dimension --
    the mass/inertia ratio), plus a few genuinely diverse tumble and tap
    instances (tumble spans inertia ratios; tap is full rank)."""
    candidates: dict[str, np.ndarray] = {}

    for i in range(n_rocking):
        pivot = rng.uniform(-0.05, 0.05, size=3)
        pivot[2] -= 0.05  # keep the pivot below the object, physically sensible

        def observable(theta, pivot=pivot):
            return rocking_observable(theta, pivot_point_body=pivot)

        jac = numeric_jacobian(observable, THETA)
        candidates[f"rocking_{i}"] = fisher_information(jac, obs_cov=np.eye(1) * 1e-4)

    for i in range(n_tumble):
        omega0 = rng.uniform(0.1, 4.0, size=3)

        def observable(theta, omega0=omega0):
            return tumble_observable(theta, omega0, t_eval_s=0.05)

        jac = numeric_jacobian(observable, THETA)
        candidates[f"tumble_{i}"] = fisher_information(jac, obs_cov=np.eye(3) * 1e-3)

    for i in range(n_tap):
        r_app = rng.uniform(-0.04, 0.04, size=3)
        impulse = rng.uniform(-0.5, 0.5, size=3)

        def observable(theta, r_app=r_app, impulse=impulse):
            return tap_observable(theta, r_app_body=r_app, impulse_body=impulse)

        jac = numeric_jacobian(observable, THETA)
        candidates[f"tap_{i}"] = fisher_information(jac, obs_cov=np.eye(6) * 1e-6)

    return candidates


def test_greedy_selection_reduces_variance_monotonically():
    rng = np.random.default_rng(0)
    candidates = _build_candidate_pool(rng)
    prior_cov = _weak_prior_cov()  # a weak prior, scaled to each parameter's own natural magnitude
    query = inertia_scale_direction(
        THETA
    )  # stand-in for a query sensitive to inertia's overall scale/structure

    selection = greedy_select(prior_cov, candidates, query, budget=5)
    variances = selection.variance_after_each
    from itertools import pairwise

    assert all(v2 <= v1 + 1e-9 for v1, v2 in pairwise(variances)), (
        "adding more information should never increase posterior variance"
    )


def test_greedy_prefers_diverse_regimes_over_redundant_rocking():
    """With a query that depends on more than just the m/I_pivot ratio
    rocking reveals, the greedy policy should reach for the diverse
    (tumble/tap) candidates rather than exhausting the redundant rocking
    pool first."""
    rng = np.random.default_rng(1)
    candidates = _build_candidate_pool(rng)
    prior_cov = _weak_prior_cov()
    query = np.zeros(10)
    query[4:10] = 1.0  # a query sensitive to the full inertia tensor structure, not just mass/com

    selection = greedy_select(prior_cov, candidates, query, budget=3)
    kinds = {name.split("_")[0] for name in selection.order}
    assert "tumble" in kinds or "tap" in kinds, (
        f"greedy should reach for diverse regimes for this query, picked {selection.order}"
    )


def test_a_few_greedy_choices_beat_many_random_ones():
    """The plan's actual target claim: a handful of well-chosen, diverse
    actions should achieve a LOWER (better) posterior query variance than
    a much larger number of actions picked at random from a
    redundancy-heavy pool."""
    rng = np.random.default_rng(2)
    # a much larger redundant pool this time: 15 random picks out of ~45
    # candidates is a genuine "needle in a haystack" draw, with real odds of
    # missing some of the few diverse (tumble/tap) candidates -- 15 out of
    # 20 (this module's default pool) would almost always include nearly
    # every informative candidate anyway, which is not the comparison the
    # project's own claim ("three chosen beats twenty random") is about.
    candidates = _build_candidate_pool(rng, n_rocking=40, n_tumble=3, n_tap=2)
    prior_cov = _weak_prior_cov()
    query = np.zeros(10)
    query[4:10] = 1.0

    greedy_budget = 3
    random_budget = 15  # a minority of the ~45-candidate pool

    greedy_result = greedy_select(prior_cov, candidates, query, budget=greedy_budget)
    greedy_variance = greedy_result.variance_after_each[-1]

    random_variance = random_select_variance(prior_cov, candidates, query, n_actions=random_budget, rng=rng)

    assert greedy_variance < random_variance, (
        f"{greedy_budget} greedily chosen actions should beat {random_budget} random ones: "
        f"greedy={greedy_variance:.3e}, random={random_variance:.3e}"
    )


def test_posterior_variance_matches_direct_computation():
    """A direct sanity check on the linear-Gaussian update formula itself,
    independent of the regime machinery."""
    prior_cov = np.diag([4.0] * 10)
    fim = np.eye(10) * 2.0
    query = np.zeros(10)
    query[0] = 1.0

    expected_posterior_var = 1.0 / (1.0 / 4.0 + 2.0)  # scalar case: 1/(1/prior_var + fisher_info)
    actual = posterior_query_variance(prior_cov, [fim], query)
    assert actual == pytest.approx(expected_posterior_var, rel=1e-9)
