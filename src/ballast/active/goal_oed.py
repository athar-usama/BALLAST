"""Active interrogation, stated as what it actually is: not a new active-
sensing algorithm, but a reduction. Because motion depends on the interior
only through the 10 numbers `theta`, choosing which motion to perform next
in order to learn about a specific interior QUERY is classical linear-
Gaussian optimal experiment design with a GOAL-ORIENTED objective --
minimize the posterior variance of the query, not D-optimality on theta
itself (which is the standard, parameter-only objective every prior
inertial-parameter OED method uses).

The one thing worth actually claiming here: since every regime's Fisher
information is a rank-limited contribution to a 10-dimensional parameter
space, a greedy goal-oriented policy should need only a HANDFUL of well-
chosen, DIVERSE regimes to closely approach the achievable variance floor,
while a large pool of RANDOM actions -- mostly redundant, low-rank
contributions in the same one or two directions -- needs many more
repetitions to do as well. That comparison is the actual deliverable of
this module, not the greedy algorithm itself (greedy submodular selection
for a linear-Gaussian objective is textbook).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


def posterior_query_variance(
    prior_cov_theta: np.ndarray, fisher_infos: list[np.ndarray], query_weight_theta: np.ndarray
) -> float:
    """Var(w^T theta) after incorporating a set of actions with the given
    Fisher information matrices, under a linear-Gaussian update:

        Sigma_posterior = (Sigma_prior^-1 + sum_k FIM_k)^-1
        Var(w^T theta)  = w^T Sigma_posterior w
    """
    precision = np.linalg.inv(prior_cov_theta) + sum(fisher_infos, start=np.zeros_like(prior_cov_theta))
    cov_posterior = np.linalg.inv(precision)
    return float(query_weight_theta @ cov_posterior @ query_weight_theta)


@dataclass(frozen=True)
class GreedySelection:
    order: list[str]  # candidate names, in the order chosen
    variance_after_each: list[float]  # posterior query variance after each successive pick


def greedy_select(
    prior_cov_theta: np.ndarray,
    candidates: dict[str, np.ndarray],
    query_weight_theta: np.ndarray,
    budget: int,
) -> GreedySelection:
    """Greedily pick, one at a time, whichever remaining candidate action
    most reduces the posterior variance of the query -- the goal-oriented
    analogue of greedy D-optimal design, using the SAME Fisher-information
    building blocks the observability atlas already computes."""
    remaining = dict(candidates)
    chosen_fims: list[np.ndarray] = []
    order: list[str] = []
    variances: list[float] = []

    for _ in range(min(budget, len(candidates))):
        best_name, best_var = None, np.inf
        for name, fim in remaining.items():
            trial_var = posterior_query_variance(prior_cov_theta, [*chosen_fims, fim], query_weight_theta)
            if trial_var < best_var:
                best_name, best_var = name, trial_var
        assert best_name is not None
        order.append(best_name)
        variances.append(best_var)
        chosen_fims.append(remaining.pop(best_name))

    return GreedySelection(order=order, variance_after_each=variances)


def random_select_variance(
    prior_cov_theta: np.ndarray,
    candidates: dict[str, np.ndarray],
    query_weight_theta: np.ndarray,
    n_actions: int,
    rng: np.random.Generator,
    n_trials: int = 50,
) -> float:
    """Average posterior query variance from picking `n_actions` candidates
    uniformly at random (averaged over `n_trials` draws, without
    replacement each time) -- the baseline the greedy policy is compared
    against."""
    names = list(candidates.keys())
    n_actions = min(n_actions, len(names))
    variances = []
    for _ in range(n_trials):
        picked = rng.choice(names, size=n_actions, replace=False)
        fims = [candidates[name] for name in picked]
        variances.append(posterior_query_variance(prior_cov_theta, fims, query_weight_theta))
    return float(np.mean(variances))


def saturation_curve(
    prior_cov_theta: np.ndarray,
    candidates: dict[str, np.ndarray],
    query_weight_theta: np.ndarray,
    budget: int,
) -> np.ndarray:
    """Posterior query variance after each of up to `budget` greedily
    chosen actions -- the ceiling this project's theory predicts: since
    theta has only 10 dimensions, variance reduction per ADDITIONAL action
    should fall off sharply once the diverse, informative directions are
    already covered by a handful of well-chosen regimes."""
    selection = greedy_select(prior_cov_theta, candidates, query_weight_theta, budget)
    return np.array(selection.variance_after_each)
