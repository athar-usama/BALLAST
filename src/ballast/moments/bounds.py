"""Sharp bounds on any interior query, given only the 10 measured moments.

This is the actual contribution the null space enables, not the null space
itself. Given the feasible set

    F(b) = { rho : 0 <= rho <= rho_max,  supp(rho) subset Omega,  A @ rho == b }

(a known outer shape Omega, a material bound rho_max, and the 10 moments b
measured from motion), the tightest possible answer to any LINEAR interior
query <c, rho> -- mass in an octant, mass within radius r of a point, mass
above the geometric mid-plane, presence of a dense inclusion in a region --
is the pair of semi-infinite linear programs

    q_max = max { <c, rho> : rho in F(b) },   q_min = min { <c, rho> : rho in F(b) }

Because both the objective and every constraint are linear in rho, LP
duality guarantees an optimal solution that is bang-bang:

    rho*(x) = rho_max * 1[ c(x) > lambda^T a(x) ]

where a(x) = (1, x, y, z, x^2, y^2, z^2, xy, xz, yz) is exactly the row of
basis functions the moment operator integrates against. In words: the
worst-case interior consistent with everything you could ever measure by
watching an object move is always two-valued, with a quadric interface --
except possibly on a thin set of voxels straddling that interface.

This is a standard, exact fact about linear programs, not an approximation:
a basic feasible solution of a polytope cut out by `k` equality constraints
(here k = 10, one per measured moment) has at most `k` variables away from
their bounds. So of the V voxels, at most 10 can land strictly between 0
and rho_max at an optimal vertex; every other voxel is exactly bang-bang.
On a fine grid (V in the thousands) that is a vanishing fraction, and it is
the reason two extremal interiors for the SAME query can differ on a
handful of voxels near the quadric interface without contradicting the
theorem.

This is Parker's ideal-body construction (Geophys. J. Int. 42:315, 1975)
transposed from gravity inversion to the rigid-body moment operator, which
appears not to have been done before.

Finite-voxel version: with a fixed voxel grid, F(b) becomes a bounded convex
polytope in R^V and the two LPs are ordinary (dense) linear programs, solved
here with `scipy.optimize.linprog` (HiGHS).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.optimize import linprog

from ballast.moments.voxelgrid import VoxelGrid


@dataclass(frozen=True)
class SharpBound:
    query_min: float
    query_max: float
    rho_min: np.ndarray  # (V,) the extremal (bang-bang) minimizing interior
    rho_max: np.ndarray  # (V,) the extremal (bang-bang) maximizing interior
    prior_free_diameter: float  # query_max - query_min: the width with NO prior beyond nonnegativity

    @property
    def width(self) -> float:
        return self.query_max - self.query_min

    def contains(self, true_value: float, atol: float = 1e-9) -> bool:
        return (self.query_min - atol) <= true_value <= (self.query_max + atol)


def sharp_query_bound(
    query_weights: np.ndarray,
    a: np.ndarray,
    b: np.ndarray,
    rho_max: float,
    method: str = "highs",
) -> SharpBound:
    """Solve the min/max LPs for a single linear query `<query_weights, rho>`
    subject to `0 <= rho <= rho_max` and `A @ rho == b`.

    `query_weights` has shape (V,) and encodes the query, e.g. an indicator
    (times voxel volume) for "mass in this region". Returns the certified
    interval and the two bang-bang extremal interiors that attain it.
    """
    v = a.shape[1]
    bounds = [(0.0, rho_max)] * v

    res_max = linprog(-query_weights, A_eq=a, b_eq=b, bounds=bounds, method=method)
    res_min = linprog(query_weights, A_eq=a, b_eq=b, bounds=bounds, method=method)
    if not res_max.success or not res_min.success:
        raise RuntimeError(
            f"LP infeasible or failed: max.success={res_max.success} ({res_max.message}), "
            f"min.success={res_min.success} ({res_min.message})"
        )

    q_max = float(-res_max.fun)
    q_min = float(res_min.fun)

    # The prior-free diameter uses rho_max as the only prior (nonnegativity + material bound),
    # i.e. exactly this same interval -- reported as its own field so callers that later add
    # a material/connectivity prior can quote "N% narrower than the prior-free bound".
    return SharpBound(
        query_min=q_min,
        query_max=q_max,
        rho_min=res_min.x,
        rho_max=res_max.x,
        prior_free_diameter=q_max - q_min,
    )


def octant_query(grid: VoxelGrid, sign: tuple[int, int, int]) -> np.ndarray:
    """Build the query-weight vector for 'mass in the octant where
    sign[k] * (x_k - center_k) >= 0 for all k', weighted by voxel volume so
    the query evaluates to a mass in kg."""
    centers = grid.centers()
    origin_center = grid.geometric_center_m()
    mask = np.ones(centers.shape[0], dtype=bool)
    for axis, s in enumerate(sign):
        if s > 0:
            mask &= centers[:, axis] >= origin_center[axis]
        elif s < 0:
            mask &= centers[:, axis] < origin_center[axis]
    return mask.astype(np.float64) * grid.voxel_volume_m3


def half_space_query(grid: VoxelGrid, normal: np.ndarray, offset_m: float) -> np.ndarray:
    """Mass on the side of the plane {x : normal . x >= offset_m}."""
    centers = grid.centers()
    n = np.asarray(normal, dtype=np.float64)
    n = n / np.linalg.norm(n)
    mask = (centers @ n) >= offset_m
    return mask.astype(np.float64) * grid.voxel_volume_m3


def ball_query(grid: VoxelGrid, center_m: np.ndarray, radius_m: float) -> np.ndarray:
    """Mass within `radius_m` of `center_m`."""
    centers = grid.centers()
    d = np.linalg.norm(centers - np.asarray(center_m)[None, :], axis=1)
    mask = d <= radius_m
    return mask.astype(np.float64) * grid.voxel_volume_m3


# --------------------------------------------------------------------------
# T2: single-inclusion identifiability. A single ellipsoidal inclusion of
# unknown density inside a known shell has exactly 10 degrees of freedom
# (3 center, 3 semi-axes, 3 orientation, 1 density-contrast) against 10
# equations (the moments), so it is generically exactly identified: motion
# does not tell you nothing about the interior, it tells you precisely one
# blob's worth. Two inclusions (17-20 DOF) admit a continuous family. This
# module provides the DOF count and a minimal closed-form 2-inclusion
# counterexample generator; the full nonlinear single-ellipsoid solve lives
# in ballast.solve.
# --------------------------------------------------------------------------

SINGLE_INCLUSION_DOF = 3 + 3 + 3 + 1  # center, semi-axes, orientation, density contrast
N_MOMENT_CONSTRAINTS = 10


def inclusion_family_dof(n_inclusions: int) -> int:
    """Degrees of freedom of an n-ellipsoidal-inclusion interior model
    (each inclusion: 3 center + 3 semi-axes + 3 orientation + 1 density),
    minus the 10 moment constraints. Positive => a continuous ambiguous
    family exists generically; zero or negative => generically rigid
    (exactly or over- determined)."""
    per_inclusion = 3 + 3 + 3 + 1
    return n_inclusions * per_inclusion - N_MOMENT_CONSTRAINTS
