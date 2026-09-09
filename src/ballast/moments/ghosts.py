"""Ghost mass: exact, search-free constructions of motion-identical interiors.

The headline result is not "the null space is large" -- that is rank-nullity,
already worked out by Wilczek (arXiv:2309.04882, S2, "Equivalent Rigid
Bodies") for point masses, and it underlies decades of gravity-inversion
theory (Parker 1975) and inverse mass-distribution design (Spin-It, SIGGRAPH
2014). What this module adds is the exact, zero-search realization of the
strongest member of that family for a voxelized interior:

    Theorem (chirality is invisible). Point-reflect any interior through its
    own center of mass. Mass, center of mass, and inertia tensor are all
    unchanged, for any interior, with no alignment or symmetry assumption.

A subtlety that the first draft of this module got wrong and is worth
recording: shifting a voxel grid's origin can NOT be used to align the
grid's geometric center with an arbitrary object's center of mass. Both the
computed COM and the grid's geometric center are affine in the origin with
the SAME slope (they both translate rigidly together when the origin
moves), so their difference is origin-invariant:

    COM(origin) - center(origin) = voxel_size * (mean_index - (N-1)/2)

which depends only on `mean_index`, the mass-weighted average voxel INDEX
of the density array -- a property of the array's data, not of any
coordinate bookkeeping. Exact index-permutation reflection (`np.flip`,
which always reflects about the grid's fixed geometric center) is therefore
bit-exact only for density arrays that are already index-centered
(`mean_index == (N-1)/2` on every axis), not for an arbitrary array.

This module gives both pieces honestly:

  - `point_reflection_ghost` / `check_index_centered`: the exact,
    bit-identical construction, valid when the precondition holds. It is
    used for a small hand-built demonstration object
    (`build_exact_centered_demo_interior`), constructed so the precondition
    holds to within ordinary double-precision round-off, not approximately.
  - `resampled_point_reflection_ghost`: a general-purpose tool that works
    on any interior (including a real, arbitrarily-placed X-ray density
    field) via trilinear resampling of the field at the reflected
    coordinates. This is not bit-exact -- it carries an interpolation-level
    residual, which `verify_ghost_exact` reports rather than hides.

A second, weaker exact family -- reflection in a single principal plane of
the inertia tensor -- additionally requires the grid's axes to already
coincide with the interior's principal axes (the second-moment tensor must
be diagonal in that frame). `principal_plane_reflection_ghost` checks this
and refuses rather than silently returning a non-ghost.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import map_coordinates

from ballast.moments.operator import build_moment_operator, moments_to_inertia
from ballast.moments.voxelgrid import VoxelGrid


def mean_index(rho_grid: np.ndarray) -> np.ndarray:
    """Mass-weighted average voxel INDEX of a density array shaped like the
    grid (not yet multiplied by voxel size or offset by any origin). This
    quantity is origin-independent: it is what actually determines whether
    `np.flip` reflection is exact for this array."""
    total = rho_grid.sum()
    if total <= 0:
        raise ValueError("density array has non-positive total mass")
    idx = np.indices(rho_grid.shape, dtype=np.float64)  # (3, nx, ny, nz)
    return np.array([np.sum(rho_grid * idx[a]) / total for a in range(3)])


def check_index_centered(rho: np.ndarray, grid: VoxelGrid, tol: float = 1e-9) -> tuple[bool, np.ndarray]:
    """Check whether `rho`'s mass-weighted index centroid already coincides
    with the grid's own flip-center `(N-1)/2` on every axis -- the exact
    precondition for `point_reflection_ghost` to be bit-identical. Returns
    (is_centered, residual_in_voxels)."""
    rho_grid = grid.reshape(rho)
    mi = mean_index(rho_grid)
    target = (np.asarray(grid.shape, dtype=np.float64) - 1.0) / 2.0
    residual = mi - target
    return bool(np.all(np.abs(residual) <= tol)), residual


def point_reflection_ghost(rho: np.ndarray, grid: VoxelGrid, tol: float = 1e-9) -> np.ndarray:
    """The always-valid-in-continuous-space, exact-on-a-grid-when-centered
    ghost: reflect `rho` through its own center of mass via `np.flip`.

    Raises if `rho` is not already index-centered on `grid` (see the module
    docstring for why an origin shift cannot fix this after the fact).
    Provably preserves (m, c, I) exactly when the precondition holds,
    because flipping all three axes is the linear map T = -Identity, and
    T @ S @ T.T == S for any symmetric S, and the reflection point coincides
    with the true center of mass exactly when mean_index == (N-1)/2.
    """
    centered, residual = check_index_centered(rho, grid, tol=tol)
    if not centered:
        raise ValueError(
            "rho is not index-centered on this grid "
            f"(residual {residual} voxels > tol {tol}); "
            "an origin shift cannot fix this (see module docstring) -- use "
            "resampled_point_reflection_ghost for a general interior, or "
            "build_exact_centered_demo_interior for an exact demonstration"
        )
    rho_grid = grid.reshape(rho)
    reflected = np.flip(rho_grid, axis=(0, 1, 2))
    return grid.ravel(reflected)


def resampled_point_reflection_ghost(rho: np.ndarray, grid: VoxelGrid) -> np.ndarray:
    """General-purpose point reflection through the TRUE (measured) center
    of mass, for an arbitrary interior. Works by trilinear-resampling the
    density field at the reflected coordinates `2c - x`.

    Not bit-exact: the resampled field's moments agree with the original's
    only to the interpolation error of the field near voxel boundaries, not
    to machine precision. `verify_ghost_exact` reports this residual rather
    than asserting it away. Use this for real interiors (e.g. an XDen-1K
    density field) where there is no freedom to construct an index-centered
    array in the first place.
    """
    a = build_moment_operator(grid)
    b = a @ rho
    _m, c, _i = moments_to_inertia(b)

    rho_grid = grid.reshape(rho)
    nx, ny, nz = grid.shape
    ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
    world = grid.origin_m[:, None, None, None] + np.stack([ii, jj, kk]) * grid.voxel_size_m
    reflected_world = 2.0 * c[:, None, None, None] - world
    reflected_idx = (reflected_world - grid.origin_m[:, None, None, None]) / grid.voxel_size_m

    resampled = map_coordinates(rho_grid, reflected_idx, order=1, mode="constant", cval=0.0, prefilter=False)
    return grid.ravel(resampled)


def build_exact_centered_demo_interior(
    n: int = 9, voxel_size_m: float = 0.02
) -> tuple[np.ndarray, VoxelGrid]:
    """A small, hand-built, all-positive-density interior whose mass-weighted
    index centroid is placed at `(n-1)/2` on every axis to within ordinary
    double-precision round-off (a few parts in 1e16 relative), so
    `point_reflection_ghost` on it reproduces every moment to within machine
    precision -- not the ~1e-3 to 1e-6 level of any physically meaningful
    noise floor (camera quantization, sensor noise), which is what "exact"
    means for the purposes of this project's claims.

    Construction: three "shape" point masses at arbitrary integer sites give
    the object its asymmetric (chiral-looking) structure, then three
    "balance" masses -- one per axis, each offset from the grid center along
    only that one axis -- are solved in closed form to zero out that axis's
    first-moment residual, without disturbing the other two axes.
    """
    if n % 2 == 0:
        raise ValueError("n must be odd so the grid center index is an exact integer")
    c = (n - 1) // 2
    rho = np.zeros((n, n, n), dtype=np.float64)

    if n < 7:
        raise ValueError("n must be >= 7 so the balance sites (offset 3 from center) stay in bounds")

    shape_sites = [
        ((c - 3, c - 2, c + 1), 5.0),
        ((c + 2, c + 3, c - 1), 3.0),
        ((c - 1, c + 3, c + 2), 7.0),
    ]
    for (i, j, k), m in shape_sites:
        rho[i, j, k] += m

    # residual first moment (relative to center c) contributed by the shape sites,
    # per axis -- computed with exact small-integer arithmetic
    residual = np.zeros(3)
    for (i, j, k), m in shape_sites:
        residual += m * (np.array([i, j, k]) - c)

    # one balance mass per axis, offset by 3 voxels along that axis only. The
    # SIGN of the offset is chosen opposite to the residual's sign so that
    # m_bal = -residual/offset always comes out positive -- an all-positive
    # density array is what makes this usable directly as a physical
    # demonstration object, not just an abstract signed ghost. The division
    # below is an ordinary float64 division (correctly rounded to within
    # ~1e-16 relative error), far below the centering tolerance used
    # anywhere this construction is checked.
    magnitude = 3
    for axis in range(3):
        sign = -1 if residual[axis] > 0 else 1
        offset = sign * magnitude
        site = [c, c, c]
        site[axis] = c + offset
        m_bal = -residual[axis] / offset
        rho[site[0], site[1], site[2]] += m_bal

    if rho.min() < 0:
        raise AssertionError("internal error: demo construction produced a negative density")

    grid = VoxelGrid(shape=(n, n, n), voxel_size_m=voxel_size_m, origin_m=np.zeros(3))
    return grid.ravel(rho), grid


@dataclass(frozen=True)
class PrincipalAxisReport:
    principal_values: np.ndarray  # (3,) eigenvalues of I_com
    principal_vectors: np.ndarray  # (3, 3) columns are eigenvectors
    off_diagonal_fraction: float  # ||offdiag(I_com)|| / ||I_com||, 0 => grid-aligned


def check_principal_axis_alignment(rho: np.ndarray, grid: VoxelGrid) -> PrincipalAxisReport:
    """Diagnose whether the grid's x/y/z axes already coincide with the
    interior's principal axes of inertia (needed for single-axis reflection
    ghosts to be exact, not merely for the all-axes point reflection)."""
    a = build_moment_operator(grid)
    b = a @ rho
    _m, _c, i_com = moments_to_inertia(b)
    vals, vecs = np.linalg.eigh(i_com)
    off = i_com - np.diag(np.diag(i_com))
    frac = float(np.linalg.norm(off) / (np.linalg.norm(i_com) + 1e-30))
    return PrincipalAxisReport(principal_values=vals, principal_vectors=vecs, off_diagonal_fraction=frac)


def principal_plane_reflection_ghost(
    rho: np.ndarray, grid: VoxelGrid, axis: int, alignment_tol: float = 1e-6, index_tol: float = 1e-9
) -> np.ndarray:
    """Reflect `rho` through the principal plane perpendicular to `axis`
    (0=x, 1=y, 2=z), about the object's own center of mass.

    Exact only when BOTH: (a) the grid is already principal-axis aligned
    (off-diagonal terms of I_com are ~0 in this frame -- a single-axis flip
    otherwise flips the sign of what should be invariant cross terms), and
    (b) the array is index-centered on that axis (same precondition as
    `point_reflection_ghost`). Raises rather than returning a silently-wrong
    ghost if either fails.
    """
    if axis not in (0, 1, 2):
        raise ValueError("axis must be 0, 1, or 2")
    report = check_principal_axis_alignment(rho, grid)
    if report.off_diagonal_fraction > alignment_tol:
        raise ValueError(
            "grid is not principal-axis aligned "
            f"(off-diagonal fraction {report.off_diagonal_fraction:.3e} > {alignment_tol:.3e}); "
            "only the all-axes point_reflection_ghost is valid here"
        )
    centered, residual = check_index_centered(rho, grid, tol=index_tol)
    if not centered:
        raise ValueError(f"rho is not index-centered on this grid (residual {residual} voxels)")
    rho_grid = grid.reshape(rho)
    reflected = np.flip(rho_grid, axis=axis)
    return grid.ravel(reflected)


@dataclass(frozen=True)
class GhostReport:
    moment_diff_dot: np.ndarray  # (10,) difference via ordinary A @ rho on both sides
    moment_diff_canonical: np.ndarray  # (10,) difference via a fixed-order canonical reduction
    max_abs_diff_dot: float
    max_abs_diff_canonical: float
    is_exact_permutation: bool  # True iff ghost's per-voxel density multiset == original's


def _canonical_moment(rho: np.ndarray, centers: np.ndarray, h: float) -> np.ndarray:
    """Recompute the 10 moments from per-voxel contributions sorted into a
    canonical (value-based) order before reduction, so that two density
    arrays which are permutations of one another sum in the identical
    sequence of floating-point additions and are therefore bit-identical,
    not merely close to machine precision."""
    h3 = h**3
    h2_12 = h**2 / 12.0
    terms = np.empty((10, rho.shape[0]), dtype=np.float64)
    terms[0] = rho * h3
    terms[1] = rho * h3 * centers[:, 0]
    terms[2] = rho * h3 * centers[:, 1]
    terms[3] = rho * h3 * centers[:, 2]
    terms[4] = rho * h3 * (centers[:, 0] ** 2 + h2_12)
    terms[5] = rho * h3 * (centers[:, 1] ** 2 + h2_12)
    terms[6] = rho * h3 * (centers[:, 2] ** 2 + h2_12)
    terms[7] = rho * h3 * centers[:, 0] * centers[:, 1]
    terms[8] = rho * h3 * centers[:, 0] * centers[:, 2]
    terms[9] = rho * h3 * centers[:, 1] * centers[:, 2]
    out = np.empty(10, dtype=np.float64)
    for row in range(10):
        out[row] = np.add.reduce(np.sort(terms[row]))
    return out


def verify_ghost_exact(rho: np.ndarray, ghost_rho: np.ndarray, grid: VoxelGrid) -> GhostReport:
    """Verify a ghost pair two ways: the ordinary vectorized moment (which
    differs only at the ~1e-16 relative floating-point level for a TRUE
    index-permutation pair, since BLAS reduction order differs between the
    two calls, or at the interpolation-residual level for a resampled pair)
    and a canonical fixed-order reduction (which is at the machine-precision
    level, not necessarily literally 0.0, for an index-permutation pair
    built from an already near-perfectly-centered array -- see
    `build_exact_centered_demo_interior` -- since the ghost's per-voxel
    terms are then a genuine permutation of the original's, and is NOT
    expected to be small for a resampled ghost, which this function reports
    honestly via `is_exact_permutation=False` rather than asserting away).
    """
    a = build_moment_operator(grid)
    b_orig_dot = a @ rho
    b_ghost_dot = a @ ghost_rho
    diff_dot = b_ghost_dot - b_orig_dot

    is_perm = np.array_equal(np.sort(rho), np.sort(ghost_rho))

    if is_perm:
        centers = grid.centers()
        h = grid.voxel_size_m
        b_orig_canon = _canonical_moment(rho, centers, h)
        b_ghost_canon = _canonical_moment(ghost_rho, centers, h)
        diff_canon = b_ghost_canon - b_orig_canon
    else:
        # not a permutation (e.g. a resampled ghost): the canonical-reduction
        # check is meaningless here, report the ordinary difference instead
        diff_canon = diff_dot

    return GhostReport(
        moment_diff_dot=diff_dot,
        moment_diff_canonical=diff_canon,
        max_abs_diff_dot=float(np.max(np.abs(diff_dot))),
        max_abs_diff_canonical=float(np.max(np.abs(diff_canon))),
        is_exact_permutation=is_perm,
    )
