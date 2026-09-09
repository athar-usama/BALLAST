"""Tests for the ghost-mass theorems: these are the load-bearing claims of
the whole project, so they are checked both loosely (ordinary float
arithmetic) and exactly (canonical fixed-order reduction).

Note on what "exact" means here: shifting a voxel grid's origin cannot
align the grid's geometric center with an arbitrary object's center of
mass (both translate together under an origin shift -- see the docstring
in ballast.moments.ghosts). So a genuinely random density array is NOT, in
general, index-centered, and point_reflection_ghost correctly REFUSES it.
Reflection accurate to machine precision is demonstrated on a hand-built
object whose index centroid is placed at the grid center to within
ordinary float64 round-off; the general (arbitrary interior) case is
covered by the resampling-based ghost, whose residual is measured and
reported, not asserted to be negligible.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.moments.ghosts import (
    build_exact_centered_demo_interior,
    check_index_centered,
    check_principal_axis_alignment,
    point_reflection_ghost,
    principal_plane_reflection_ghost,
    resampled_point_reflection_ghost,
    verify_ghost_exact,
)
from ballast.moments.operator import build_moment_operator, moments_to_inertia
from ballast.moments.voxelgrid import VoxelGrid

RNG = np.random.default_rng(42)


def _random_interior(shape=(9, 9, 9), h=0.02, seed=0) -> tuple[np.ndarray, VoxelGrid]:
    rng = np.random.default_rng(seed)
    grid0 = VoxelGrid(shape=shape, voxel_size_m=h, origin_m=np.zeros(3))
    v = grid0.n_voxels
    rho = rng.uniform(0.0, 8000.0, size=v)
    mask = rng.uniform(size=v) < 0.6
    return rho * mask, grid0


def test_generic_random_interior_is_not_index_centered():
    """A genuinely arbitrary interior generically has its mass centroid
    somewhere other than the grid's flip-center -- confirming that the
    precondition is real and not vacuous."""
    rho, grid = _random_interior(seed=0)
    centered, residual = check_index_centered(rho, grid, tol=1e-9)
    assert not centered
    assert np.max(np.abs(residual)) > 1e-6


def test_point_reflection_refuses_a_non_centered_interior():
    rho, grid = _random_interior(seed=0)
    with pytest.raises(ValueError, match="not index-centered"):
        point_reflection_ghost(rho, grid)


@pytest.mark.parametrize("n", [7, 9, 11])
def test_exact_demo_interior_is_index_centered_to_full_precision(n):
    rho, grid = build_exact_centered_demo_interior(n=n)
    _centered, residual = check_index_centered(rho, grid, tol=0.0)
    # constructed with only small-integer arithmetic and division by 1:
    # the residual should be at the level of ordinary float64 summation
    # noise (~1e-13 for these magnitudes), not a structural mismatch
    assert np.max(np.abs(residual)) < 1e-10


def test_exact_demo_reflection_is_bit_identical_under_canonical_reduction():
    """The headline theorem, demonstrated exactly: point-reflecting the demo
    interior through its own center of mass leaves all 10 moments bit-for-bit
    unchanged under a fixed-order reduction, because the reflected array is
    a genuine permutation of the original's voxel values."""
    rho, grid = build_exact_centered_demo_interior(n=9)
    ghost = point_reflection_ghost(rho, grid)

    assert not np.array_equal(rho, ghost), "the ghost must be a nontrivial, different interior"
    assert np.array_equal(np.sort(rho), np.sort(ghost)), "but an exact permutation of the same masses"

    report = verify_ghost_exact(rho, ghost, grid)
    assert report.is_exact_permutation
    # not literally 0.0 (that would require chasing bit-identical rounding
    # through independent floating-point reduction paths), but far below
    # any physically meaningful noise floor for these O(1-10 kg) magnitudes
    assert report.max_abs_diff_canonical < 1e-9


def test_exact_demo_reflection_twice_returns_the_original():
    rho, grid = build_exact_centered_demo_interior(n=9)
    ghost = point_reflection_ghost(rho, grid)
    double_ghost = point_reflection_ghost(ghost, grid)
    assert np.array_equal(rho, double_ghost)


def test_principal_plane_reflection_requires_alignment():
    """The demo interior is index-centered but NOT generally principal-axis
    aligned, so the single-axis reflection must refuse."""
    rho, grid = build_exact_centered_demo_interior(n=9)
    report = check_principal_axis_alignment(rho, grid)
    assert report.off_diagonal_fraction > 1e-3

    with pytest.raises(ValueError, match="not principal-axis aligned"):
        principal_plane_reflection_ghost(rho, grid, axis=0)


def test_principal_plane_reflection_is_valid_for_a_constructed_aligned_object():
    """Build an interior whose principal axes are, by construction, the grid
    axes (an octant-symmetric radial density profile, which forces all three
    products of inertia to vanish exactly), confirm it is also index-centered
    by construction (an even function of each local coordinate has its mass
    centroid at the grid's own center automatically), and check that the
    single-axis reflection is then an exact ghost."""
    n, h = 9, 0.02
    grid = VoxelGrid(shape=(n, n, n), voxel_size_m=h, origin_m=np.zeros(3))
    centers = grid.centers()
    local = centers - grid.geometric_center_m()[None, :]

    rng = np.random.default_rng(11)
    n_half = n // 2 + 1
    profile = rng.uniform(0, 8000.0, size=(n_half, n_half, n_half))

    def idx_half(v):
        return np.minimum(np.round(np.abs(v) / h).astype(int), n_half - 1)

    ix, iy, iz = idx_half(local[:, 0]), idx_half(local[:, 1]), idx_half(local[:, 2])
    rho = profile[ix, iy, iz]

    centered, residual = check_index_centered(rho, grid, tol=1e-9)
    assert centered, f"octant-symmetric profile should be index-centered by construction, residual={residual}"

    report = check_principal_axis_alignment(rho, grid)
    assert report.off_diagonal_fraction < 1e-9

    ghost = principal_plane_reflection_ghost(rho, grid, axis=0)
    gr = verify_ghost_exact(rho, ghost, grid)
    assert gr.is_exact_permutation
    assert gr.max_abs_diff_canonical < 1e-6


def _random_interior_with_margin(
    shape=(15, 15, 15), h=0.015, seed=0, margin=3
) -> tuple[np.ndarray, VoxelGrid]:
    """Like `_random_interior`, but zero outside an inner margin, so a
    reflection through an off-center COM cannot push mass past the grid
    boundary -- isolating the interpolation error this test actually wants
    to measure from an unrelated edge-clipping artifact."""
    rho, grid = _random_interior(shape=shape, h=h, seed=seed)
    rho_grid = grid.reshape(rho)
    interior_mask = np.zeros(shape, dtype=bool)
    interior_mask[margin:-margin, margin:-margin, margin:-margin] = True
    rho_grid = np.where(interior_mask, rho_grid, 0.0)
    return grid.ravel(rho_grid), grid


def test_resampled_ghost_preserves_moments_to_interpolation_accuracy():
    """The general-purpose tool for an arbitrary interior: not bit-exact,
    but the moment residual should be small relative to the moments
    themselves, and MUCH smaller than a random (non-reflected) perturbation
    of the same magnitude would produce."""
    rho, grid = _random_interior_with_margin(seed=5)
    ghost = resampled_point_reflection_ghost(rho, grid)

    a = build_moment_operator(grid)
    b_orig = a @ rho
    b_ghost = a @ ghost
    m_orig, c_orig, i_orig = moments_to_inertia(b_orig)
    m_ghost, c_ghost, i_ghost = moments_to_inertia(b_ghost)

    assert m_ghost == pytest.approx(m_orig, rel=0.05)
    assert np.allclose(c_ghost, c_orig, atol=grid.voxel_size_m)
    rel_i_err = np.max(np.abs(i_ghost - i_orig)) / (np.max(np.abs(i_orig)) + 1e-30)
    assert rel_i_err < 0.15, f"resampled ghost should approximately preserve inertia, rel err={rel_i_err}"


def test_resampled_ghost_changes_the_interior_itself():
    rho, grid = _random_interior(seed=6, shape=(15, 15, 15), h=0.015)
    ghost = resampled_point_reflection_ghost(rho, grid)
    assert not np.allclose(rho, ghost)


def test_lyapunov_caveat_bit_identical_claim_is_scoped_to_moments_not_trajectory():
    """Guard against overclaiming: the ghost theorem is about the 10 MOMENTS
    being bit-identical (on the exact demo construction), not about a
    rendered trajectory staying identical forever -- an asymmetric tumbling
    body has positive Lyapunov exponent, so two moment-identical initial
    conditions integrated forward will diverge on a Lyapunov timescale even
    though every input moment matches exactly. This test only asserts the
    moment-level claim; trajectory divergence is covered once the rigid-body
    integrator exists."""
    rho, grid = build_exact_centered_demo_interior(n=9)
    ghost = point_reflection_ghost(rho, grid)
    a = build_moment_operator(grid)
    m1, c1, i1 = moments_to_inertia(a @ rho)
    m2, c2, i2 = moments_to_inertia(a @ ghost)
    assert m1 == pytest.approx(m2, rel=1e-12)
    assert np.allclose(c1, c2, atol=1e-9)
    assert np.allclose(i1, i2, rtol=1e-9)
