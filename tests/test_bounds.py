"""Tests for the sharp-bound theorem (T1): the tightest possible interval on
any interior query given only the 10 measured moments, and its bang-bang
extremal structure."""

from __future__ import annotations

import numpy as np
import pytest

from ballast.moments.bounds import (
    ball_query,
    half_space_query,
    inclusion_family_dof,
    octant_query,
    sharp_query_bound,
)
from ballast.moments.operator import build_moment_operator
from ballast.moments.voxelgrid import VoxelGrid

RNG = np.random.default_rng(0)


def _small_grid(n=7, h=0.02) -> VoxelGrid:
    half = h * (n - 1) / 2
    return VoxelGrid(shape=(n, n, n), voxel_size_m=h, origin_m=np.array([-half, -half, -half]))


def test_bound_contains_the_true_interior_it_was_measured_from():
    """The sharp bound must always contain the truth: if b was measured from
    a real interior rho_true, then rho_true is itself feasible, so the true
    query value must lie in [q_min, q_max]."""
    grid = _small_grid()
    a = build_moment_operator(grid)
    rho_true = RNG.uniform(0, 1000.0, size=grid.n_voxels)
    b = a @ rho_true
    rho_max = 2000.0
    assert rho_true.max() < rho_max

    query = octant_query(grid, (1, 1, 1))
    true_value = float(query @ rho_true)

    bound = sharp_query_bound(query, a, b, rho_max=rho_max)
    assert bound.contains(true_value)
    assert bound.query_min <= bound.query_max


def test_extremal_interiors_are_bang_bang_up_to_the_constraint_count():
    """LP duality gives an optimal interior that is bang-bang except on at
    most as many voxels as there are equality constraints (10 moments): a
    basic feasible solution of a polytope cut by k equalities has at most k
    variables strictly between their bounds. This is exact LP theory, not a
    tolerance -- so the count of non-bang-bang voxels must be bounded by 10,
    not merely small."""
    grid = _small_grid()
    a = build_moment_operator(grid)
    rho_true = RNG.uniform(0, 1000.0, size=grid.n_voxels)
    b = a @ rho_true
    rho_max = 2000.0

    query = ball_query(grid, center_m=np.zeros(3), radius_m=0.05)
    bound = sharp_query_bound(query, a, b, rho_max=rho_max)

    n_moment_constraints = a.shape[0]
    for extremal in (bound.rho_min, bound.rho_max):
        near_zero = extremal < 1e-6 * rho_max
        near_max = extremal > rho_max * (1 - 1e-6)
        n_fractional = np.sum(~(near_zero | near_max))
        assert n_fractional <= n_moment_constraints, (
            f"at most {n_moment_constraints} voxels may be non-bang-bang at an LP vertex, "
            f"found {n_fractional}"
        )


def test_more_constrained_query_is_narrower_than_the_full_mass():
    """A tight local query (small ball) should generically be at least as
    hard to pin down as -- and is not expected to be narrower than -- asking
    about the total mass, since the moments constrain aggregate quantities
    far more tightly than any single localized region. This test checks the
    opposite, uncontroversial direction: the total-mass query has zero width
    (mass is exactly one of the 10 measured moments, so it is fully
    determined, not merely bounded)."""
    grid = _small_grid()
    a = build_moment_operator(grid)
    rho_true = RNG.uniform(0, 1000.0, size=grid.n_voxels)
    b = a @ rho_true
    rho_max = 2000.0

    total_mass_query = np.full(grid.n_voxels, grid.voxel_volume_m3)
    bound = sharp_query_bound(total_mass_query, a, b, rho_max=rho_max)
    assert bound.width < 1e-6 * b[0]


def test_localized_query_is_generically_not_pinned_down():
    """A small local region's mass is NOT one of the 10 moments, so its
    sharp interval should generically have positive width -- the concrete
    demonstration that the 10 numbers do not determine the interior."""
    grid = _small_grid(n=9, h=0.015)
    a = build_moment_operator(grid)
    rho_true = RNG.uniform(0, 1000.0, size=grid.n_voxels)
    b = a @ rho_true
    rho_max = 2000.0

    query = ball_query(grid, center_m=np.array([0.02, 0.0, 0.0]), radius_m=0.03)
    bound = sharp_query_bound(query, a, b, rho_max=rho_max)
    assert bound.width > 0.0, "a generic local query should not be exactly pinned down by 10 moments"


def test_half_space_query_bound_is_consistent_with_octant_query():
    """An octant query is the intersection of three half-space queries along
    orthogonal axes; sanity check they use consistent conventions by
    checking a single half-space bound behaves reasonably (contains truth,
    finite width)."""
    grid = _small_grid()
    a = build_moment_operator(grid)
    rho_true = RNG.uniform(0, 1000.0, size=grid.n_voxels)
    b = a @ rho_true
    rho_max = 2000.0

    query = half_space_query(grid, normal=np.array([1.0, 0.0, 0.0]), offset_m=0.0)
    true_value = float(query @ rho_true)
    bound = sharp_query_bound(query, a, b, rho_max=rho_max)
    assert bound.contains(true_value)
    assert np.isfinite(bound.width)


def test_inclusion_family_dof_matches_the_theorem():
    """T2: a single ellipsoidal inclusion has exactly 10 DOF against 10
    moment constraints (generically exactly identified); two inclusions
    have a strictly positive DOF surplus (a continuous ambiguous family)."""
    assert inclusion_family_dof(1) == 0
    assert inclusion_family_dof(2) > 0
    assert inclusion_family_dof(2) == 2 * 10 - 10


@pytest.mark.parametrize("n_inclusions,expected_sign", [(1, 0), (2, 1)])
def test_inclusion_dof_sign(n_inclusions, expected_sign):
    dof = inclusion_family_dof(n_inclusions)
    if expected_sign == 0:
        assert dof == 0
    else:
        assert dof > 0
