"""Correctness tests for the moment operator: the entire project rests on
these being exact, not approximate."""

from __future__ import annotations

import numpy as np
import pytest

from ballast.moments.operator import (
    build_moment_operator,
    gram,
    inertia_to_moments,
    moments_to_inertia,
    nullspace_projector,
    project_to_constraint,
)
from ballast.moments.voxelgrid import VoxelGrid

RNG = np.random.default_rng(0)


def _cube_grid(n: int = 9, h: float = 0.01, origin=(0.0, 0.0, 0.0)) -> VoxelGrid:
    return VoxelGrid(shape=(n, n, n), voxel_size_m=h, origin_m=np.array(origin, dtype=np.float64))


def test_uniform_sphere_matches_analytic():
    """A uniform-density sphere's inertia tensor is (2/5) m r^2 * I3. Voxelize
    a sphere on a fine-ish grid and check convergence to the analytic value."""
    n, h = 41, 0.01
    grid = _cube_grid(n=n, h=h, origin=(-h * (n - 1) / 2, -h * (n - 1) / 2, -h * (n - 1) / 2))
    centers = grid.centers()
    r = 0.15
    density = 1000.0
    rho = np.where(np.linalg.norm(centers, axis=1) <= r, density, 0.0)

    a = build_moment_operator(grid)
    b = a @ rho
    m, c, i_com = moments_to_inertia(b)

    true_m = density * (4.0 / 3.0) * np.pi * r**3
    true_i = (2.0 / 5.0) * true_m * r**2

    assert m == pytest.approx(true_m, rel=0.03)
    assert np.allclose(c, 0.0, atol=1e-3)
    assert np.allclose(np.diag(i_com), true_i, rtol=0.05)
    # off-diagonal should vanish by symmetry, up to voxel-staircase discretization
    # noise on the sphere's boundary (a real indexing/permutation bug would produce
    # an off-diagonal comparable to the diagonal itself, not a fraction of a percent)
    off = i_com - np.diag(np.diag(i_com))
    assert np.max(np.abs(off)) < 1e-2 * true_i


def test_uniform_cylinder_matches_analytic():
    """Solid cylinder about its own axis: I_axis = (1/2) m r^2."""
    n, h = 41, 0.008
    half = h * (n - 1) / 2
    grid = _cube_grid(n=n, h=h, origin=(-half, -half, -half))
    centers = grid.centers()
    r, half_len = 0.12, 0.14
    density = 500.0
    radial = np.linalg.norm(centers[:, :2], axis=1)
    inside = (radial <= r) & (np.abs(centers[:, 2]) <= half_len)
    rho = np.where(inside, density, 0.0)

    a = build_moment_operator(grid)
    b = a @ rho
    m, _c, i_com = moments_to_inertia(b)

    true_m = density * np.pi * r**2 * (2 * half_len)
    true_i_axis = 0.5 * true_m * r**2

    assert m == pytest.approx(true_m, rel=0.03)
    assert i_com[2, 2] == pytest.approx(true_i_axis, rel=0.05)


def test_thin_shell_matches_analytic():
    """Thin spherical shell: I = (2/3) m r^2 (all three axes, by symmetry)."""
    n, h = 51, 0.006
    half = h * (n - 1) / 2
    grid = _cube_grid(n=n, h=h, origin=(-half, -half, -half))
    centers = grid.centers()
    r_out, r_in = 0.14, 0.12
    density = 2000.0
    radial = np.linalg.norm(centers, axis=1)
    inside = (radial <= r_out) & (radial >= r_in)
    rho = np.where(inside, density, 0.0)

    a = build_moment_operator(grid)
    b = a @ rho
    m, _c, i_com = moments_to_inertia(b)

    vol = (4.0 / 3.0) * np.pi * (r_out**3 - r_in**3)
    true_m = density * vol
    r_mean = 0.5 * (r_out + r_in)
    true_i = (2.0 / 3.0) * true_m * r_mean**2

    assert m == pytest.approx(true_m, rel=0.05)
    assert np.allclose(np.diag(i_com), true_i, rtol=0.1)


def test_off_center_point_mass_parallel_axis():
    """A single dense voxel offset from the grid origin must obey the
    parallel-axis theorem exactly (up to the voxel's own h^2/12 self-inertia)."""
    n, h = 5, 0.02
    grid = _cube_grid(n=n, h=h, origin=(0.0, 0.0, 0.0))
    v = grid.n_voxels
    rho = np.zeros(v)
    idx = np.ravel_multi_index((3, 1, 2), (n, n, n))
    density = 7850.0  # steel
    rho[idx] = density

    a = build_moment_operator(grid)
    b = a @ rho
    m, c, i_com = moments_to_inertia(b)

    voxel_mass = density * h**3
    assert m == pytest.approx(voxel_mass)
    expected_pos = grid.origin_m + np.array([3, 1, 2]) * h
    assert np.allclose(c, expected_pos)

    # inertia of the single voxel about its own center of mass is that of a
    # uniform cube of side h: (1/6) * m * h^2 on the diagonal, zero off-diagonal.
    self_i = (1.0 / 6.0) * voxel_mass * h**2
    assert np.allclose(np.diag(i_com), self_i, rtol=1e-8)
    off = i_com - np.diag(np.diag(i_com))
    assert np.max(np.abs(off)) < 1e-12


def test_moments_round_trip():
    """inertia_to_moments must exactly invert moments_to_inertia."""
    for _ in range(20):
        m = RNG.uniform(0.1, 10.0)
        c = RNG.uniform(-1, 1, size=3)
        eigs = RNG.uniform(0.01, 5.0, size=3)
        q, _ = np.linalg.qr(RNG.normal(size=(3, 3)))
        i_com = q @ np.diag(eigs) @ q.T

        b = inertia_to_moments(m, c, i_com)
        m2, c2, i2 = moments_to_inertia(b)

        assert m2 == pytest.approx(m)
        assert np.allclose(c2, c)
        assert np.allclose(i2, i_com)


def test_h2_over_12_correction_is_needed_for_a_single_chunky_voxel():
    """The clearest demonstration of why the +h^2/12 term matters: a single
    voxel (a "chunky" discrete mass element, exactly the representation used
    for CP-SAT slug ghosts later in this project) has a real, nonzero
    self-inertia about its own center -- that of a uniform cube of side h,
    (1/6) m h^2 on the diagonal. Point-mass bookkeeping that omits the
    +h^2/12 term returns exactly zero self-inertia for every voxel, which is
    wrong regardless of how much the grid is refined elsewhere: it is not a
    discretization error that shrinks with resolution, it is a term the
    point-mass model structurally cannot produce."""
    h = 0.05
    density = 7850.0
    voxel_mass = density * h**3
    true_self_i = (1.0 / 6.0) * voxel_mass * h**2

    grid = _cube_grid(n=5, h=h, origin=(0.0, 0.0, 0.0))
    rho = np.zeros(grid.n_voxels)
    idx = np.ravel_multi_index((2, 2, 2), (5, 5, 5))
    rho[idx] = density

    a = build_moment_operator(grid)
    b = a @ rho
    _m, _c, i_com_corrected = moments_to_inertia(b)
    assert np.allclose(np.diag(i_com_corrected), true_self_i, rtol=1e-10)

    # rebuild the second-moment tensor WITHOUT the +h^2/12 term
    centers = grid.centers()
    h3 = h**3
    s_world_uncorrected = h3 * np.einsum("v,vi,vj->ij", rho, centers, centers)
    m_val = h3 * np.sum(rho)
    c_val = h3 * np.einsum("v,vi->i", rho, centers) / m_val
    s_com_uncorrected = s_world_uncorrected - m_val * np.outer(c_val, c_val)
    i_com_uncorrected = np.trace(s_com_uncorrected) * np.eye(3) - s_com_uncorrected

    # the point-mass model puts every voxel's mass exactly at its center, so
    # it necessarily returns zero self-inertia for a single voxel -- not an
    # approximation of true_self_i, an absence of the term entirely
    assert np.allclose(i_com_uncorrected, 0.0, atol=1e-12)
    assert true_self_i > 1e-10  # sanity: the true value is not itself negligible


def test_nullspace_projector_is_orthogonal_to_row_space():
    """A rho projected into ker(A) must satisfy A @ rho_null ~= 0."""
    grid = _cube_grid(n=7, h=0.02)
    a = build_moment_operator(grid)
    project = nullspace_projector(a)

    rho = RNG.uniform(0, 1, size=grid.n_voxels)
    rho_null = project(rho)
    residual = a @ rho_null
    assert np.max(np.abs(residual)) < 1e-8


def test_project_to_constraint_hits_target_exactly():
    grid = _cube_grid(n=7, h=0.02)
    a = build_moment_operator(grid)
    rho = RNG.uniform(0, 5, size=grid.n_voxels)
    b_target = a @ rho + RNG.normal(scale=0.01, size=10)

    rho_adjusted = project_to_constraint(rho, a, b_target)
    assert np.allclose(a @ rho_adjusted, b_target, atol=1e-6)


def test_gram_matrix_is_spd_for_nondegenerate_grid():
    grid = _cube_grid(n=6, h=0.015)
    a = build_moment_operator(grid)
    g = gram(a)
    eigvals = np.linalg.eigvalsh(g)
    assert np.all(eigvals > 0), "moment operator should have rank 10 for a generic grid"
