"""The moment operator A: a voxelized interior density field maps to exactly
10 numbers -- mass, center of mass, and inertia tensor -- and nothing else.

Every downstream theorem in this project (ghost mass, sharp bounds, the
observability atlas) rests on this operator being exact. In particular the
second-moment rows use the closed-form integral of x_i*x_j over a cube, not
the point-mass approximation rho_v * x_v_i * x_v_j: the +h^2/12 correction on
the diagonal is the difference between a moment estimate that converges under
grid refinement and one that silently does not.

Moment ordering, b = A @ rho, b has shape (10,):
    b[0]      = m                      total mass
    b[1:4]    = m * c                  first moment about the grid origin
    b[4:10]   = S_xx, S_yy, S_zz, S_xy, S_xz, S_yz
                the raw second-moment tensor about the grid origin,
                S_ij = integral(rho * x_i * x_j) dV
"""

from __future__ import annotations

import numpy as np

from ballast.moments.voxelgrid import VoxelGrid

N_MOMENTS = 10
_S_DIAG = (4, 5, 6)  # indices of Sxx, Syy, Szz within b
_S_OFFDIAG = (7, 8, 9)  # indices of Sxy, Sxz, Syz within b
_S_OFFDIAG_PAIRS = ((0, 1), (0, 2), (1, 2))  # (i, j) axis pairs matching Sxy, Sxz, Syz


def build_moment_operator(grid: VoxelGrid) -> np.ndarray:
    """Return the dense (10, V) matrix A such that `A @ rho.ravel()` gives the
    10 raw moments of a density field `rho` (kg/m^3 per voxel) sampled on `grid`.

    V is kept small enough throughout this project (<= ~10^5) that a dense
    array is simpler and no slower than a sparse one, and it composes cleanly
    with dense SVD-based null-space and least-squares operations elsewhere.
    """
    centers = grid.centers()  # (V, 3)
    h = grid.voxel_size_m
    h3 = h**3
    h2_12 = h**2 / 12.0
    v = centers.shape[0]

    a = np.empty((N_MOMENTS, v), dtype=np.float64)
    a[0] = h3
    a[1] = h3 * centers[:, 0]
    a[2] = h3 * centers[:, 1]
    a[3] = h3 * centers[:, 2]
    for row, axis in zip(_S_DIAG, range(3)):
        a[row] = h3 * (centers[:, axis] ** 2 + h2_12)
    for row, (i, j) in zip(_S_OFFDIAG, _S_OFFDIAG_PAIRS):
        a[row] = h3 * centers[:, i] * centers[:, j]
    return a


def _second_moment_matrix(s: np.ndarray) -> np.ndarray:
    sxx, syy, szz, sxy, sxz, syz = s
    return np.array(
        [
            [sxx, sxy, sxz],
            [sxy, syy, syz],
            [sxz, syz, szz],
        ],
        dtype=np.float64,
    )


def _second_moment_vector(s_mat: np.ndarray) -> np.ndarray:
    return np.array(
        [s_mat[0, 0], s_mat[1, 1], s_mat[2, 2], s_mat[0, 1], s_mat[0, 2], s_mat[1, 2]],
        dtype=np.float64,
    )


def moments_to_inertia(b: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    """Raw moments about the grid origin -> (mass, center of mass, inertia
    tensor about the center of mass), via the parallel-axis theorem.

    inertia (Euler's convention): I_ij = tr(S) * delta_ij - S_ij, where
    S = integral(rho * x x^T) dV is the raw second-moment tensor.
    """
    b = np.asarray(b, dtype=np.float64)
    m = float(b[0])
    if m <= 0:
        raise ValueError(f"non-positive mass in moment vector: m={m}")
    c = b[1:4] / m
    s_world = _second_moment_matrix(b[4:10])
    s_com = s_world - m * np.outer(c, c)
    i_com = np.trace(s_com) * np.eye(3) - s_com
    return m, c, i_com


def inertia_to_moments(m: float, c: np.ndarray, i_com: np.ndarray) -> np.ndarray:
    """Inverse of `moments_to_inertia`: (m, c, I_about_com) -> raw moments b
    about the grid origin. Exact inverse; used to build synthetic ground
    truth and in round-trip tests."""
    c = np.asarray(c, dtype=np.float64)
    i_com = np.asarray(i_com, dtype=np.float64)
    s_com = (np.trace(i_com) / 2.0) * np.eye(3) - i_com
    s_world = s_com + m * np.outer(c, c)
    b = np.empty(N_MOMENTS, dtype=np.float64)
    b[0] = m
    b[1:4] = m * c
    b[4:10] = _second_moment_vector(s_world)
    return b


def gram(a: np.ndarray) -> np.ndarray:
    """A @ A^T, the (10, 10) Gram matrix used by the null-space projector and
    by every Fisher-information computation downstream."""
    return a @ a.T


def nullspace_projector(a: np.ndarray):
    """Return a callable `project(rho) -> rho_null` giving the component of
    `rho` lying in the null space of A (the "ghost" directions: density
    perturbations invisible to every moment).

    Implemented as `rho - A^T (A A^T)^-1 A rho`, the minimum-norm projection
    onto ker(A). This never materializes the (V - 10)-dimensional null-space
    basis explicitly, which would be a `(V, V-10)` array -- at V = 32**3 that
    is on the order of hundreds of gigabytes.
    """
    aat = gram(a)
    aat_inv = np.linalg.inv(aat)

    def project(rho: np.ndarray) -> np.ndarray:
        b = a @ rho
        correction = a.T @ (aat_inv @ b)
        return rho - correction

    return project


def project_to_constraint(rho: np.ndarray, a: np.ndarray, b_target: np.ndarray) -> np.ndarray:
    """Minimum-L2-change adjustment of `rho` so that `A @ rho == b_target`
    exactly (up to solver tolerance): rho + A^T (A A^T)^-1 (b_target - A rho).

    This is the fusion primitive for T5: given an appearance-based prior
    density field as `rho`, project it onto the affine set of interiors
    consistent with the motion-measured moments `b_target`.
    """
    aat = gram(a)
    residual = b_target - a @ rho
    correction = a.T @ np.linalg.solve(aat, residual)
    return rho + correction
