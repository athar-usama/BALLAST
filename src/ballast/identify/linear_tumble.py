"""Recovering inertial parameters from a tracked trajectory, without ever
simulating or fitting an ODE.

Two results, both linear-algebraic rather than optimization-based, which
matters because it means no initialization, no local minima, and a
condition number that IS the diagnostic for how hard the problem is:

  - `inertia_from_angular_momentum`: for torque-free tumbling, world-frame
    angular momentum L = R(t) I omega(t) is exactly conserved (Theorem,
    ballast.physics.rigid conservation tests). Differencing this identity
    between frame pairs gives a homogeneous linear system in the 6
    independent entries of I, solved by SVD -- the smallest singular
    vector is I up to an unknown positive scale, exactly matching the
    theory: torque-free tumbling identifies inertia RATIOS (5 of the 6
    independent components, since scale is not observable) plus the
    principal-axis frame, never absolute scale. The condition number of
    this system is the near-symmetry diagnostic: a body close to an
    axisymmetric shape gives a near-degenerate second-smallest singular
    value, which is exactly where inertia-ratio estimation gets hard.

  - `com_from_free_flight`: in torque-free flight, the tracked position of
    the body-frame origin obeys p(t) = X0 + V0 t - 0.5 g t^2 z - R(t) c,
    which is LINEAR in (X0, V0, c) given the tracked rotation R(t) -- an
    ordinary linear least-squares problem. This routine also detects the
    one real degeneracy: if R(t) is a rotation about a single fixed axis
    (uniform spin, not genuine tumbling), the component of c along that
    axis aliases with X0 and is flagged as unidentified rather than
    silently returned.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY_M_S2 = 9.80665

# index map for the 6 independent entries of a symmetric 3x3 tensor,
# vec(I) = [Ixx, Iyy, Izz, Ixy, Ixz, Iyz]
_SYM_INDEX_PAIRS = ((0, 0), (1, 1), (2, 2), (0, 1), (0, 2), (1, 2))


def _omega_matrix(omega: np.ndarray) -> np.ndarray:
    """The (3, 6) matrix Omega(w) such that Omega(w) @ vec(I) == I @ w, for
    any symmetric I. Built once per frame from the angular velocity alone."""
    wx, wy, wz = omega
    return np.array(
        [
            [wx, 0.0, 0.0, wy, wz, 0.0],
            [0.0, wy, 0.0, wx, 0.0, wz],
            [0.0, 0.0, wz, 0.0, wx, wy],
        ]
    )


def _vec_to_sym(v: np.ndarray) -> np.ndarray:
    ixx, iyy, izz, ixy, ixz, iyz = v
    return np.array(
        [
            [ixx, ixy, ixz],
            [ixy, iyy, iyz],
            [ixz, iyz, izz],
        ]
    )


@dataclass(frozen=True)
class InertiaFromTumbleResult:
    i_direction: np.ndarray  # (3, 3), unit-Frobenius-norm symmetric tensor, scale unobservable
    condition_number: float  # ratio of largest to second-smallest singular value of the stacked system
    singular_values: np.ndarray  # (6,) ascending order, for diagnostics
    principal_ratios: np.ndarray  # (3,) principal moments of i_direction, ascending, normalized so max=1


def inertia_from_angular_momentum(rotations: np.ndarray, omegas_body: np.ndarray) -> InertiaFromTumbleResult:
    """Recover the inertia tensor UP TO POSITIVE SCALE from a tracked
    torque-free tumble, by solving the homogeneous linear system built from
    conservation of world-frame angular momentum: for any two frames,

        R(t1) @ (I @ w(t1))  ==  R(t2) @ (I @ w(t2))  ==  L  (constant)

    Differencing frame pairs eliminates the unknown constant L, leaving a
    linear system `M @ vec(I) = 0`. No absolute scale is present in this
    system by construction (it is homogeneous), which is exactly the
    theorem: torque-free tumbling identifies inertia ratios and the
    principal-axis frame, never mass or the overall scale of I.

    `rotations` is (F, 3, 3) body-to-world rotation matrices, `omegas_body`
    is (F, 3) body-frame angular velocities, both from a tracked (not
    simulated) trajectory. Uses ALL consecutive frame pairs, not just one,
    for a well-determined least-squares null vector.
    """
    f = rotations.shape[0]
    if f < 3:
        raise ValueError("need at least 3 frames to form a well-posed differenced system")

    rows = []
    for t in range(f - 1):
        r1, w1 = rotations[t], omegas_body[t]
        r2, w2 = rotations[t + 1], omegas_body[t + 1]
        diff = r1 @ _omega_matrix(w1) - r2 @ _omega_matrix(w2)
        rows.append(diff)
    m = np.concatenate(rows, axis=0)  # (3*(F-1), 6)

    _u, s, vt = np.linalg.svd(m)
    vec_i = vt[-1]  # smallest singular value's right singular vector

    i_direction = _vec_to_sym(vec_i)
    # normalize sign and scale: positive-definite convention, unit Frobenius norm
    if np.trace(i_direction) < 0:
        i_direction = -i_direction
    i_direction = i_direction / np.linalg.norm(i_direction)

    principal_vals = np.sort(np.linalg.eigvalsh(i_direction))
    principal_ratios = principal_vals / principal_vals[-1]

    s_sorted = np.sort(s)
    condition_number = float(s_sorted[1] / max(s_sorted[0], 1e-300))

    return InertiaFromTumbleResult(
        i_direction=i_direction,
        condition_number=condition_number,
        singular_values=s_sorted,
        principal_ratios=principal_ratios,
    )


@dataclass(frozen=True)
class ComFromFlightResult:
    c_body: np.ndarray  # (3,) recovered center of mass in the body frame
    c_sigma: np.ndarray  # (3,) per-component uncertainty (from least-squares residual covariance)
    x0: np.ndarray  # (3,) recovered initial world-frame position of the body-frame origin
    v0: np.ndarray  # (3,) recovered initial velocity
    fixed_axis_degenerate: bool  # True if R(t) rotates about a single fixed axis (c along it is unobservable)
    degenerate_axis_body: np.ndarray | None  # that axis, in the body frame, if degenerate


def _detect_fixed_axis(rotations: np.ndarray, tol: float = 1e-6) -> np.ndarray | None:
    """Check whether every R(t) shares a common invariant axis (i.e. the
    motion is a uniform spin about one body-frame axis, not genuine
    tumbling). Returns that axis in the body frame if so, else None."""
    # a common invariant axis n satisfies R(t) @ n = n for ALL t (since the
    # rotation is a pure spin about n, n is a fixed point of every R(t))
    accum = np.zeros((3, 3))
    for r in rotations:
        accum += (r - np.eye(3)).T @ (r - np.eye(3))
    vals, vecs = np.linalg.eigh(accum)
    if vals[0] < tol * max(vals[-1], 1e-12):
        return vecs[:, 0]
    return None


def com_from_free_flight(
    rotations: np.ndarray,
    positions_world: np.ndarray,
    t_s: np.ndarray,
    gravity: float = GRAVITY_M_S2,
    gravity_axis: int = 2,
) -> ComFromFlightResult:
    """Recover the body-frame center of mass `c` (and, incidentally, the
    initial position and velocity of the body-frame origin) from a tracked
    torque-free flight, via the linear model

        p(t) = X0 + V0 * t - 0.5 * g * t^2 * z_hat - R(t) @ c

    `positions_world` is the tracked (F, 3) world-frame position of the
    body-frame ORIGIN (not the center of mass -- that is what we are
    solving for), `rotations` the (F, 3, 3) tracked rotations, `t_s` the
    (F,) frame timestamps. Detects the one real degeneracy: if the motion
    is a uniform spin about a single fixed axis, the component of `c` along
    that axis aliases with `X0` and cannot be separated; that component is
    still returned (for downstream convenience) but flagged as degenerate.
    """
    f = rotations.shape[0]
    fixed_axis = _detect_fixed_axis(rotations)

    # unknowns, stacked: [X0 (3), V0 (3), c (3)] -- 9 unknowns
    rows = []
    rhs = []
    gravity_term = np.zeros(3)
    gravity_term[gravity_axis] = -0.5 * gravity
    for t_idx in range(f):
        t = t_s[t_idx]
        r = rotations[t_idx]
        # p(t) = X0 + V0*t + gravity_term*t^2 - R(t) @ c
        # => [I3, t*I3, -R(t)] @ [X0; V0; c] = p(t) - gravity_term*t^2
        block = np.concatenate([np.eye(3), t * np.eye(3), -r], axis=1)
        rows.append(block)
        rhs.append(positions_world[t_idx] - gravity_term * t**2)
    a = np.concatenate(rows, axis=0)  # (3F, 9)
    b = np.concatenate(rhs, axis=0)  # (3F,)

    solution, _residuals, _rank, _sv = np.linalg.lstsq(a, b, rcond=None)
    x0, v0, c_body = solution[0:3], solution[3:6], solution[6:9]

    # residual-based per-component sigma (diagonal of the covariance estimate)
    dof = max(a.shape[0] - a.shape[1], 1)
    resid_vec = a @ solution - b
    sigma2 = float(resid_vec @ resid_vec) / dof
    try:
        cov = sigma2 * np.linalg.inv(a.T @ a)
        c_sigma = np.sqrt(np.clip(np.diag(cov)[6:9], 0, None))
    except np.linalg.LinAlgError:
        c_sigma = np.full(3, np.nan)

    return ComFromFlightResult(
        c_body=c_body,
        c_sigma=c_sigma,
        x0=x0,
        v0=v0,
        fixed_axis_degenerate=fixed_axis is not None,
        degenerate_axis_body=fixed_axis,
    )
