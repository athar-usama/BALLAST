"""The observability atlas: which motion regime reveals which of the 10
inertial parameters, quantified as a Fisher information matrix (FIM) over
a single flat 10-vector parameterization

    theta = (m, cx, cy, cz, Ixx, Iyy, Izz, Ixy, Ixz, Iyz)

rather than the raw moments `b` from `ballast.moments.operator` -- this
parameterization is chosen specifically because it makes the two
homogeneity theorems trivial DIRECTIONS to write down and check, which is
the whole point of the "physics-derived correctness test" below: the
theorems are not just documentation, they are executable unit tests on
whatever the FIM comes out to be, for every regime.

The observable for each regime is built directly from the modules already
in this project (`ballast.physics.rigid`, `ballast.physics.analytic`), and
the FIM is `J^T Sigma^-1 J` with `J` a finite-difference Jacobian of that
observable with respect to `theta` -- 10 columns, so a numeric Jacobian is
exact enough that autodiff would be overkill, exactly as for the linear
tumble estimators.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass
from typing import Literal

import numpy as np

from ballast.physics.analytic import rocking_frequency_forward, rolling_acceleration_forward
from ballast.physics.rigid import RigidBodyState, integrate

N_THETA = 10
Regime = Literal["tumble", "rocking", "rolling", "tap", "toss_drag"]


# --------------------------------------------------------------------------
# theta packing and the two homogeneity directions
# --------------------------------------------------------------------------


def pack_theta(m: float, c: np.ndarray, i_com: np.ndarray) -> np.ndarray:
    ixx, iyy, izz = i_com[0, 0], i_com[1, 1], i_com[2, 2]
    ixy, ixz, iyz = i_com[0, 1], i_com[0, 2], i_com[1, 2]
    return np.array([m, *c, ixx, iyy, izz, ixy, ixz, iyz], dtype=np.float64)


def unpack_theta(theta: np.ndarray) -> tuple[float, np.ndarray, np.ndarray]:
    m = theta[0]
    c = theta[1:4]
    ixx, iyy, izz, ixy, ixz, iyz = theta[4:10]
    i_com = np.array([[ixx, ixy, ixz], [ixy, iyy, iyz], [ixz, iyz, izz]])
    return m, c, i_com


def mass_direction(theta: np.ndarray) -> np.ndarray:
    """Pure mass perturbation at fixed (c, I): e_0 in this parameterization.
    Torque-free tumbling does not depend on m at all, so this is always a
    null direction of the tumble FIM, and is exactly HALF of the combined
    gravity-regime homogeneity direction below."""
    d = np.zeros(N_THETA)
    d[0] = 1.0
    return d


def com_direction(theta: np.ndarray, axis: int) -> np.ndarray:
    """Pure center-of-mass perturbation along one axis, at fixed (m, I):
    used to confirm that an observable which never involves translation
    (pure angular-velocity tumble tracking, with no position track) is, as
    expected, exactly independent of c along every axis."""
    d = np.zeros(N_THETA)
    d[1 + axis] = 1.0
    return d


def inertia_scale_direction(theta: np.ndarray) -> np.ndarray:
    """Pure inertia-SCALE perturbation at fixed (m, c): (0, 0, I). Euler's
    equations are homogeneous of degree 1 in I (dw/dt = I^-1(...) is
    invariant under I -> lambda I for any lambda), so this is always a
    null direction of the tumble FIM too -- a SEPARATE direction from pure
    mass, which is why tumble's null space is 2-dimensional, not 1."""
    _m, _c, i_com = unpack_theta(theta)
    d = np.zeros(N_THETA)
    d[4:10] = [i_com[0, 0], i_com[1, 1], i_com[2, 2], i_com[0, 1], i_com[0, 2], i_com[1, 2]]
    return d


def homogeneity_direction(theta: np.ndarray) -> np.ndarray:
    """The Theorem 1 direction (m, 0, I): under gravity plus unilateral
    contact with an unknown reaction force, the dynamics are homogeneous of
    degree 1 in mass (m -> lambda m, I -> lambda I leaves the trajectory
    identical). The infinitesimal generator of that scaling group AT theta
    is exactly (m, 0, I) using theta's OWN m and I -- not a unit mass
    component plus theta's I, which is a different, non-proportional
    direction and does not actually lie along the scaling curve. Built
    directly (not as `mass_direction(theta) + inertia_scale_direction
    (theta)`, which mismatches `mass_direction`'s fixed unit-mass
    convention against `inertia_scale_direction`'s theta-scaled one) to
    avoid exactly that mismatch."""
    m, _c, i_com = unpack_theta(theta)
    return pack_theta(m, np.zeros(3), i_com)


# --------------------------------------------------------------------------
# per-regime observables
# --------------------------------------------------------------------------


def tumble_observable(
    theta: np.ndarray, omega0_body: np.ndarray, t_eval_s: float, dt_s: float = 1e-3
) -> np.ndarray:
    """Body-frame angular velocity after torque-free tumbling for
    `t_eval_s`, starting from `omega0_body`. Depends on I only (not m, not
    c), which the FIM computed from this observable must show as an exact,
    not approximate, null direction."""
    _m, _c, i_com = unpack_theta(theta)
    n_steps = max(round(t_eval_s / dt_s), 1)
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0, 0, 0]),
        omega_body_rad_s=omega0_body,
    )
    traj = integrate(state0, m=1.0, i_body=i_com, dt=t_eval_s / n_steps, n_steps=n_steps)
    return traj.omega_body_rad_s[-1]


def rocking_observable(theta: np.ndarray, pivot_point_body: np.ndarray, g: float = 9.80665) -> np.ndarray:
    """Rocking frequency about a fixed pivot point: depends on m and I only
    through the ratio m / I_pivot (Theorem 1's escape hatch does not apply
    here, since there is no known applied force), plus on c through the
    pivot distance `d`. A simplified single effective pivot moment
    (average of the principal moments plus the parallel-axis term) stands
    in for a full 3D pivot-axis treatment, adequate for the observability
    STRUCTURE this atlas characterizes."""
    m, c, i_com = unpack_theta(theta)
    d = np.linalg.norm(c - pivot_point_body)
    i_pivot = np.trace(i_com) / 3.0 + m * d**2
    omega0 = rocking_frequency_forward(m, d, i_pivot, g=g)
    return np.array([omega0])


def rolling_observable(theta: np.ndarray, roll_axis: int, radius_m: float, incline_rad: float) -> np.ndarray:
    """Rolling acceleration down an incline: depends on m and the rolling-
    axis moment of inertia only through the ratio I_axis / (m R^2)."""
    m, _c, i_com = unpack_theta(theta)
    i_axis = i_com[roll_axis, roll_axis]
    moi_factor = i_axis / (m * radius_m**2)
    a = rolling_acceleration_forward(moi_factor, incline_rad)
    return np.array([a])


def tap_observable(theta: np.ndarray, r_app_body: np.ndarray, impulse_body: np.ndarray) -> np.ndarray:
    """A known tap impulse: full rank. `delta_v = J/m` gives absolute mass
    directly (breaking Theorem 1's degeneracy, since J is KNOWN rather than
    an unknown reaction force); `delta_omega = I^-1 (r_eff x J)` with
    `r_eff = r_app - c` couples in both c and the absolute scale of I."""
    m, c, i_com = unpack_theta(theta)
    delta_v = impulse_body / m
    r_eff = r_app_body - c
    torque_impulse = np.cross(r_eff, impulse_body)
    delta_omega = np.linalg.solve(i_com, torque_impulse)
    return np.concatenate([delta_v, delta_omega])


def toss_drag_observable(
    theta: np.ndarray,
    v0_body_origin: np.ndarray,
    omega0_body: np.ndarray,
    drag_coeff: float,
    t_flight_s: float,
    dt_s: float = 1e-3,
) -> np.ndarray:
    """A ballistic toss with drag, tracking a body-fixed FEATURE point (not
    the center of mass -- see `ballast.identify.linear_tumble`). Full rank:
    m enters through the drag deceleration of the center of mass, c enters
    directly through the `R(t) @ c` offset between the feature point and
    the center of mass, and I enters through the tumbling that determines
    R(t)."""
    m, c, i_com = unpack_theta(theta)

    def drag_force(state: RigidBodyState) -> np.ndarray:
        g_force = np.array([0.0, 0.0, -m * 9.80665])
        v = state.velocity_m_s
        speed = np.linalg.norm(v)
        drag = -drag_coeff * speed * v if speed > 0 else np.zeros(3)
        return g_force + drag

    n_steps = max(round(t_flight_s / dt_s), 1)
    # the feature point sits at the body-frame origin, so the COM starts
    # offset from it by -c in the body frame, i.e. +c in world at q=identity...
    # simplify: initialize the COM position at the origin and recover the
    # feature position afterward via feature = com_world - R(t) @ c
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=v0_body_origin,
        quaternion=np.array([1.0, 0, 0, 0]),
        omega_body_rad_s=omega0_body,
    )
    traj = integrate(
        state0, m=m, i_body=i_com, dt=t_flight_s / n_steps, n_steps=n_steps, force_world_fn=drag_force
    )
    r_final = traj.rotation_matrices()[-1]
    feature_world = traj.position_m[-1] - r_final @ c
    return feature_world


_REGIME_ARITY: dict[Regime, int] = {
    "tumble": 3,
    "rocking": 1,
    "rolling": 1,
    "tap": 6,
    "toss_drag": 3,
}


# --------------------------------------------------------------------------
# Fisher information
# --------------------------------------------------------------------------


def numeric_jacobian(
    observable_fn: Callable[[np.ndarray], np.ndarray], theta: np.ndarray, eps: float = 1e-6
) -> np.ndarray:
    """(n_obs, 10) Jacobian of `observable_fn` w.r.t. theta by central
    differences -- 10 columns, so this is exact enough that autodiff would
    be overkill, matching the convention used throughout this project's
    identification modules."""
    n_obs = observable_fn(theta).shape[0]
    jac = np.zeros((n_obs, N_THETA))
    for i in range(N_THETA):
        step = eps * max(abs(theta[i]), 1.0)
        d = np.zeros(N_THETA)
        d[i] = step
        jac[:, i] = (observable_fn(theta + d) - observable_fn(theta - d)) / (2 * step)
    return jac


def fisher_information(jac: np.ndarray, obs_cov: np.ndarray) -> np.ndarray:
    """FIM = J^T Sigma^-1 J, the standard Gaussian-observation Fisher
    information for a linearized observable with covariance `obs_cov`."""
    sigma_inv = np.linalg.inv(obs_cov)
    return jac.T @ sigma_inv @ jac


@dataclass(frozen=True)
class NullspaceCheck:
    eigenvalues: np.ndarray  # (10,) ascending
    n_null: int  # count of eigenvalues below the tolerance
    homogeneity_residual: (
        float  # ||F @ homogeneity_direction|| / ||F||, should be ~0 for gravity+contact regimes
    )
    mass_residual: float  # ||F @ mass_direction||, should be ~0 for tumble
    inertia_scale_residual: float  # ||F @ inertia_scale_direction||, should be ~0 for tumble


def check_nullspace(fim: np.ndarray, theta: np.ndarray, atol: float = 1e-6) -> NullspaceCheck:
    """The physics-derived correctness test: verify specific, KNOWN
    directions (from the two homogeneity theorems) are or are not in the
    FIM's null space, to near machine precision -- if the relevant
    residual is not tiny where the theorem says it should be, the
    SIMULATOR is wrong, not just the FIM's conditioning.

    `n_null` (a global eigenvalue count against `atol * ||F||`) is reported
    for diagnostics only and should not be asserted on across regimes that
    mix parameters of very different physical units (mass ~1, position
    ~1e-2, inertia ~1e-3 in this project's typical scale): the resulting
    FIM can have eigenvalues spanning ten or more orders of magnitude, at
    which point double-precision eigenvalue computation for the SMALL
    eigenvalues is dominated by round-off from the large ones (observed
    directly: small negative eigenvalues appear for a full-rank FIM here,
    which is impossible for a true PSD matrix and is exactly this
    round-off, not a sign of missing rank). The `*_residual` fields do not
    have this problem, since each is a single well-scaled directional
    projection, and are what the tests in this project actually assert on.
    """
    eigvals = np.sort(np.linalg.eigvalsh(fim))
    scale = np.linalg.norm(fim) + 1e-300
    n_null = int(np.sum(eigvals < atol * scale))

    homog = homogeneity_direction(theta)
    mass_d = mass_direction(theta)
    scale_d = inertia_scale_direction(theta)

    def residual(direction: np.ndarray) -> float:
        v = fim @ direction
        return float(np.linalg.norm(v)) / scale

    return NullspaceCheck(
        eigenvalues=eigvals,
        n_null=n_null,
        homogeneity_residual=residual(homog),
        mass_residual=residual(mass_d),
        inertia_scale_residual=residual(scale_d),
    )
