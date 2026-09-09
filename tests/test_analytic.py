"""Tests for the closed-form per-regime models. Rocking and rolling are
validated against their own governing ODEs (not the general rigid-body +
contact simulator, which is a separate forward-only generator built later
purely to render clips); the tap-impulse recovery is validated against the
impulse-momentum theorem directly, since that relationship is exact and
algebraic rather than differential."""

from __future__ import annotations

import numpy as np
import pytest

from ballast.physics.analytic import (
    GRAVITY_M_S2,
    critical_tip_angle,
    estimate_frequency_fft,
    inertia_scale_from_impulse,
    mass_from_impulse,
    rocking_frequency_forward,
    rocking_mass_over_inertia,
    rolling_acceleration_forward,
    rolling_moi_factor_from_acceleration,
)


def _integrate_small_angle_pendulum(omega0: float, theta0: float, dt: float, n_steps: int) -> np.ndarray:
    """theta'' = -omega0^2 * theta, plain RK4 on the 2-state ODE. A minimal,
    independent reference implementation -- this is deliberately NOT built
    from ballast.physics.rigid, so the test is checking the analytic model
    against an independent numerical solution, not against itself."""

    def deriv(state):
        theta, theta_dot = state
        return np.array([theta_dot, -(omega0**2) * theta])

    state = np.array([theta0, 0.0])
    thetas = [state[0]]
    for _ in range(n_steps):
        k1 = deriv(state)
        k2 = deriv(state + dt / 2 * k1)
        k3 = deriv(state + dt / 2 * k2)
        k4 = deriv(state + dt * k3)
        state = state + (dt / 6) * (k1 + 2 * k2 + 2 * k3 + k4)
        thetas.append(state[0])
    return np.array(thetas)


def test_rocking_frequency_round_trip():
    m, d, i_pivot = 0.4, 0.03, 0.002
    omega0 = rocking_frequency_forward(m, d, i_pivot)
    recovered_ratio = rocking_mass_over_inertia(omega0, d)
    assert recovered_ratio == pytest.approx(m / i_pivot, rel=1e-10)


def test_rocking_frequency_recovered_from_fft_of_a_simulated_tilt_trace():
    """The realistic path: given only a tracked tilt-angle time series (not
    the frequency itself), FFT peak-picking plus the analytic model should
    recover m/I_pivot to good accuracy from a few seconds of clean small-
    angle oscillation."""
    m, d, i_pivot = 0.5, 0.025, 0.0015
    true_omega0 = rocking_frequency_forward(m, d, i_pivot)

    dt = 1.0 / 240.0  # a plausible high-frame-rate tracker
    duration_s = 3.0
    n_steps = int(duration_s / dt)
    theta = _integrate_small_angle_pendulum(true_omega0, theta0=0.05, dt=dt, n_steps=n_steps)

    est_omega0 = estimate_frequency_fft(theta, dt)
    assert est_omega0 == pytest.approx(true_omega0, rel=0.02)

    recovered_ratio = rocking_mass_over_inertia(est_omega0, d)
    assert recovered_ratio == pytest.approx(m / i_pivot, rel=0.04)


def test_rolling_acceleration_round_trip():
    incline = np.radians(15.0)
    true_moi_factor = 2.0 / 5.0  # a uniform sphere
    a = rolling_acceleration_forward(true_moi_factor, incline)
    recovered = rolling_moi_factor_from_acceleration(a, incline)
    assert recovered == pytest.approx(true_moi_factor, rel=1e-10)


def test_rolling_distinguishes_hollow_from_solid():
    """A concrete, qualitative sanity check: a hollow shell (moi factor
    2/3) accelerates measurably slower down the same incline than a solid
    sphere (moi factor 2/5)."""
    incline = np.radians(20.0)
    a_solid = rolling_acceleration_forward(2.0 / 5.0, incline)
    a_hollow = rolling_acceleration_forward(2.0 / 3.0, incline)
    assert a_solid > a_hollow
    assert (a_solid - a_hollow) / a_solid > 0.1


@pytest.mark.parametrize(
    "offset,height,expected_deg",
    [
        (1.0, 1.0, 45.0),
        (0.0, 1.0, 0.0),
        (1.0, 1e6, 0.0),
    ],
)
def test_critical_tip_angle_geometry(offset, height, expected_deg):
    angle = critical_tip_angle(offset, height)
    assert np.degrees(angle) == pytest.approx(expected_deg, abs=1e-3)


def test_critical_tip_angle_increases_with_offset():
    """A center of mass further from the pivot edge tips at a larger angle
    (more stable) -- the qualitative direction a reviewer would sanity-check."""
    angles = [critical_tip_angle(offset, height_above_pivot_m=1.0) for offset in (0.1, 0.3, 0.6)]
    assert angles == sorted(angles)


def test_mass_from_impulse_is_exact_for_a_clean_impulse():
    m_true = 0.35
    impulse = np.array([1.2, -0.4, 0.3])
    delta_v = impulse / m_true
    recovered = mass_from_impulse(delta_v, impulse)
    assert recovered == pytest.approx(m_true, rel=1e-10)


def test_inertia_scale_from_impulse_recovers_the_true_absolute_scale():
    """The concrete demonstration of Theorem 1's escape hatch: given only
    the SCALE-FREE inertia direction (as a torque-free tumble alone would
    supply) plus one known tap, recover the true absolute inertia scale."""
    i_true = np.diag([1.0, 2.0, 3.0]) * 5.0  # true absolute inertia tensor
    i_direction = i_true / np.linalg.norm(i_true)  # what a tumble-only fit would give: direction, not scale

    r_eff = np.array([0.05, 0.0, 0.02])
    impulse = np.array([0.0, 0.8, 0.1])
    torque_impulse = np.cross(r_eff, impulse)
    delta_omega = np.linalg.solve(i_true, torque_impulse)  # exact impulse-momentum response

    recovered_scale = inertia_scale_from_impulse(i_direction, r_eff, impulse, delta_omega)
    true_scale = np.linalg.norm(i_true)
    assert recovered_scale == pytest.approx(true_scale, rel=1e-8)


def test_gravity_constant_is_standard():
    assert GRAVITY_M_S2 == pytest.approx(9.80665)
