"""Tests for the observability atlas: these are physics-derived
correctness tests, not just numerical sanity checks. If the FIM's null
space does not exactly match what the two homogeneity theorems predict,
the underlying simulator is wrong, not merely ill-conditioned.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.identify.observability import (
    check_nullspace,
    fisher_information,
    homogeneity_direction,
    numeric_jacobian,
    pack_theta,
    rocking_observable,
    rolling_observable,
    tap_observable,
    toss_drag_observable,
    tumble_observable,
)

M_TRUE = 0.5
C_TRUE = np.array([0.012, -0.018, 0.009])
I_TRUE = np.diag([0.0020, 0.0032, 0.0045])  # distinct principal moments: genuinely asymmetric
THETA = pack_theta(M_TRUE, C_TRUE, I_TRUE)


def test_pack_unpack_round_trip():
    from ballast.identify.observability import unpack_theta

    m, c, i_com = unpack_theta(THETA)
    assert m == pytest.approx(M_TRUE)
    assert np.allclose(c, C_TRUE)
    assert np.allclose(i_com, I_TRUE)


def test_tumble_omega_only_is_independent_of_mass_scale_and_every_com_axis():
    """Angular-velocity-only tumble tracking (no position track) depends on
    I alone: mass, inertia scale, AND every component of c must all be
    exact null directions, to near machine precision. Uses several
    well-spaced time samples so the FIM has enough rank to distinguish
    'genuinely null' from 'not enough equations yet'."""
    from ballast.identify.observability import com_direction

    omega0 = np.array([0.3, 4.0, 0.1])  # genuine NPA tumble

    def observable(theta):
        return np.concatenate([tumble_observable(theta, omega0, t_eval_s=t) for t in (0.02, 0.05, 0.09)])

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(9))
    check = check_nullspace(fim, THETA, atol=1e-6)

    assert check.mass_residual < 1e-6, (
        f"mass should be an exact null direction, residual={check.mass_residual}"
    )
    assert check.inertia_scale_residual < 1e-6, (
        f"inertia scale should be an exact null direction, residual={check.inertia_scale_residual}"
    )
    for axis in range(3):
        residual = float(np.linalg.norm(fim @ com_direction(THETA, axis))) / (np.linalg.norm(fim) + 1e-300)
        assert residual < 1e-6, f"com axis {axis} should be an exact null direction for omega-only tumble"
    # at most 5 of the 6 independent inertia components are ever identifiable
    # from tumbling (ratios only, never absolute scale), on top of mass and
    # all 3 com components being unobservable: at least 5 null directions
    assert check.n_null >= 5


def test_free_flight_tumble_makes_com_observable_unlike_omega_only_tumble():
    """Regime 1 in full (rotation AND translation together, no drag): now
    that a position track is available, c becomes observable (a genuinely
    non-principal-axis tumble breaks the fixed-axis degeneracy that would
    otherwise alias it with the initial position) -- the crisp, well-
    conditioned contrast with the omega-only tumble test above, where every
    com axis was an EXACT null direction. Mass and inertia scale remain
    exactly null either way, since translation with no drag still cannot
    see mass, and rotation still cannot see inertia's overall scale."""
    from ballast.identify.observability import com_direction

    v0 = np.array([1.2, 0.0, 2.5])
    omega0 = np.array([0.3, 4.0, 0.1])  # genuine NPA tumble, not a fixed-axis spin

    def observable(theta):
        return np.concatenate(
            [toss_drag_observable(theta, v0, omega0, drag_coeff=0.0, t_flight_s=t) for t in (0.05, 0.12, 0.2)]
        )

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(9) * 1e-6)
    check = check_nullspace(fim, THETA, atol=1e-6)

    assert check.mass_residual < 1e-6
    assert check.inertia_scale_residual < 1e-6

    for axis in range(3):
        residual = float(np.linalg.norm(fim @ com_direction(THETA, axis))) / (np.linalg.norm(fim) + 1e-300)
        assert residual > 1e-3, (
            f"com axis {axis} should now be clearly observable, unlike the omega-only case"
        )


def test_tumble_observable_is_literally_independent_of_mass_and_com():
    """A direct, non-FIM confirmation: perturbing m or c alone must leave
    the tumble observable EXACTLY unchanged (it is never even referenced
    in the computation), not just approximately."""
    omega0 = np.array([0.3, 4.0, 0.1])
    base = tumble_observable(THETA, omega0, t_eval_s=0.05)

    theta_diff_mass = pack_theta(M_TRUE * 7.3, C_TRUE, I_TRUE)
    theta_diff_com = pack_theta(M_TRUE, C_TRUE + np.array([0.05, -0.03, 0.02]), I_TRUE)

    assert np.array_equal(base, tumble_observable(theta_diff_mass, omega0, t_eval_s=0.05))
    assert np.array_equal(base, tumble_observable(theta_diff_com, omega0, t_eval_s=0.05))


def test_rocking_fim_has_the_combined_homogeneity_null_direction_but_not_each_half():
    """Rocking is a gravity + unknown-contact-force regime: the COMBINED
    (m, 0, I) direction should be null (Theorem 1), but mass and inertia
    scale should NOT be separately null -- rocking clearly depends on each
    of them individually (through the ratio m / I_pivot), just not on
    their overall scale together."""
    pivot = np.array([0.0, 0.0, -0.05])

    def observable(theta):
        return rocking_observable(theta, pivot_point_body=pivot)

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(1) * 1e-4)
    check = check_nullspace(fim, THETA, atol=1e-6)

    assert check.homogeneity_residual < 1e-6, (
        "the combined (m, 0, I) direction should be an exact null direction"
    )
    assert check.mass_residual > 1e-3, "mass alone should NOT be a null direction for rocking"
    assert check.inertia_scale_residual > 1e-3, (
        "inertia scale alone should NOT be a null direction for rocking"
    )


def test_rolling_fim_has_the_combined_homogeneity_null_direction():
    def observable(theta):
        return rolling_observable(theta, roll_axis=2, radius_m=0.03, incline_rad=np.radians(15))

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(1) * 1e-4)
    check = check_nullspace(fim, THETA, atol=1e-6)

    assert check.homogeneity_residual < 1e-6
    assert check.mass_residual > 1e-3
    assert check.inertia_scale_residual > 1e-3


def test_tap_breaks_the_mass_homogeneity_degeneracy():
    """The known-impulse regime is the escape hatch from Theorem 1: with a
    KNOWN applied force, the homogeneity direction should NOT be null."""
    r_app = np.array([0.03, 0.0, 0.02])
    impulse = np.array([0.0, 0.4, 0.05])

    def observable(theta):
        return tap_observable(theta, r_app_body=r_app, impulse_body=impulse)

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(6) * 1e-6)
    check = check_nullspace(fim, THETA, atol=1e-6)

    # homogeneity_residual is the well-scaled, robust check (a single
    # directional projection); a global eigenvalue-based n_null count is
    # NOT asserted here (see check_nullspace's docstring on why it is
    # unreliable across mixed physical units).
    assert check.homogeneity_residual > 1e-3, "the known impulse should break the mass-homogeneity degeneracy"


def test_toss_with_drag_breaks_the_mass_homogeneity_degeneracy():
    """The other escape hatch: aerodynamic drag also breaks the
    mass-homogeneity degeneracy, since drag deceleration depends on m
    while gravity does not enter this differential way."""
    v0 = np.array([1.5, 0.0, 3.0])
    omega0 = np.array([0.3, 4.0, 0.1])

    def observable(theta):
        return toss_drag_observable(theta, v0, omega0, drag_coeff=0.02, t_flight_s=0.2)

    jac = numeric_jacobian(observable, THETA)
    fim = fisher_information(jac, obs_cov=np.eye(3) * 1e-6)
    check = check_nullspace(fim, THETA, atol=1e-6)

    assert check.homogeneity_residual > 1e-3, "drag should break the mass-homogeneity degeneracy"


def test_homogeneity_direction_is_proportional_to_theta_itself_in_mass_and_inertia():
    """homogeneity_direction must be exactly (m, 0, I) using theta's OWN
    values -- the actual infinitesimal generator of the m -> lambda m,
    I -> lambda I scaling group at theta -- not built by combining
    `mass_direction` (a fixed unit mass component, correct for tumble's
    OWN standalone null check) with `inertia_scale_direction` (scaled to
    theta), which would silently produce a direction NOT proportional to
    (m, 0, I) and therefore not actually along the degeneracy curve."""
    homog = homogeneity_direction(THETA)
    assert homog[0] == pytest.approx(M_TRUE)
    assert np.allclose(homog[1:4], 0.0)
    assert np.allclose(homog[4:10], THETA[4:10])
