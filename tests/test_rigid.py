"""Correctness tests for the rigid-body integrator: everything downstream
(pose fitting, the observability atlas, the interior solver) depends on
this being right, not approximately right.

Three independent checks: exact conservation of rotational energy and of
angular momentum in the world frame for torque-free motion (a strong,
frame-aware invariant -- momentum conservation checks both magnitude and
direction, catching bugs energy conservation alone would miss), and
reproduction of the Dzhanibekov (intermediate-axis) instability: a body
spun about its intermediate principal axis periodically flips, while the
same perturbation about the major or minor axis stays bounded.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.physics.rigid import (
    RigidBodyState,
    angular_momentum_world,
    free_flight_force,
    integrate,
    quat_normalize,
    rotational_energy,
)

I_ASYMMETRIC = np.diag([1.0, 2.0, 3.0])  # distinct principal moments: minor, intermediate, major


def _tumble(omega0, i_body=I_ASYMMETRIC, dt=0.001, n_steps=20000):
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=np.zeros(3),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.array(omega0, dtype=np.float64),
    )
    return integrate(state0, m=1.0, i_body=i_body, dt=dt, n_steps=n_steps)


def test_quaternion_stays_normalized_throughout_integration():
    traj = _tumble([0.3, 4.0, 0.1])
    norms = np.linalg.norm(traj.quaternion, axis=1)
    assert np.allclose(norms, 1.0, atol=1e-10)


def test_rotational_energy_is_conserved_for_torque_free_motion():
    traj = _tumble([0.3, 4.0, 0.1])
    energies = np.array([rotational_energy(w, I_ASYMMETRIC) for w in traj.omega_body_rad_s])
    rel_drift = np.abs(energies - energies[0]) / energies[0]
    assert np.max(rel_drift) < 1e-9, (
        f"energy should be conserved to near machine precision, got {rel_drift.max()}"
    )


def test_angular_momentum_is_conserved_in_world_frame():
    """Conserved as a VECTOR (magnitude and direction), which energy
    conservation alone does not guarantee -- this is the stronger check."""
    traj = _tumble([0.3, 4.0, 0.1])
    l_vectors = np.stack(
        [
            angular_momentum_world(q, I_ASYMMETRIC, w)
            for q, w in zip(traj.quaternion, traj.omega_body_rad_s, strict=True)
        ]
    )
    l0 = l_vectors[0]
    mag_drift = np.abs(np.linalg.norm(l_vectors, axis=1) - np.linalg.norm(l0)) / np.linalg.norm(l0)
    assert np.max(mag_drift) < 1e-9

    cos_angle = (l_vectors @ l0) / (np.linalg.norm(l_vectors, axis=1) * np.linalg.norm(l0))
    max_angle_deg = np.degrees(np.arccos(np.clip(cos_angle, -1.0, 1.0))).max()
    # RK4's local truncation error accumulates over 20,000 steps; this is
    # still nine orders of magnitude below any physically meaningful drift
    assert max_angle_deg < 1e-5, f"L direction should not drift, max deviation {max_angle_deg} deg"


@pytest.mark.parametrize(
    "omega0,main_axis,off_axes",
    [
        ([0.05, 5.0, 0.02], 1, (0, 2)),  # spin near the INTERMEDIATE axis (I=2)
    ],
)
def test_intermediate_axis_spin_flips(omega0, main_axis, off_axes):
    """The Dzhanibekov effect: a small perturbation off the intermediate
    principal axis grows until the spin axis itself reverses -- a genuine
    flip, not bounded nutation. Detected two ways: the off-axis components
    grow to a large fraction of the main spin magnitude (not staying near
    the small initial perturbation), and the main component's sign
    reverses at least once."""
    traj = _tumble(omega0, n_steps=40000)
    w = traj.omega_body_rad_s
    main_speed = abs(omega0[main_axis])

    off_axis_peak = max(np.max(np.abs(w[:, ax])) for ax in off_axes)
    assert off_axis_peak > 0.3 * main_speed, (
        "off-axis components should grow to a large fraction of the spin speed during a flip, "
        f"got peak {off_axis_peak} vs main speed {main_speed}"
    )

    sign_changes = np.sum(np.diff(np.sign(w[:, main_axis])) != 0)
    assert sign_changes >= 1, "the main spin component should reverse sign at least once (a flip)"


@pytest.mark.parametrize(
    "omega0,off_axes",
    [
        ([0.05, 0.02, 5.0], (0, 1)),  # spin near the MAJOR axis (I=3)
        ([5.0, 0.05, 0.02], (1, 2)),  # spin near the MINOR axis (I=1)
    ],
)
def test_major_and_minor_axis_spin_is_stable(omega0, off_axes):
    """The complementary check: the same size perturbation about the major
    or minor principal axis stays bounded -- no flip, just small nutation
    -- which is what makes the intermediate-axis case above a genuine
    instability rather than an artifact of the integrator or the test
    tolerance."""
    traj = _tumble(omega0, n_steps=40000)
    w = traj.omega_body_rad_s
    initial_perturbation = max(abs(omega0[ax]) for ax in off_axes)

    for ax in off_axes:
        peak = np.max(np.abs(w[:, ax]))
        assert peak < 5.0 * initial_perturbation, (
            f"off-axis component {ax} should stay bounded near its initial perturbation "
            f"({initial_perturbation}), got peak {peak} -- looks like an unexpected flip"
        )


def test_free_flight_with_gravity_is_a_parabola_in_the_absence_of_drag():
    """Sanity check on the translational half of the integrator: with no
    drag, the center of mass should follow an exact ballistic parabola,
    independent of the (still-tumbling) rotational motion."""
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=np.array([2.0, 0.0, 5.0]),
        quaternion=quat_normalize(np.array([1.0, 0.1, 0.2, 0.0])),
        omega_body_rad_s=np.array([0.3, 4.0, 0.1]),
    )
    m = 1.0
    force_fn = free_flight_force(m, drag_coeff=0.0)
    traj = integrate(state0, m=m, i_body=I_ASYMMETRIC, dt=0.001, n_steps=2000, force_world_fn=force_fn)

    g = 9.80665
    t = traj.t_s
    expected_x = state0.position_m[0] + state0.velocity_m_s[0] * t
    expected_z = state0.position_m[2] + state0.velocity_m_s[2] * t - 0.5 * g * t**2

    assert np.allclose(traj.position_m[:, 0], expected_x, atol=1e-6)
    assert np.allclose(traj.position_m[:, 2], expected_z, atol=1e-6)


def test_drag_causes_measurable_deviation_from_the_no_drag_parabola():
    """This is regime 2's actual informational content (see the project
    plan): drag is one of the few channels that breaks the mass-homogeneity
    degeneracy, so it must produce a real, non-negligible deviation."""
    state0 = RigidBodyState(
        position_m=np.zeros(3),
        velocity_m_s=np.array([5.0, 0.0, 3.0]),
        quaternion=np.array([1.0, 0.0, 0.0, 0.0]),
        omega_body_rad_s=np.array([0.1, 0.1, 0.1]),
    )
    m = 0.2
    traj_no_drag = integrate(
        state0,
        m=m,
        i_body=I_ASYMMETRIC,
        dt=0.001,
        n_steps=1500,
        force_world_fn=free_flight_force(m, drag_coeff=0.0),
    )
    traj_drag = integrate(
        state0,
        m=m,
        i_body=I_ASYMMETRIC,
        dt=0.001,
        n_steps=1500,
        force_world_fn=free_flight_force(m, drag_coeff=0.05),
    )
    final_deviation = np.linalg.norm(traj_drag.position_m[-1] - traj_no_drag.position_m[-1])
    assert final_deviation > 0.01, "drag should produce a clearly measurable deviation over the flight"
