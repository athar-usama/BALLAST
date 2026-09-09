"""Rigid-body dynamics: the map from the 10 inertial parameters (mass,
center of mass, inertia tensor) to observable motion.

Everything downstream of this module -- pose fitting, the observability
atlas, the interior solver -- depends on this integrator being correct, not
approximately correct. It is validated three independent ways in the test
suite: exact conservation of rotational kinetic energy and of angular
momentum in the world frame for torque-free motion, and reproduction of the
Dzhanibekov (intermediate-axis) instability, which is both a sharp
correctness check and, rendered, a striking figure in its own right.

Convention: quaternions are scalar-first, `q = (w, x, y, z)`, representing
the rotation from body frame to world frame. Angular velocity `omega` is
expressed in the BODY frame throughout, which is the frame in which Euler's
equations take their simple form and in which the inertia tensor `I_body`
is constant.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

GRAVITY_M_S2 = 9.80665


def quat_normalize(q: np.ndarray) -> np.ndarray:
    return q / np.linalg.norm(q)


def quat_multiply(q1: np.ndarray, q2: np.ndarray) -> np.ndarray:
    """Hamilton product, scalar-first convention."""
    w1, x1, y1, z1 = q1
    w2, x2, y2, z2 = q2
    return np.array(
        [
            w1 * w2 - x1 * x2 - y1 * y2 - z1 * z2,
            w1 * x2 + x1 * w2 + y1 * z2 - z1 * y2,
            w1 * y2 - x1 * z2 + y1 * w2 + z1 * x2,
            w1 * z2 + x1 * y2 - y1 * x2 + z1 * w2,
        ]
    )


def quat_to_rotmat(q: np.ndarray) -> np.ndarray:
    """Body-to-world rotation matrix from a scalar-first unit quaternion."""
    w, x, y, z = q
    return np.array(
        [
            [1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
            [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
            [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)],
        ]
    )


def quat_derivative(q: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """dq/dt = 0.5 * q (x) [0, omega_body] (body-frame angular velocity)."""
    omega_quat = np.array([0.0, *omega_body])
    return 0.5 * quat_multiply(q, omega_quat)


def euler_alpha(omega_body: np.ndarray, i_body: np.ndarray, torque_body: np.ndarray) -> np.ndarray:
    """Euler's equations: I @ dw/dt + w x (I @ w) = torque, solved for dw/dt.

    `i_body` is the (3, 3) inertia tensor about the center of mass, expressed
    in the body frame (constant in this frame, which is the entire reason
    body-frame angular velocity is the natural state variable here).
    """
    iw = i_body @ omega_body
    rhs = torque_body - np.cross(omega_body, iw)
    return np.linalg.solve(i_body, rhs)


@dataclass
class RigidBodyState:
    position_m: np.ndarray  # (3,) world-frame position of the center of mass
    velocity_m_s: np.ndarray  # (3,) world-frame linear velocity of the COM
    quaternion: np.ndarray  # (4,) scalar-first, body-to-world rotation
    omega_body_rad_s: np.ndarray  # (3,) angular velocity, expressed in the body frame

    def copy(self) -> RigidBodyState:
        return RigidBodyState(
            position_m=self.position_m.copy(),
            velocity_m_s=self.velocity_m_s.copy(),
            quaternion=self.quaternion.copy(),
            omega_body_rad_s=self.omega_body_rad_s.copy(),
        )

    def rotation_matrix(self) -> np.ndarray:
        return quat_to_rotmat(self.quaternion)


def _state_derivative(
    state: RigidBodyState,
    m: float,
    i_body: np.ndarray,
    force_world_fn,
    torque_body_fn,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    force = force_world_fn(state)
    torque = torque_body_fn(state)
    d_pos = state.velocity_m_s
    d_vel = force / m
    d_quat = quat_derivative(state.quaternion, state.omega_body_rad_s)
    d_omega = euler_alpha(state.omega_body_rad_s, i_body, torque)
    return d_pos, d_vel, d_quat, d_omega


def _add_scaled(state: RigidBodyState, deriv, scale: float) -> RigidBodyState:
    d_pos, d_vel, d_quat, d_omega = deriv
    return RigidBodyState(
        position_m=state.position_m + scale * d_pos,
        velocity_m_s=state.velocity_m_s + scale * d_vel,
        quaternion=state.quaternion + scale * d_quat,
        omega_body_rad_s=state.omega_body_rad_s + scale * d_omega,
    )


def rk4_step(
    state: RigidBodyState,
    m: float,
    i_body: np.ndarray,
    dt: float,
    force_world_fn=None,
    torque_body_fn=None,
) -> RigidBodyState:
    """One classical RK4 step. Forces are expressed in the world frame,
    torques in the body frame (matching `euler_alpha`). Defaults to
    torque-free, force-free motion (a coasting free-flying body)."""
    if force_world_fn is None:
        force_world_fn = lambda s: np.zeros(3)
    if torque_body_fn is None:
        torque_body_fn = lambda s: np.zeros(3)

    k1 = _state_derivative(state, m, i_body, force_world_fn, torque_body_fn)
    s2 = _add_scaled(state, k1, dt / 2)
    s2.quaternion = quat_normalize(s2.quaternion)
    k2 = _state_derivative(s2, m, i_body, force_world_fn, torque_body_fn)
    s3 = _add_scaled(state, k2, dt / 2)
    s3.quaternion = quat_normalize(s3.quaternion)
    k3 = _state_derivative(s3, m, i_body, force_world_fn, torque_body_fn)
    s4 = _add_scaled(state, k3, dt)
    s4.quaternion = quat_normalize(s4.quaternion)
    k4 = _state_derivative(s4, m, i_body, force_world_fn, torque_body_fn)

    new_pos = state.position_m + (dt / 6) * (k1[0] + 2 * k2[0] + 2 * k3[0] + k4[0])
    new_vel = state.velocity_m_s + (dt / 6) * (k1[1] + 2 * k2[1] + 2 * k3[1] + k4[1])
    new_quat = state.quaternion + (dt / 6) * (k1[2] + 2 * k2[2] + 2 * k3[2] + k4[2])
    new_omega = state.omega_body_rad_s + (dt / 6) * (k1[3] + 2 * k2[3] + 2 * k3[3] + k4[3])

    return RigidBodyState(
        position_m=new_pos,
        velocity_m_s=new_vel,
        quaternion=quat_normalize(new_quat),
        omega_body_rad_s=new_omega,
    )


@dataclass
class Trajectory:
    t_s: np.ndarray  # (F,)
    position_m: np.ndarray  # (F, 3)
    velocity_m_s: np.ndarray  # (F, 3)
    quaternion: np.ndarray  # (F, 4)
    omega_body_rad_s: np.ndarray  # (F, 3)

    def rotation_matrices(self) -> np.ndarray:
        return np.stack([quat_to_rotmat(q) for q in self.quaternion])


def integrate(
    state0: RigidBodyState,
    m: float,
    i_body: np.ndarray,
    dt: float,
    n_steps: int,
    force_world_fn=None,
    torque_body_fn=None,
) -> Trajectory:
    """Integrate `n_steps` of RK4 starting from `state0`, returning the full
    trajectory including the initial state (so length is n_steps + 1)."""
    states = [state0]
    s = state0.copy()
    for _ in range(n_steps):
        s = rk4_step(s, m, i_body, dt, force_world_fn, torque_body_fn)
        states.append(s)

    t = np.arange(n_steps + 1) * dt
    return Trajectory(
        t_s=t,
        position_m=np.stack([s.position_m for s in states]),
        velocity_m_s=np.stack([s.velocity_m_s for s in states]),
        quaternion=np.stack([s.quaternion for s in states]),
        omega_body_rad_s=np.stack([s.omega_body_rad_s for s in states]),
    )


def rotational_energy(omega_body: np.ndarray, i_body: np.ndarray) -> float:
    """T = 0.5 * omega^T I omega, conserved exactly for torque-free motion."""
    return float(0.5 * omega_body @ i_body @ omega_body)


def angular_momentum_world(quaternion: np.ndarray, i_body: np.ndarray, omega_body: np.ndarray) -> np.ndarray:
    """L = R (I omega), a constant VECTOR in the world frame for torque-free
    motion (energy conservation alone only bounds |omega|, not its
    direction; this is the stronger, vector-valued invariant)."""
    r = quat_to_rotmat(quaternion)
    return r @ (i_body @ omega_body)


def free_flight_force(m: float, gravity: float = GRAVITY_M_S2, drag_coeff: float = 0.0):
    """A ballistic force: gravity plus optional quadratic air drag
    (`drag_coeff` has units such that `F_drag = -drag_coeff * |v| * v`,
    already absorbing 0.5 * rho_air * C_d * A)."""

    def force(state: RigidBodyState) -> np.ndarray:
        g_force = np.array([0.0, 0.0, -m * gravity])
        v = state.velocity_m_s
        speed = np.linalg.norm(v)
        drag_force = -drag_coeff * speed * v if speed > 0 else np.zeros(3)
        return g_force + drag_force

    return force
