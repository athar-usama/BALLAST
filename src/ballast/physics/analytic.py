"""Closed-form, per-regime models for the motions that are cheap to measure
precisely and that do not need the general contact simulator.

Each of these is deliberately NOT fit by differentiating through a contact
simulation (see the project plan's rejection of that approach: gradients
through a multi-impact settle blow up and the loss landscape is a
staircase). Instead each regime reduces to a small closed-form relationship
that a simple, well-conditioned fit (an FFT peak, a straight-line
acceleration, a ratio of two vectors) recovers directly:

  - rocking on a plane: a physical-pendulum frequency, `omega0^2 = m g d /
    I_pivot`. This measures the ratio `m/I_pivot`, not m and I separately,
    exactly as the mass-homogeneity theorem requires (gravity + contact,
    unknown contact force -> homogeneous of degree 1 in mass).
  - rolling down an incline: `a = g sin(theta) / (1 + I_axis/(m R^2))`,
    which measures the dimensionless moment-of-inertia factor `I_axis / (m
    R^2)` -- the same quantity used throughout planetary interior
    inversion (`C/MR^2`), which is what lets the planetary validation later
    in this project reuse this exact function.
  - critical tip angle: purely quasi-static, depends only on where the
    center of mass sits relative to the support polygon, not on any
    contact parameter (friction, restitution) or on chaos. The cleanest
    possible measurement: a real object, a protractor, no electronics.
  - a known tap impulse: the ONE channel (along with drag) that breaks the
    mass-homogeneity degeneracy, because the applied force is known rather
    than an unknown reaction. `mass_from_impulse` recovers absolute mass
    directly; `inertia_scale_from_impulse` recovers the absolute SCALE of
    an inertia tensor whose direction is already known (e.g. from a prior
    torque-free tumble fit, which only ever gives scale-free ratios).
"""

from __future__ import annotations

import numpy as np

GRAVITY_M_S2 = 9.80665


# --------------------------------------------------------------------------
# Rocking on a plane: physical-pendulum frequency measures m / I_pivot.
# --------------------------------------------------------------------------


def rocking_frequency_forward(m: float, d_m: float, i_pivot: float, g: float = GRAVITY_M_S2) -> float:
    """Forward model: small-oscillation angular frequency (rad/s) of an
    object rocking about a fixed contact edge, with its center of mass a
    distance `d_m` from that edge and moment of inertia `i_pivot` about it."""
    return float(np.sqrt(m * g * d_m / i_pivot))


def rocking_mass_over_inertia(omega0_rad_s: float, d_m: float, g: float = GRAVITY_M_S2) -> float:
    """Inverse model: from a measured rocking frequency, recover m /
    I_pivot -- the only combination rocking alone can identify."""
    return float(omega0_rad_s**2 / (g * d_m))


def estimate_frequency_fft(signal: np.ndarray, dt_s: float) -> float:
    """Dominant angular frequency (rad/s) of a real-valued oscillation
    (e.g. a tracked tilt-angle time series), via the FFT peak with
    quadratic (parabolic) sub-bin interpolation. This is the practical
    route to `omega0` from a real (or simulated) rocking clip: no model
    fitting, just a peak pick -- but two effects otherwise leave several
    percent of error on a short (few-second) clip: the raw discrete peak is
    quantized to the FFT's bin spacing `1/duration`, and a non-integer
    number of cycles in the window causes spectral leakage under the
    implicit rectangular window, which biases naive parabolic interpolation
    of the raw spectrum by several percent even after correcting for bin
    quantization. A Hann window suppresses the leakage (its sidelobes decay
    far faster than a rectangular window's), and quadratic interpolation of
    the three bins around the windowed spectrum's peak then recovers
    sub-bin accuracy cleanly: at 3 seconds of clean tilt data this combination
    is already under 1.5% error, and it is under 0.1% by 8 seconds -- the
    ~1% accuracy the project's regime atlas quotes for rocking.
    """
    n = signal.shape[0]
    window = np.hanning(n)
    spectrum = np.abs(np.fft.rfft((signal - np.mean(signal)) * window))
    freqs_hz = np.fft.rfftfreq(n, d=dt_s)
    if len(spectrum) < 4:
        raise ValueError("signal too short to resolve a frequency")

    peak_idx = 1 + int(np.argmax(spectrum[1:]))  # skip the DC bin at index 0
    if peak_idx <= 0 or peak_idx >= len(spectrum) - 1:
        return float(2 * np.pi * freqs_hz[peak_idx])  # peak at an edge bin: no room to interpolate

    alpha, beta, gamma = spectrum[peak_idx - 1], spectrum[peak_idx], spectrum[peak_idx + 1]
    denom = alpha - 2 * beta + gamma
    delta = 0.5 * (alpha - gamma) / denom if abs(denom) > 1e-300 else 0.0
    bin_width_hz = freqs_hz[1] - freqs_hz[0]
    interpolated_hz = freqs_hz[peak_idx] + delta * bin_width_hz
    return float(2 * np.pi * interpolated_hz)


# --------------------------------------------------------------------------
# Rolling down an incline: acceleration measures I_axis / (m R^2).
# --------------------------------------------------------------------------


def rolling_acceleration_forward(moi_factor: float, incline_rad: float, g: float = GRAVITY_M_S2) -> float:
    """Forward model: linear acceleration of a symmetric roller's center of
    mass down a frictionless-slip incline, given the dimensionless
    moment-of-inertia factor `moi_factor = I_axis / (m R^2)` about the
    rolling axis (0 for a point mass, 2/5 for a uniform sphere, 1 for a
    thin hoop -- the same quantity as a planetary `C/MR^2`)."""
    return float(g * np.sin(incline_rad) / (1.0 + moi_factor))


def rolling_moi_factor_from_acceleration(a_m_s2: float, incline_rad: float, g: float = GRAVITY_M_S2) -> float:
    """Inverse model: from a measured downhill acceleration, recover the
    moment-of-inertia factor `I_axis / (m R^2)` -- again a ratio, not an
    absolute inertia, consistent with the mass-homogeneity theorem."""
    return float(g * np.sin(incline_rad) / a_m_s2 - 1.0)


# --------------------------------------------------------------------------
# Critical tip angle: purely quasi-static, depends only on geometry.
# --------------------------------------------------------------------------


def critical_tip_angle(horizontal_offset_m: float, height_above_pivot_m: float) -> float:
    """The tilt angle (rad) at which an object balanced on a pivot edge
    becomes unstable and tips: the angle at which the center of mass's
    vertical projection passes over the pivot edge.

    `horizontal_offset_m` is the horizontal distance, measured in the
    object's own upright frame, from the pivot edge to the point directly
    below the center of mass; `height_above_pivot_m` is the center of
    mass's height above that edge. No contact parameter (friction,
    restitution) enters at all -- this is why it is the cleanest possible
    real-object measurement in the whole regime set.
    """
    return float(np.arctan2(horizontal_offset_m, height_above_pivot_m))


# --------------------------------------------------------------------------
# A known tap impulse: the channel that breaks the mass-homogeneity
# degeneracy, giving absolute mass and absolute inertia scale.
# --------------------------------------------------------------------------


def mass_from_impulse(delta_v_m_s: np.ndarray, impulse_ns: np.ndarray) -> float:
    """m = |J| / |delta_v|, from the impulse-momentum theorem m * delta_v =
    J. Uses a least-squares projection (J . delta_v / delta_v . delta_v)
    rather than a single component, so noisy or near-degenerate directions
    average out rather than picking an arbitrary axis."""
    dv = np.asarray(delta_v_m_s, dtype=np.float64)
    j = np.asarray(impulse_ns, dtype=np.float64)
    denom = dv @ dv
    if denom <= 0:
        raise ValueError("delta_v must be nonzero to recover mass from an impulse")
    return float((j @ dv) / denom)


def inertia_scale_from_impulse(
    i_direction: np.ndarray,
    r_eff_body_m: np.ndarray,
    impulse_body_ns: np.ndarray,
    delta_omega_body_rad_s: np.ndarray,
) -> float:
    """Recover the absolute SCALE `lambda` of an inertia tensor whose
    direction is already known (`i_direction`, e.g. the scale-free result
    of `inertia_from_angular_momentum`), from one tap: `lambda * I_direction
    @ delta_omega ~= r_eff x J`. Solved by least squares over the scalar
    `lambda`, exactly the channel a torque-free tumble alone can never
    supply (Theorem 1: only ratios and the principal-axis frame are
    observable without a known applied force or torque)."""
    torque_impulse = np.cross(r_eff_body_m, impulse_body_ns)
    predicted_direction = i_direction @ delta_omega_body_rad_s
    denom = predicted_direction @ predicted_direction
    if denom <= 0:
        raise ValueError("delta_omega must be nonzero to recover an inertia scale from an impulse")
    return float((torque_impulse @ predicted_direction) / denom)
