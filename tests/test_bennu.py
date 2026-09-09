"""Tests for the Bennu real-anchor demonstration: two physically valid
interior models sharing Bennu's real mass, shape, and spin axis, one
uniform and one matching Scheeres et al.'s real heterogeneity finding,
shown to be dynamically indistinguishable under pure principal-axis
rotation -- the project's own degeneracy theorem, applied to a real,
independently published result rather than a constructed example.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.data.bennu import (
    BENNU_MASS_KG,
    heterogeneous_bennu_model,
    spin_axis_alignment,
    uniform_bennu_model,
)

Z_AXIS_WORLD = np.array([0.0, 0.0, 1.0])


def test_both_models_match_bennus_real_published_mass():
    uniform = uniform_bennu_model(n=33)
    hetero = heterogeneous_bennu_model(n=33)
    assert uniform.mass_kg == pytest.approx(BENNU_MASS_KG, rel=0.02)
    assert hetero.mass_kg == pytest.approx(BENNU_MASS_KG, rel=1e-6)


def test_the_two_models_have_genuinely_different_interiors():
    """The heterogeneous model must be a REAL structural difference, not a
    trivial rescaling of the uniform one: different inertia ratios, not
    just a different overall scale (which Theorem 1 already says is
    unobservable for other reasons)."""
    uniform = uniform_bennu_model(n=33)
    hetero = heterogeneous_bennu_model(n=33)

    uniform_ratios = np.sort(np.linalg.eigvalsh(uniform.inertia_kgm2))
    hetero_ratios = np.sort(np.linalg.eigvalsh(hetero.inertia_kgm2))
    uniform_ratios /= uniform_ratios[-1]
    hetero_ratios /= hetero_ratios[-1]

    assert not np.allclose(uniform_ratios, hetero_ratios, atol=0.01), (
        "the heterogeneous model should have genuinely different inertia ratios, "
        f"got uniform={uniform_ratios}, hetero={hetero_ratios}"
    )


def test_both_models_are_oblate_spinning_about_the_axis_of_greatest_inertia():
    """Sanity check on the shape itself: Bennu's flattened (z shortest)
    profile should make z the axis of GREATEST moment of inertia for both
    models -- matching the real, physically stable major-axis spin of an
    actual relaxed rubble pile."""
    for model in (uniform_bennu_model(n=33), heterogeneous_bennu_model(n=33)):
        major_axis = model.principal_axes[:, -1]
        assert np.abs(np.dot(major_axis, Z_AXIS_WORLD)) > 0.99


def test_both_models_are_equally_consistent_with_the_observed_spin_axis():
    """The actual demonstration: despite having GENUINELY DIFFERENT
    interior mass distributions (confirmed above by their different
    inertia ratios), both models align with Bennu's real observed spin
    axis equally well -- principal-axis rotation alone cannot distinguish
    them, exactly as this project's own observability theorem predicts.
    This is why Scheeres et al.'s real heterogeneity finding needed gravity
    tracking data, not spin dynamics."""
    uniform = uniform_bennu_model(n=33)
    hetero = heterogeneous_bennu_model(n=33)

    uniform_alignment = spin_axis_alignment(uniform, Z_AXIS_WORLD)
    hetero_alignment = spin_axis_alignment(hetero, Z_AXIS_WORLD)

    assert uniform_alignment > 0.99
    assert hetero_alignment > 0.99
    assert abs(uniform_alignment - hetero_alignment) < 0.01, (
        "both interiors should be equally consistent with the observed spin axis, "
        f"got uniform={uniform_alignment}, hetero={hetero_alignment}"
    )


def test_heterogeneous_model_is_denser_toward_the_poles_than_center():
    """A direct, qualitative confirmation that the constructed heterogeneous
    model actually reproduces Scheeres et al.'s finding (denser poles,
    underdense center), not just an arbitrary different mass distribution."""
    model = heterogeneous_bennu_model(n=41)
    centers = model.grid.centers()
    from ballast.data.bennu import BENNU_SEMI_AXES_M

    polar_fraction = np.abs(centers[:, 2]) / BENNU_SEMI_AXES_M[2]
    center_radius = np.linalg.norm(centers / BENNU_SEMI_AXES_M[None, :], axis=1)

    near_pole = (polar_fraction > 0.7) & (center_radius < 0.9) & (model.rho > 0)
    near_core = (center_radius < 0.1) & (model.rho > 0)

    assert model.rho[near_pole].mean() > model.rho[near_core].mean()
