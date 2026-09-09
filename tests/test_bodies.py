"""Tests for the two-layer planetary core-radius / moment-of-inertia
cross-check: the scale-ladder validation, using only published mass,
radius, literature densities, and independently measured spin data."""

from __future__ import annotations

import pytest

from ballast.data.bodies import (
    ALL_BODIES,
    core_radius_from_mass,
    moi_factor_of_two_layer,
    predict_and_cross_check,
)


def test_core_radius_from_mass_round_trips_with_the_forward_model():
    """Round-trip: build a two-layer sphere with a KNOWN core radius,
    compute its mass forward, then recover that same core radius from the
    mass equation alone."""
    radius_m = 6.0e6
    true_core_radius_m = 3.2e6
    core_density, mantle_density = 11500.0, 4400.0

    mass_kg, _moi_factor = moi_factor_of_two_layer(true_core_radius_m, radius_m, core_density, mantle_density)
    recovered_rc = core_radius_from_mass(mass_kg, radius_m, core_density, mantle_density)

    assert recovered_rc == pytest.approx(true_core_radius_m, rel=1e-9)


def test_core_radius_from_mass_rejects_a_denser_mantle_than_core():
    with pytest.raises(ValueError, match="core density must exceed"):
        core_radius_from_mass(1e24, 5e6, core_density_kg_m3=3000.0, mantle_density_kg_m3=4000.0)


def test_larger_core_density_contrast_gives_a_smaller_required_core():
    """A denser assumed core (relative to the mantle) needs a SMALLER core
    radius to account for the same total mass -- a basic monotonicity
    sanity check on the closed-form solution."""
    mass_kg, radius_m, mantle_density = 5e24, 6e6, 4000.0
    rc_dense_core = core_radius_from_mass(mass_kg, radius_m, 13000.0, mantle_density)
    rc_light_core = core_radius_from_mass(mass_kg, radius_m, 8000.0, mantle_density)
    assert rc_dense_core < rc_light_core


@pytest.mark.parametrize("body", ALL_BODIES, ids=[b.name for b in ALL_BODIES])
def test_real_body_core_radius_and_moi_cross_check(body):
    """For each real body: derive a core radius from mass and literature
    densities alone, then confirm (a) it lands within a generous tolerance
    of the independently published (seismic/gravity) core radius, and (b)
    the resulting predicted moment-of-inertia factor is close to the
    INDEPENDENTLY measured spin-derived value -- two different real
    measurements cross-checking one simple model, neither used to derive
    the other."""
    prediction = predict_and_cross_check(body)

    assert 0 < prediction.core_radius_m < body.radius_m

    radius_error = abs(prediction.core_radius_m - body.published_core_radius_m) / body.published_core_radius_m
    assert radius_error < 0.35, (
        f"{body.name}: predicted core radius {prediction.core_radius_m / 1e3:.0f} km vs published "
        f"{body.published_core_radius_m / 1e3:.0f} km, relative error {radius_error:.2f}"
    )

    moi_error = (
        abs(prediction.predicted_moi_factor - prediction.observed_moi_factor) / prediction.observed_moi_factor
    )
    assert moi_error < 0.1, (
        f"{body.name}: predicted moi_factor {prediction.predicted_moi_factor:.4f} vs observed "
        f"{prediction.observed_moi_factor:.4f}, relative error {moi_error:.2f}"
    )


def test_mercury_has_the_largest_core_fraction_of_the_four_bodies():
    """A robust, qualitative cross-body check that survives individual
    per-body tolerance: Mercury's unusually large core (~80% of its
    radius) should come out clearly larger, as a FRACTION of planetary
    radius, than Earth's, Mars's, or the Moon's."""
    fractions = {body.name: predict_and_cross_check(body).core_radius_fraction for body in ALL_BODIES}
    assert fractions["Mercury"] == max(fractions.values())
    assert fractions["Moon"] == min(fractions.values())
