"""Published planetary constants and a two-layer core/mantle inversion,
applied to real solar-system bodies.

This is the scale-ladder validation: the same physics used throughout this
project (a sphere's moment-of-inertia factor, `C/MR^2`, constrains how mass
is distributed radially) also constrains a planet's interior, at roughly
45 orders of magnitude larger scale than a Rattlebox object.

An earlier version of this module tried to invert the core radius directly
from the moment-of-inertia factor at a fixed, assumed mantle density, via
root-finding. That turned out to be the wrong tool: the residual is not
guaranteed monotonic over the full range, so a naive bisection can lock
onto a mathematically valid but physically absurd root (found directly:
for Earth, with a plausible-looking assumed mantle density, it converged
to a core radius of 24% of R at a core density of ~78,000 kg/m^3 -- denser
than any known material -- instead of the true ~55% / ~11,000 kg/m^3).

The robust version used here inverts the LINEAR mass equation instead,
using literature core and mantle densities (mineral-physics estimates, not
derived from this project's own data) to solve for the core radius in
closed form, then cross-checks the result against the INDEPENDENTLY
measured spin-derived moment-of-inertia factor. This is arguably the
better validation anyway: it uses two independent pieces of real
information (mass and spin) to cross-check one model, rather than
inverting a single noisy nonlinear equation.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class PlanetaryBody:
    name: str
    mass_kg: float
    radius_m: float
    moi_factor: float  # C / (M R^2), independently measured from spin/gravity data
    core_density_kg_m3: float  # literature (mineral-physics) estimate, not fit to this project's data
    mantle_density_kg_m3: float  # literature estimate
    published_core_radius_m: float  # for comparison only
    source: str


# Earth: MoI factor is a standard geodetic value. Core/mantle densities and
# core radius are standard PREM-class (Preliminary Reference Earth Model)
# figures; the core-mantle boundary is one of the best-determined
# quantities in geophysics, from seismic body-wave travel times.
EARTH = PlanetaryBody(
    name="Earth",
    mass_kg=5.972e24,
    radius_m=6.371e6,
    moi_factor=0.3307,
    core_density_kg_m3=10800.0,
    mantle_density_kg_m3=4000.0,
    published_core_radius_m=3.480e6,  # ~55% of R
    source="MoI: standard geodetic value. Densities/core radius: PREM-class seismic models.",
)

# Mars: MoI factor from Mars Global Surveyor / MER radio tracking (Konopliv
# et al. 2011). Core is Fe-S alloy (less dense than pure Fe due to sulfur
# content); core radius is the pre-InSight geophysical estimate (~50% of
# R), a more conservative figure than InSight's later, more precise value.
MARS = PlanetaryBody(
    name="Mars",
    mass_kg=6.417e23,
    radius_m=3.3895e6,
    moi_factor=0.3644,
    core_density_kg_m3=6000.0,
    mantle_density_kg_m3=3400.0,
    published_core_radius_m=1.70e6,  # ~50% of R
    source="MoI: Konopliv et al. 2011. Core radius: pre-InSight geophysical estimate.",
)

# Mercury: MoI factor from MESSENGER spin/gravity data (Margot et al.
# 2012). Mercury's core is unusually large (~80% of its radius), driving
# its high bulk density -- the standard explanation invokes a
# collisional-stripping or differential-condensation origin for the thin
# mantle.
MERCURY = PlanetaryBody(
    name="Mercury",
    mass_kg=3.3011e23,
    radius_m=2.4397e6,
    moi_factor=0.346,
    core_density_kg_m3=7200.0,
    mantle_density_kg_m3=3300.0,
    published_core_radius_m=1.998e6,  # ~82% of R
    source="Margot et al. 2012 (MESSENGER spin/gravity data).",
)

# Moon: MoI factor from lunar laser ranging / GRAIL (close to a uniform
# sphere's 0.4, since the Moon's core is small). Core radius from GRAIL
# gravity modeling.
MOON = PlanetaryBody(
    name="Moon",
    mass_kg=7.342e22,
    radius_m=1.7374e6,
    moi_factor=0.393,
    core_density_kg_m3=7500.0,
    mantle_density_kg_m3=3320.0,
    published_core_radius_m=4.03e5,  # ~23% of R
    source="Lunar laser ranging / GRAIL gravity modeling.",
)

ALL_BODIES = [EARTH, MARS, MERCURY, MOON]


def core_radius_from_mass(
    mass_kg: float, radius_m: float, core_density_kg_m3: float, mantle_density_kg_m3: float
) -> float:
    """Closed-form two-layer core radius from the (linear) mass equation:

        M = (4/3) pi [Rc^3 rho_core + (R^3 - Rc^3) rho_mantle]
        => Rc^3 = (M / ((4/3) pi) - R^3 rho_mantle) / (rho_core - rho_mantle)

    Requires `rho_core > rho_mantle` (a denser core) and the body's average
    density to exceed the mantle density (otherwise no non-negative Rc
    solves the equation) -- both true for every body in `ALL_BODIES`.
    """
    if core_density_kg_m3 <= mantle_density_kg_m3:
        raise ValueError("core density must exceed mantle density for a physically sensible two-layer split")
    numerator = mass_kg / ((4.0 / 3.0) * np.pi) - radius_m**3 * mantle_density_kg_m3
    rc_cubed = numerator / (core_density_kg_m3 - mantle_density_kg_m3)
    if rc_cubed < 0:
        raise ValueError("no non-negative core radius solves the mass equation with these densities")
    return float(rc_cubed ** (1.0 / 3.0))


def moi_factor_of_two_layer(
    core_radius_m: float, radius_m: float, core_density_kg_m3: float, mantle_density_kg_m3: float
) -> tuple[float, float]:
    """Forward model: (mass, moi_factor) of a two-layer sphere."""
    mass_kg = (
        (4.0 / 3.0)
        * np.pi
        * (core_radius_m**3 * core_density_kg_m3 + (radius_m**3 - core_radius_m**3) * mantle_density_kg_m3)
    )
    moment_c = (
        (8.0 / 15.0)
        * np.pi
        * (core_radius_m**5 * core_density_kg_m3 + (radius_m**5 - core_radius_m**5) * mantle_density_kg_m3)
    )
    moi_factor = moment_c / (mass_kg * radius_m**2)
    return mass_kg, moi_factor


@dataclass(frozen=True)
class TwoLayerPrediction:
    core_radius_m: float  # from the mass equation alone
    predicted_moi_factor: float  # forward-predicted from that core radius
    observed_moi_factor: float  # the independently measured value, for comparison
    core_radius_fraction: float  # core_radius_m / radius_m


def predict_and_cross_check(body: PlanetaryBody) -> TwoLayerPrediction:
    """The actual validation: derive a core radius from mass and
    literature densities ALONE (no spin data used), then check whether the
    resulting body's predicted moment-of-inertia factor is consistent with
    the INDEPENDENTLY measured (spin/gravity-derived) value -- two
    different real measurements cross-checking one simple model."""
    rc = core_radius_from_mass(
        body.mass_kg, body.radius_m, body.core_density_kg_m3, body.mantle_density_kg_m3
    )
    _mass_check, predicted_moi = moi_factor_of_two_layer(
        rc, body.radius_m, body.core_density_kg_m3, body.mantle_density_kg_m3
    )
    return TwoLayerPrediction(
        core_radius_m=rc,
        predicted_moi_factor=predicted_moi,
        observed_moi_factor=body.moi_factor,
        core_radius_fraction=rc / body.radius_m,
    )
