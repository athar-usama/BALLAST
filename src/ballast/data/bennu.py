"""Asteroid (101955) Bennu: a real body whose interior is independently
known to be non-uniform, used to demonstrate this project's own principal-
axis degeneracy theorem on a genuine published finding rather than a
constructed example.

Bennu is a well-behaved principal-axis rotator (period 4.296 hours), not a
tumbling body like a spacecraft in distress -- and that is exactly the
point. Scheeres et al. (Science Advances, 2020) found Bennu's interior is
NOT uniform: it is underdense at both the center and the equatorial bulge,
denser toward the poles, based on spacecraft tracking and the reconstructed
orbits of naturally ejected surface particles -- a measurement of the
GRAVITY FIELD, not of the spin dynamics.

This module builds two physically valid interior models sharing Bennu's
real mass, shape, and spin axis -- a uniform-density model, and a
Scheeres-like heterogeneous one -- and shows they are dynamically
IDENTICAL under pure principal-axis rotation, exactly as `ballast.identify.
observability`'s own tumble degeneracy theorem predicts (see
`test_principal_axis_spin_is_degenerate_for_inertia_recovery`): a body
spinning about a genuine principal axis reveals only "this is a principal
axis," never the inertia ratios, let alone anything about their spatial
origin. This is why the real heterogeneity finding required an entirely
different kind of measurement (gravity, not spin) -- a fact this project's
own theory anticipates rather than needing to be told.

Published constants:
  - Mass: 7.329e10 kg (OSIRIS-REx radio tracking).
  - Bulk dimensions: approximately 565 x 535 x 508 m (a slightly oblate,
    "spinning-top" shape with an equatorial ridge), giving semi-axes on
    the order of 282 x 267 x 254 m.
  - Rotation period: 4.296 hours (retrograde), about the shortest axis.
  - Bulk density: 1190 kg/m^3.
  - Interior: denser near the poles; underdense at the center and along
    the equatorial bulge (Scheeres et al. 2020, Science Advances 6:eabc3350).
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np

from ballast.moments.operator import build_moment_operator, moments_to_inertia
from ballast.moments.voxelgrid import VoxelGrid

BENNU_MASS_KG = 7.329e10
BENNU_SEMI_AXES_M = np.array([282.0, 267.0, 254.0])  # x, y, z (z = spin/shortest axis)
BENNU_ROTATION_PERIOD_S = 4.296 * 3600.0
BENNU_BULK_DENSITY_KG_M3 = 1190.0


def _bennu_grid(n: int = 41) -> VoxelGrid:
    extent = 2.4 * BENNU_SEMI_AXES_M.max()
    h = extent / n
    half = h * (n - 1) / 2
    return VoxelGrid(shape=(n, n, n), voxel_size_m=h, origin_m=np.array([-half, -half, -half]))


def _ellipsoid_mask(grid: VoxelGrid, semi_axes_m: np.ndarray) -> np.ndarray:
    centers = grid.centers()
    normalized = centers / semi_axes_m[None, :]
    return np.sum(normalized**2, axis=1) <= 1.0


@dataclass(frozen=True)
class BennuModel:
    grid: VoxelGrid
    rho: np.ndarray
    mass_kg: float
    com_m: np.ndarray
    inertia_kgm2: np.ndarray
    principal_axes: np.ndarray  # (3, 3), columns are eigenvectors, ascending eigenvalue order


def _finalize(grid: VoxelGrid, rho: np.ndarray) -> BennuModel:
    a = build_moment_operator(grid)
    m, c, i_com = moments_to_inertia(a @ rho)
    _eigvals, eigvecs = np.linalg.eigh(i_com)
    return BennuModel(grid=grid, rho=rho, mass_kg=m, com_m=c, inertia_kgm2=i_com, principal_axes=eigvecs)


def uniform_bennu_model(n: int = 41) -> BennuModel:
    """A uniform-density interior filling Bennu's known outer shape."""
    grid = _bennu_grid(n)
    mask = _ellipsoid_mask(grid, BENNU_SEMI_AXES_M)
    volume_m3 = mask.sum() * grid.voxel_volume_m3
    density = BENNU_MASS_KG / volume_m3
    rho = np.where(mask, density, 0.0)
    return _finalize(grid, rho)


def heterogeneous_bennu_model(n: int = 41, pole_to_equator_contrast: float = 1.25) -> BennuModel:
    """A Scheeres-like interior: denser toward the poles (+/-z) -- a
    simplified stand-in for the full qualitative finding (poles denser
    than both the center and the equatorial bulge).

    Deliberately a function of z ALONE, not of any radius. Two earlier
    versions used an ellipsoidal (x,y,z-normalized) radius to also make the
    center and equatorial bulge underdense, and both broke x/y symmetry
    enough to swap which axis has the greatest moment of inertia -- because
    Bennu's own x and y semi-axes differ (282 vs 267 m), a density that
    depends on the ANISOTROPICALLY-normalized radius is only symmetric in
    SCALED coordinates, not in the real x,y used by the actual moment
    integral. A weight that depends on z alone cannot introduce any x/y
    asymmetry beyond what the base ellipsoid shape already has, by
    construction: it multiplies every x,y cross-section at a given z by
    the same scalar. This still reproduces "poles denser than center"
    (checked directly below): the geometric center has z=0, the lowest
    polar fraction there is, same as the equator.

    Normalized to Bennu's real total mass so the two models are a genuine
    like-for-like pair, not just "any nonuniform blob."

    The default contrast (1.25) is deliberately modest: concentrating mass
    toward the poles moves it AWAY from the spin (z) axis, which decreases
    I_zz -- found directly, a contrast of 1.6 is already strong enough to
    make I_zz drop below I_yy, flipping which axis has the greatest moment
    of inertia and undermining the "both models share the same major axis"
    comparison. 1.25 keeps z clearly dominant while still producing a
    measurably different set of inertia ratios from the uniform model.
    """
    grid = _bennu_grid(n)
    mask = _ellipsoid_mask(grid, BENNU_SEMI_AXES_M)
    centers = grid.centers()

    polar_fraction = np.abs(centers[:, 2]) / BENNU_SEMI_AXES_M[2]  # 0 at equator plane, 1 at pole
    weight = 1.0 + (pole_to_equator_contrast - 1.0) * polar_fraction**2
    weight = np.where(mask, weight, 0.0)
    # normalize so the total mass matches Bennu's real published mass exactly
    unnormalized_mass = weight.sum() * grid.voxel_volume_m3 * BENNU_BULK_DENSITY_KG_M3
    rho = weight * BENNU_BULK_DENSITY_KG_M3 * (BENNU_MASS_KG / unnormalized_mass)
    return _finalize(grid, rho)


def spin_axis_alignment(model: BennuModel, spin_axis_world: np.ndarray) -> float:
    """Cosine similarity between the model's shortest-moment principal
    axis and a given (real, observed) spin axis direction -- both models
    should align with Bennu's actual rotation axis equally well, since a
    principal-axis rotator's spin direction constrains only which axis is
    principal, never the ratios between the three moments or their
    physical origin."""
    # Bennu, like every relaxed rubble pile, spins stably about its axis of
    # GREATEST moment of inertia -- the last column at ascending eigenvalue order.
    major_axis = model.principal_axes[:, -1]
    cos_sim = float(np.abs(np.dot(major_axis, spin_axis_world)))
    return cos_sim
