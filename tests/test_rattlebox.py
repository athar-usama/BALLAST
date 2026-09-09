"""Tests for the Rattlebox generator: materials, shells, fills, and the
sampler that ties them together with ground-truth moments."""

from __future__ import annotations

import numpy as np
import pytest

from ballast.moments.operator import build_moment_operator, moments_to_inertia
from ballast.rattlebox.fills import fill_layered, fill_lining, fill_slug, fill_twin_lumps, fill_void
from ballast.rattlebox.materials import MATERIALS, classify_by_attenuation, density_for
from ballast.rattlebox.sampler import sample_dataset, sample_object
from ballast.rattlebox.shells import SHELL_FACTORIES, make_box_shell, make_sphere_shell, make_tin_shell

RNG = np.random.default_rng(0)


# --------------------------------------------------------------------------
# materials
# --------------------------------------------------------------------------


def test_density_for_known_materials():
    assert density_for("steel") == pytest.approx(7850.0)
    assert density_for("air") < density_for("water") < density_for("steel") < density_for("lead")


def test_density_for_unknown_material_raises():
    with pytest.raises(KeyError):
        density_for("unobtainium")


def test_classify_by_attenuation_prefers_the_correct_material():
    """The corrected recalibration logic (T5's XDen-1K fix): given the LAC
    a real steel part would produce, classification among plausible
    candidates should pick steel, not water or aluminium."""
    true_material = "steel"
    rho_g_cm3 = MATERIALS[true_material] / 1000.0
    mu_over_rho = 1.205  # steel at ~60 keV effective
    true_lac = rho_g_cm3 * mu_over_rho

    guess = classify_by_attenuation(true_lac, ["water", "aluminium", "steel", "lead"], effective_kev=60)
    assert guess == "steel"


# --------------------------------------------------------------------------
# shells
# --------------------------------------------------------------------------


@pytest.mark.parametrize("factory", [make_box_shell, make_tin_shell, make_sphere_shell])
def test_shell_wall_and_cavity_are_disjoint_and_nonempty(factory):
    shell = factory(n=20)
    assert shell.wall_mask.sum() > 0
    assert shell.cavity_mask.sum() > 0
    assert not np.any(shell.wall_mask & shell.cavity_mask)


@pytest.mark.parametrize("factory", [make_box_shell, make_tin_shell, make_sphere_shell])
def test_cavity_is_strictly_interior_to_the_wall(factory):
    """Every cavity voxel should be surrounded by wall or more cavity, not
    exposed directly to the exterior -- otherwise the object is not
    actually sealed."""
    shell = factory(n=20)
    occupied = shell.wall_mask | shell.cavity_mask
    occ_grid = shell.grid.reshape(occupied)
    cavity_grid = shell.grid.reshape(shell.cavity_mask)

    # dilate the cavity by one voxel and check it stays within occupied space
    # (i.e. the cavity does not touch the unoccupied exterior)
    from scipy.ndimage import binary_dilation

    dilated = binary_dilation(cavity_grid, iterations=1)
    exterior_touch = dilated & ~occ_grid
    assert exterior_touch.sum() == 0, "cavity touches the exterior: shell is not sealed"


def test_all_shell_kinds_are_registered():
    assert set(SHELL_FACTORIES) == {"box", "tin", "sphere"}


# --------------------------------------------------------------------------
# fills
# --------------------------------------------------------------------------


def test_fill_void_is_air_only():
    shell = make_box_shell(n=16)
    rho = fill_void(shell.grid, shell.cavity_mask)
    assert np.all(rho[shell.cavity_mask] == pytest.approx(density_for("air")))
    assert np.all(rho[~shell.cavity_mask] == 0.0)


def test_fill_slug_places_the_right_approximate_mass():
    shell = make_box_shell(n=24)
    fraction = 0.15
    rho = fill_slug(shell.grid, shell.cavity_mask, RNG, material="steel", fraction=fraction)

    cavity_volume = shell.cavity_mask.sum() * shell.grid.voxel_volume_m3
    expected_mass = fraction * cavity_volume * density_for("steel") + (
        1 - fraction
    ) * cavity_volume * density_for("air")
    actual_mass = rho[shell.cavity_mask].sum() * shell.grid.voxel_volume_m3
    assert actual_mass == pytest.approx(expected_mass, rel=0.35)  # voxel quantization at this resolution
    assert np.any(rho == density_for("steel"))


def test_fill_twin_lumps_places_separated_dense_regions():
    shell = make_sphere_shell(n=28, radius_m=0.08)
    rho = fill_twin_lumps(shell.grid, shell.cavity_mask, RNG, material="steel", n_lumps=2, fraction_each=0.05)

    dense_mask = rho >= density_for("steel") * 0.99
    assert dense_mask.sum() > 0

    from scipy.ndimage import label

    dense_grid = shell.grid.reshape(dense_mask)
    _labeled, n_components = label(dense_grid)
    assert n_components >= 2, "twin lumps should form at least 2 separated dense connected components"


def test_fill_layered_orders_materials_along_the_axis():
    shell = make_tin_shell(n=24)
    materials = ("sand", "water", "air")
    rho = fill_layered(shell.grid, shell.cavity_mask, RNG, materials=materials, axis=2)

    centers = shell.grid.centers()
    z = centers[:, 2]
    cavity = shell.cavity_mask
    # the bottom-most cavity voxels should be the first material (sand, densest here)
    z_cavity = z[cavity]
    rho_cavity = rho[cavity]
    bottom_10pct = z_cavity <= np.percentile(z_cavity, 10)
    top_10pct = z_cavity >= np.percentile(z_cavity, 90)
    assert rho_cavity[bottom_10pct].mean() > rho_cavity[top_10pct].mean()


def test_fill_lining_has_dense_boundary_and_empty_core():
    shell = make_sphere_shell(n=28, radius_m=0.08)
    rho = fill_lining(shell.grid, shell.cavity_mask, RNG, material="lead", thickness_voxels=2)

    centers = shell.grid.centers()
    cavity_center = centers[shell.cavity_mask].mean(axis=0)
    dist_from_center = np.linalg.norm(centers - cavity_center[None, :], axis=1)

    cavity_dist = dist_from_center[shell.cavity_mask]
    cavity_rho = rho[shell.cavity_mask]
    near_core = cavity_dist <= np.percentile(cavity_dist, 20)
    near_wall = cavity_dist >= np.percentile(cavity_dist, 80)

    assert cavity_rho[near_core].mean() < cavity_rho[near_wall].mean()
    assert np.any(rho == density_for("lead"))


def test_lining_has_larger_moment_of_inertia_than_a_central_slug_of_equal_mass():
    """A physically intuitive, qualitative sanity check that also
    foreshadows 'the thermos problem': mass pushed to the outside raises
    the moment of inertia relative to the same mass concentrated at the
    center, for otherwise identical shells."""
    shell_a = make_sphere_shell(n=28, radius_m=0.08)
    shell_b = make_sphere_shell(n=28, radius_m=0.08)

    rho_lining = fill_lining(shell_a.grid, shell_a.cavity_mask, RNG, material="lead", thickness_voxels=2)
    rho_slug = fill_slug(shell_b.grid, shell_b.cavity_mask, RNG, material="lead", fraction=0.3)

    a_op = build_moment_operator(shell_a.grid)
    m_lining, _c_lining, i_lining = moments_to_inertia(a_op @ rho_lining)
    m_slug, _c_slug, i_slug = moments_to_inertia(a_op @ rho_slug)

    # compare per-unit-mass moment of inertia (trace) since the two fills
    # do not have exactly equal total mass at this voxel resolution
    assert (np.trace(i_lining) / m_lining) > (np.trace(i_slug) / m_slug)


# --------------------------------------------------------------------------
# sampler
# --------------------------------------------------------------------------


@pytest.mark.parametrize("shell_kind", ["box", "tin", "sphere"])
@pytest.mark.parametrize("fill_kind", ["void", "slug", "twin_lumps", "layered", "lining"])
def test_sample_object_produces_a_valid_object_for_every_combination(shell_kind, fill_kind):
    rng = np.random.default_rng(1)
    obj = sample_object(rng, shell_kind=shell_kind, fill_kind=fill_kind, n_voxels_per_axis=16)
    assert obj.mass_kg > 0
    assert np.all(np.isfinite(obj.com_body_m))
    eigvals = np.linalg.eigvalsh(obj.inertia_body_kgm2)
    assert np.all(eigvals > 0), "inertia tensor should be positive definite for a physical object"


def test_sample_dataset_is_reproducible_with_a_fixed_seed():
    ds1 = sample_dataset(5, seed=42, n_voxels_per_axis=14)
    ds2 = sample_dataset(5, seed=42, n_voxels_per_axis=14)
    for a, b in zip(ds1, ds2, strict=True):
        assert a.shell_kind == b.shell_kind
        assert a.fill_kind == b.fill_kind
        assert a.mass_kg == pytest.approx(b.mass_kg)
        assert np.allclose(a.com_body_m, b.com_body_m)


def test_sample_dataset_produces_variety():
    ds = sample_dataset(30, seed=7, n_voxels_per_axis=14)
    shell_kinds = {obj.shell_kind for obj in ds}
    fill_kinds = {obj.fill_kind for obj in ds}
    assert len(shell_kinds) > 1, "30 random draws should hit more than one shell kind"
    assert len(fill_kinds) > 1, "30 random draws should hit more than one fill kind"
