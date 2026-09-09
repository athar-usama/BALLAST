"""Interior fill patterns: what actually goes inside a shell's hollow
cavity. Each function returns a full (V,) density array (kg/m^3), zero
outside the cavity, so the caller (`sampler.sample_object`) only has to add
the shell wall's own density on top.

The patterns are chosen to make the project's theorems concrete objects
rather than abstract claims:

  - `fill_void`: a genuinely empty container -- and, point-reflected
    through its own center, a bit-identical twin of itself, the trivial
    but real case of the chirality theorem.
  - `fill_slug`: a single compact dense inclusion -- exactly the "one blob"
    case T2 says is generically exactly identified.
  - `fill_twin_lumps`: two separated dense inclusions -- exactly the case
    T2 says admits a continuous ambiguous family, the constructive
    negative half of the same theorem.
  - `fill_layered`: horizontal bands of different materials -- the regime
    where "a few discrete materials" as a prior does real, checkable work.
  - `fill_lining`: dense material coating the cavity WALL with an empty
    core -- visually and physically the sharpest possible contrast with a
    slug of the same mass in the middle, and the two are exactly the kind
    of pair the sharp-bound theorem (T1) says can be far apart in COM and
    inertia while looking identical from outside.
"""

from __future__ import annotations

import numpy as np
from scipy.ndimage import binary_erosion

from ballast.moments.voxelgrid import VoxelGrid
from ballast.rattlebox.materials import density_for


def _empty(grid: VoxelGrid) -> np.ndarray:
    return np.zeros(grid.n_voxels, dtype=np.float64)


def fill_void(grid: VoxelGrid, cavity_mask: np.ndarray, rng: np.random.Generator | None = None) -> np.ndarray:
    """An entirely hollow cavity: air only."""
    rho = _empty(grid)
    rho[cavity_mask] = density_for("air")
    return rho


def fill_slug(
    grid: VoxelGrid,
    cavity_mask: np.ndarray,
    rng: np.random.Generator,
    material: str = "steel",
    fraction: float = 0.15,
) -> np.ndarray:
    """A single compact spherical inclusion of `material`, sized to occupy
    roughly `fraction` of the cavity's volume, placed at a random location
    that keeps the whole sphere within the cavity."""
    rho = _empty(grid)
    rho[cavity_mask] = density_for("air")

    centers = grid.centers()
    cavity_centers = centers[cavity_mask]
    cavity_volume = cavity_mask.sum() * grid.voxel_volume_m3
    slug_volume = fraction * cavity_volume
    slug_radius = (3.0 * slug_volume / (4.0 * np.pi)) ** (1.0 / 3.0)

    bounds_min = cavity_centers.min(axis=0) + slug_radius
    bounds_max = cavity_centers.max(axis=0) - slug_radius
    if np.any(bounds_max < bounds_min):
        center = cavity_centers.mean(axis=0)  # cavity too small for the requested fraction: center it
    else:
        center = rng.uniform(bounds_min, bounds_max)

    dist = np.linalg.norm(centers - center[None, :], axis=1)
    slug_mask = (dist <= slug_radius) & cavity_mask
    rho[slug_mask] = density_for(material)
    return rho


def fill_twin_lumps(
    grid: VoxelGrid,
    cavity_mask: np.ndarray,
    rng: np.random.Generator,
    material: str = "steel",
    n_lumps: int = 2,
    fraction_each: float = 0.06,
    min_separation_m: float | None = None,
) -> np.ndarray:
    """`n_lumps` separated dense inclusions -- the constructive negative
    half of T2: two or more inclusions admit a continuous ambiguous family
    of alternative placements matching the same 10 moments."""
    rho = _empty(grid)
    rho[cavity_mask] = density_for("air")

    centers = grid.centers()
    cavity_centers = centers[cavity_mask]
    cavity_volume = cavity_mask.sum() * grid.voxel_volume_m3
    lump_volume = fraction_each * cavity_volume
    lump_radius = (3.0 * lump_volume / (4.0 * np.pi)) ** (1.0 / 3.0)
    if min_separation_m is None:
        min_separation_m = 2.5 * lump_radius

    bounds_min = cavity_centers.min(axis=0) + lump_radius
    bounds_max = cavity_centers.max(axis=0) - lump_radius
    placements: list[np.ndarray] = []
    for _ in range(200):  # rejection sampling for separation
        if len(placements) >= n_lumps:
            break
        if np.any(bounds_max < bounds_min):
            candidate = cavity_centers.mean(axis=0)
        else:
            candidate = rng.uniform(bounds_min, bounds_max)
        if all(np.linalg.norm(candidate - p) >= min_separation_m for p in placements):
            placements.append(candidate)

    for center in placements:
        dist = np.linalg.norm(centers - center[None, :], axis=1)
        lump_mask = (dist <= lump_radius) & cavity_mask
        rho[lump_mask] = density_for(material)
    return rho


def fill_layered(
    grid: VoxelGrid,
    cavity_mask: np.ndarray,
    rng: np.random.Generator,
    materials: tuple[str, ...] = ("sand", "water", "air"),
    axis: int = 2,
) -> np.ndarray:
    """Horizontal bands of different materials stacked along `axis`
    (default z, so this reads as sediment settled under gravity)."""
    rho = _empty(grid)
    centers = grid.centers()
    coord = centers[:, axis]
    cavity_coord = coord[cavity_mask]
    lo, hi = cavity_coord.min(), cavity_coord.max()

    n_layers = len(materials)
    boundaries = np.sort(rng.uniform(lo, hi, size=n_layers - 1)) if n_layers > 1 else np.array([])
    edges = np.concatenate([[lo - 1e-9], boundaries, [hi + 1e-9]])

    for layer_idx, material in enumerate(materials):
        band = (coord >= edges[layer_idx]) & (coord < edges[layer_idx + 1]) & cavity_mask
        rho[band] = density_for(material)
    return rho


def fill_lining(
    grid: VoxelGrid,
    cavity_mask: np.ndarray,
    rng: np.random.Generator,
    material: str = "lead",
    thickness_voxels: int = 2,
) -> np.ndarray:
    """Dense material coating the cavity's inner wall, empty (air) core --
    the sharpest visual and physical contrast with a slug of the same total
    mass placed in the middle: same shell, wildly different interior,
    exactly the pair the sharp-bound theorem is built to characterize."""
    rho = _empty(grid)
    rho[cavity_mask] = density_for("air")

    cavity_grid = grid.reshape(cavity_mask)
    core_grid = binary_erosion(cavity_grid, iterations=thickness_voxels, border_value=0)
    core_mask = grid.ravel(core_grid)
    lining_mask = cavity_mask & ~core_mask
    rho[lining_mask] = density_for(material)
    return rho


FILL_FACTORIES = {
    "void": fill_void,
    "slug": fill_slug,
    "twin_lumps": fill_twin_lumps,
    "layered": fill_layered,
    "lining": fill_lining,
}
