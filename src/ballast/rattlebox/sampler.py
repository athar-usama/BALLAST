"""Sample a complete Rattlebox object: a shell (with its own wall material)
plus a filled interior, with the ground-truth moments computed directly
from the voxel density field via the exact moment operator. This is the
generator's only interface to the rest of the project -- everything
downstream (rendering, pose fitting, the interior solver) treats a
`RattleboxObject` as opaque and only ever gets to see its MOTION, never
`rho` directly, which is what makes this an honest benchmark rather than a
leak of the answer.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Literal

import numpy as np

from ballast.moments.operator import build_moment_operator, moments_to_inertia
from ballast.moments.voxelgrid import VoxelGrid
from ballast.rattlebox.fills import FILL_FACTORIES
from ballast.rattlebox.materials import density_for
from ballast.rattlebox.shells import SHELL_FACTORIES, ShellKind

FillKind = Literal["void", "slug", "twin_lumps", "layered", "lining"]

WALL_MATERIALS = ("plastic", "aluminium", "steel", "wood")


@dataclass(frozen=True)
class RattleboxObject:
    grid: VoxelGrid
    rho: np.ndarray  # (V,) full density field -- GROUND TRUTH, never exposed to any estimator
    shell_kind: ShellKind
    fill_kind: FillKind
    wall_material: str
    mass_kg: float
    com_body_m: np.ndarray  # (3,) center of mass, in the grid's own coordinate frame
    inertia_body_kgm2: np.ndarray  # (3, 3) inertia tensor about the center of mass
    metadata: dict = field(default_factory=dict)


def sample_object(
    rng: np.random.Generator,
    shell_kind: ShellKind | None = None,
    fill_kind: FillKind | None = None,
    n_voxels_per_axis: int = 24,
) -> RattleboxObject:
    """Draw one random Rattlebox object: a random shell shape and wall
    material, and a random interior fill pattern (or the ones given, if the
    caller wants a specific combination -- e.g. for the ghost-gallery and
    reconstruction-sheet figures, which need matched pairs)."""
    if shell_kind is None:
        shell_kind = rng.choice(list(SHELL_FACTORIES))
    if fill_kind is None:
        fill_kind = rng.choice(list(FILL_FACTORIES))
    wall_material = rng.choice(WALL_MATERIALS)

    shell_factory = SHELL_FACTORIES[shell_kind]
    shell = shell_factory(n=n_voxels_per_axis)

    fill_factory = FILL_FACTORIES[fill_kind]
    rho = fill_factory(shell.grid, shell.cavity_mask, rng)
    rho[shell.wall_mask] = density_for(wall_material)

    a = build_moment_operator(shell.grid)
    b = a @ rho
    mass, com, inertia = moments_to_inertia(b)

    return RattleboxObject(
        grid=shell.grid,
        rho=rho,
        shell_kind=shell_kind,
        fill_kind=fill_kind,
        wall_material=wall_material,
        mass_kg=mass,
        com_body_m=com,
        inertia_body_kgm2=inertia,
        metadata={"n_voxels_per_axis": n_voxels_per_axis},
    )


def sample_dataset(n: int, seed: int = 0, n_voxels_per_axis: int = 24) -> list[RattleboxObject]:
    """Draw `n` independent random objects -- the Rattlebox dataset
    generator's entry point, one call per object with a fixed seed for
    reproducibility (`scripts/run_all.py` calls this)."""
    rng = np.random.default_rng(seed)
    return [sample_object(rng, n_voxels_per_axis=n_voxels_per_axis) for _ in range(n)]
