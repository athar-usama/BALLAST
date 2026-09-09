"""Outer shell geometry: the opaque, known-shape container whose INTERIOR is
what this project reconstructs. A shell is a thin wall of shell material
enclosing a hollow cavity, matching the "sealed opaque object" framing --
the wall itself contributes a known, fixed part of the moments, and the
`fills` module is what fills the cavity with an unknown interior.

Shells are defined directly as voxel occupancy masks (not triangle meshes)
because everything up to and including the observability atlas and the
interior solver only needs the density field, never a surface. Meshing for
rendering is a separate, later concern (`ballast.render`), and deliberately
does not touch this module: appearance must stay independent of contents,
so nothing here is allowed to leak into what the renderer draws.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Literal

import numpy as np

from ballast.moments.voxelgrid import VoxelGrid

ShellKind = Literal["box", "tin", "sphere"]


@dataclass(frozen=True)
class Shell:
    grid: VoxelGrid
    wall_mask: np.ndarray  # (V,) bool, voxels occupied by the shell wall itself
    cavity_mask: np.ndarray  # (V,) bool, hollow interior voxels available for filling
    kind: ShellKind


def _grid_for(extent_m: float, n: int) -> VoxelGrid:
    h = extent_m / n
    half = h * (n - 1) / 2
    return VoxelGrid(shape=(n, n, n), voxel_size_m=h, origin_m=np.array([-half, -half, -half]))


def make_box_shell(extent_m: float = 0.16, n: int = 24, wall_voxels: int = 2) -> Shell:
    """A rectangular box (e.g. a cardboard parcel), wall thickness in voxels."""
    grid = _grid_for(extent_m, n)
    centers = grid.centers()
    half_extent = extent_m / 2.0
    outer = np.all(np.abs(centers) <= half_extent, axis=1)
    wall_thickness_m = wall_voxels * grid.voxel_size_m
    inner = np.all(np.abs(centers) <= (half_extent - wall_thickness_m), axis=1)
    wall_mask = outer & ~inner
    cavity_mask = inner
    return Shell(grid=grid, wall_mask=wall_mask, cavity_mask=cavity_mask, kind="box")


def make_tin_shell(
    radius_m: float = 0.06, height_m: float = 0.12, n: int = 24, wall_voxels: int = 2
) -> Shell:
    """A cylindrical tin (axis along z, e.g. a coffee can)."""
    extent_m = 2 * max(radius_m, height_m / 2.0) * 1.05
    grid = _grid_for(extent_m, n)
    centers = grid.centers()
    radial = np.linalg.norm(centers[:, :2], axis=1)
    axial = np.abs(centers[:, 2])
    outer = (radial <= radius_m) & (axial <= height_m / 2.0)
    wall_thickness_m = wall_voxels * grid.voxel_size_m
    inner = (radial <= radius_m - wall_thickness_m) & (axial <= height_m / 2.0 - wall_thickness_m)
    wall_mask = outer & ~inner
    cavity_mask = inner
    return Shell(grid=grid, wall_mask=wall_mask, cavity_mask=cavity_mask, kind="tin")


def make_sphere_shell(radius_m: float = 0.07, n: int = 24, wall_voxels: int = 2) -> Shell:
    """A spherical shell (e.g. a hollow ball)."""
    extent_m = 2 * radius_m * 1.1
    grid = _grid_for(extent_m, n)
    centers = grid.centers()
    r = np.linalg.norm(centers, axis=1)
    wall_thickness_m = wall_voxels * grid.voxel_size_m
    outer = r <= radius_m
    inner = r <= radius_m - wall_thickness_m
    wall_mask = outer & ~inner
    cavity_mask = inner
    return Shell(grid=grid, wall_mask=wall_mask, cavity_mask=cavity_mask, kind="sphere")


SHELL_FACTORIES = {
    "box": make_box_shell,
    "tin": make_tin_shell,
    "sphere": make_sphere_shell,
}


def make_shell(kind: ShellKind, **kwargs) -> Shell:
    return SHELL_FACTORIES[kind](**kwargs)
