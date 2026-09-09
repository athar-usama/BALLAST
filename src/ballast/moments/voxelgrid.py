"""Axis-aligned voxel grid used to discretize an object's interior density field."""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class VoxelGrid:
    """A regular cubic-voxel grid in world coordinates.

    `origin_m` is the world position of the CENTER of voxel index (0, 0, 0).
    Voxel (i, j, k) has its center at `origin_m + [i, j, k] * voxel_size_m`
    and occupies a cube of side `voxel_size_m` around that point.

    The origin is a free real-valued offset: it does not need to align to any
    external coordinate system, and shifting it never requires resampling the
    density array, since it only changes the coordinates used when the moment
    operator evaluates each voxel's position. That freedom is what lets
    `moments.ghosts.recentered_grid` place the grid's geometric center exactly
    on an object's center of mass without touching a single density value.
    """

    shape: tuple[int, int, int]
    voxel_size_m: float
    origin_m: np.ndarray  # (3,)

    def __post_init__(self) -> None:
        object.__setattr__(self, "origin_m", np.asarray(self.origin_m, dtype=np.float64))
        if self.origin_m.shape != (3,):
            raise ValueError("origin_m must have shape (3,)")
        if self.voxel_size_m <= 0:
            raise ValueError("voxel_size_m must be positive")
        if any(n <= 0 for n in self.shape):
            raise ValueError("shape must be all-positive")

    @property
    def n_voxels(self) -> int:
        nx, ny, nz = self.shape
        return nx * ny * nz

    @property
    def voxel_volume_m3(self) -> float:
        return self.voxel_size_m**3

    def centers(self) -> np.ndarray:
        """Return (V, 3) array of voxel center world coordinates, C-order flattened
        to match `rho.ravel()` on an array of shape `self.shape`."""
        nx, ny, nz = self.shape
        ii, jj, kk = np.meshgrid(np.arange(nx), np.arange(ny), np.arange(nz), indexing="ij")
        idx = np.stack([ii.ravel(), jj.ravel(), kk.ravel()], axis=1).astype(np.float64)
        return self.origin_m[None, :] + idx * self.voxel_size_m

    def geometric_center_m(self) -> np.ndarray:
        """World position of the grid's own geometric center (the fixed point of
        `np.flip` along all three axes)."""
        n = np.asarray(self.shape, dtype=np.float64)
        return self.origin_m + (n - 1.0) / 2.0 * self.voxel_size_m

    def with_origin(self, new_origin_m: np.ndarray) -> VoxelGrid:
        return VoxelGrid(shape=self.shape, voxel_size_m=self.voxel_size_m, origin_m=new_origin_m)

    def reshape(self, rho_flat: np.ndarray) -> np.ndarray:
        return rho_flat.reshape(self.shape)

    def ravel(self, rho_grid: np.ndarray) -> np.ndarray:
        return rho_grid.reshape(-1)
