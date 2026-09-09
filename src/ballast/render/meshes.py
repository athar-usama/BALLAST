"""Simple parametric triangle meshes for the three Rattlebox shell shapes,
matching `ballast.rattlebox.shells` dimensions exactly so the rendered
exterior is the same object the physics is computed for.

These are hand-built analytic meshes rather than a marching-cubes surface
extracted from the voxel occupancy, deliberately: the renderer only ever
needs to draw the known OUTER shape (appearance must stay independent of
contents -- see the module docstring in `ballast.rattlebox.shells`), and an
analytic mesh is both exact and far cheaper than voxel meshing.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np


@dataclass(frozen=True)
class Mesh:
    vertices: np.ndarray  # (N, 3)
    faces: np.ndarray  # (M, 3) int, CCW winding when viewed from outside
    normals: np.ndarray  # (N, 3) per-vertex normals


def _face_normals_to_vertex_normals(vertices: np.ndarray, faces: np.ndarray) -> np.ndarray:
    normals = np.zeros_like(vertices)
    for f in faces:
        v0, v1, v2 = vertices[f]
        n = np.cross(v1 - v0, v2 - v0)
        for idx in f:
            normals[idx] += n
    lengths = np.linalg.norm(normals, axis=1, keepdims=True)
    lengths = np.where(lengths < 1e-12, 1.0, lengths)
    return normals / lengths


def box_mesh(half_extent_m: float) -> Mesh:
    h = half_extent_m
    verts = np.array(
        [
            [-h, -h, -h],
            [h, -h, -h],
            [h, h, -h],
            [-h, h, -h],
            [-h, -h, h],
            [h, -h, h],
            [h, h, h],
            [-h, h, h],
        ]
    )
    faces = np.array(
        [
            [0, 2, 1],
            [0, 3, 2],  # bottom (z = -h), CCW from outside (looking from -z)
            [4, 5, 6],
            [4, 6, 7],  # top (z = +h)
            [0, 1, 5],
            [0, 5, 4],  # front (y = -h)
            [1, 2, 6],
            [1, 6, 5],  # right (x = +h)
            [2, 3, 7],
            [2, 7, 6],  # back (y = +h)
            [3, 0, 4],
            [3, 4, 7],  # left (x = -h)
        ]
    )
    normals = _face_normals_to_vertex_normals(verts, faces)
    return Mesh(vertices=verts, faces=faces, normals=normals)


def cylinder_mesh(radius_m: float, height_m: float, n_segments: int = 32) -> Mesh:
    theta = np.linspace(0, 2 * np.pi, n_segments, endpoint=False)
    x, y = radius_m * np.cos(theta), radius_m * np.sin(theta)
    half_h = height_m / 2.0

    bottom_ring = np.stack([x, y, np.full(n_segments, -half_h)], axis=1)
    top_ring = np.stack([x, y, np.full(n_segments, half_h)], axis=1)
    bottom_center = np.array([[0.0, 0.0, -half_h]])
    top_center = np.array([[0.0, 0.0, half_h]])

    verts = np.concatenate([bottom_ring, top_ring, bottom_center, top_center], axis=0)
    bc_idx, tc_idx = 2 * n_segments, 2 * n_segments + 1

    faces = []
    for i in range(n_segments):
        j = (i + 1) % n_segments
        # side quad, split into 2 triangles, CCW from outside
        faces.append([i, n_segments + i, n_segments + j])
        faces.append([i, n_segments + j, j])
        # bottom cap (normal points -z, so CCW when viewed from below)
        faces.append([bc_idx, j, i])
        # top cap (normal points +z)
        faces.append([tc_idx, n_segments + i, n_segments + j])
    faces = np.array(faces)

    normals = _face_normals_to_vertex_normals(verts, faces)
    return Mesh(vertices=verts, faces=faces, normals=normals)


def sphere_mesh(radius_m: float, n_lat: int = 16, n_lon: int = 32) -> Mesh:
    lat = np.linspace(-np.pi / 2, np.pi / 2, n_lat)
    lon = np.linspace(0, 2 * np.pi, n_lon, endpoint=False)
    verts = []
    for phi in lat:
        for lam in lon:
            verts.append(
                [
                    radius_m * np.cos(phi) * np.cos(lam),
                    radius_m * np.cos(phi) * np.sin(lam),
                    radius_m * np.sin(phi),
                ]
            )
    verts = np.array(verts)

    faces = []
    for i in range(n_lat - 1):
        for j in range(n_lon):
            j2 = (j + 1) % n_lon
            a = i * n_lon + j
            b = i * n_lon + j2
            c = (i + 1) * n_lon + j
            d = (i + 1) * n_lon + j2
            faces.append([a, b, d])
            faces.append([a, d, c])
    faces = np.array(faces)

    # sphere normals are exact: the outward radial direction
    normals = verts / (np.linalg.norm(verts, axis=1, keepdims=True) + 1e-12)
    return Mesh(vertices=verts, faces=faces, normals=normals)


def mesh_for_shell(shell_kind: str, **shell_kwargs) -> Mesh:
    """Build the exterior mesh matching a `ballast.rattlebox.shells` factory
    call, using the SAME default dimensions so a caller that only passes
    `shell_kind` gets a mesh consistent with the default physics shell."""
    if shell_kind == "box":
        extent_m = shell_kwargs.get("extent_m", 0.16)
        return box_mesh(extent_m / 2.0)
    if shell_kind == "tin":
        radius_m = shell_kwargs.get("radius_m", 0.06)
        height_m = shell_kwargs.get("height_m", 0.12)
        return cylinder_mesh(radius_m, height_m)
    if shell_kind == "sphere":
        radius_m = shell_kwargs.get("radius_m", 0.07)
        return sphere_mesh(radius_m)
    raise ValueError(f"unknown shell kind '{shell_kind}'")
