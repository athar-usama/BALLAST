"""A pure-NumPy scanline rasterizer: silhouette mask and depth buffer, no
GPU required. This exists so the repository runs with zero GPU (the
compute budget in the project plan explicitly does not require one for
rendering), and its output is checked against the `moderngl` rasterizer's
in a golden-image test -- two independent implementations of the same
projection agreeing on the silhouette is real evidence neither has a sign
or winding bug.

Per-triangle bounding-box rasterization with plain (non-perspective-
corrected) barycentric depth interpolation, which is accurate enough for
objects whose size is small relative to their distance from the camera
(true throughout this project: ~10-20 cm objects viewed from ~1 m).
"""

from __future__ import annotations

import numpy as np

from ballast.render.camera import Camera
from ballast.render.meshes import Mesh


def _project_with_depth(vertices_world: np.ndarray, camera: Camera) -> tuple[np.ndarray, np.ndarray]:
    view = camera.view_matrix()
    r, t = view[:3, :3], view[:3, 3]
    cam_pts = vertices_world @ r.T + t
    depth = -cam_pts[:, 2]  # positive in front of the camera
    px = camera.project(vertices_world)
    return px, depth


def rasterize_cpu(
    mesh: Mesh, rotation: np.ndarray, translation: np.ndarray, camera: Camera
) -> tuple[np.ndarray, np.ndarray]:
    """Render `mesh` posed by (rotation, translation) [body-to-world] from
    `camera`. Returns (mask, depth): `mask` is (H, W) bool, `depth` is
    (H, W) float32 filled with `np.inf` where nothing was drawn."""
    w, h = camera.width_px, camera.height_px
    mask = np.zeros((h, w), dtype=bool)
    depth = np.full((h, w), np.inf, dtype=np.float32)

    verts_world = mesh.vertices @ rotation.T + translation[None, :]
    px, vdepth = _project_with_depth(verts_world, camera)

    for face in mesh.faces:
        d = vdepth[face]
        if np.any(d <= camera.near_m):
            continue  # a vertex is behind (or at) the camera: skip, avoids projective blow-up

        p = px[face]  # (3, 2)
        x_min = max(int(np.floor(p[:, 0].min())), 0)
        x_max = min(int(np.ceil(p[:, 0].max())), w - 1)
        y_min = max(int(np.floor(p[:, 1].min())), 0)
        y_max = min(int(np.ceil(p[:, 1].max())), h - 1)
        if x_max < x_min or y_max < y_min:
            continue

        xs, ys = np.meshgrid(np.arange(x_min, x_max + 1), np.arange(y_min, y_max + 1))
        xs = xs.astype(np.float64) + 0.5
        ys = ys.astype(np.float64) + 0.5

        # barycentric coordinates via the standard edge-function / signed-area method
        x0, y0 = p[0]
        x1, y1 = p[1]
        x2, y2 = p[2]
        denom = (y1 - y2) * (x0 - x2) + (x2 - x1) * (y0 - y2)
        if abs(denom) < 1e-12:
            continue  # degenerate (zero-area) triangle in screen space

        w0 = ((y1 - y2) * (xs - x2) + (x2 - x1) * (ys - y2)) / denom
        w1 = ((y2 - y0) * (xs - x2) + (x0 - x2) * (ys - y2)) / denom
        w2 = 1.0 - w0 - w1

        inside = (w0 >= 0) & (w1 >= 0) & (w2 >= 0)
        if not np.any(inside):
            continue

        interp_depth = w0 * d[0] + w1 * d[1] + w2 * d[2]

        sub_y, sub_x = np.nonzero(inside)
        global_y = sub_y + y_min
        global_x = sub_x + x_min
        candidate_depth = interp_depth[inside]

        current = depth[global_y, global_x]
        closer = candidate_depth < current
        gy, gx, cd = global_y[closer], global_x[closer], candidate_depth[closer]
        depth[gy, gx] = cd
        mask[gy, gx] = True

    return mask, depth
