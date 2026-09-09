"""A minimal pinhole camera: intrinsics plus a look-at world pose. Kept
deliberately small -- everything downstream (the GL rasterizer, the CPU
fallback, the SDT-based pose refinement) only ever needs to project a
world point to a pixel and to know the view/projection matrices for a
rasterizer.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np


def _default_up() -> np.ndarray:
    return np.array([0.0, 0.0, 1.0])


@dataclass(frozen=True)
class Camera:
    width_px: int
    height_px: int
    focal_length_px: float
    eye_m: np.ndarray  # (3,) world-frame camera position
    target_m: np.ndarray  # (3,) world-frame point the camera looks at
    up_hint: np.ndarray = field(default_factory=_default_up)
    near_m: float = 0.01
    far_m: float = 10.0

    def view_matrix(self) -> np.ndarray:
        """World-to-camera rigid transform (4, 4), camera looks down -Z in
        its own frame, +X right, +Y up (OpenGL convention)."""
        forward = self.target_m - self.eye_m
        forward = forward / np.linalg.norm(forward)
        right = np.cross(forward, self.up_hint)
        right = right / np.linalg.norm(right)
        true_up = np.cross(right, forward)

        r = np.stack([right, true_up, -forward], axis=0)  # rows: camera axes in world coords
        t = -r @ self.eye_m
        m = np.eye(4)
        m[:3, :3] = r
        m[:3, 3] = t
        return m

    def projection_matrix(self) -> np.ndarray:
        """OpenGL-style perspective projection (4, 4) from the pinhole
        intrinsics, mapping camera-space points to clip space."""
        fx = self.focal_length_px
        w, h = self.width_px, self.height_px
        n, f = self.near_m, self.far_m
        m = np.zeros((4, 4))
        m[0, 0] = 2 * fx / w
        m[1, 1] = 2 * fx / h
        m[2, 2] = -(f + n) / (f - n)
        m[2, 3] = -2 * f * n / (f - n)
        m[3, 2] = -1.0
        return m

    def intrinsics_matrix(self) -> np.ndarray:
        """(3, 3) pinhole intrinsics K, principal point at image center."""
        fx = self.focal_length_px
        cx, cy = self.width_px / 2.0, self.height_px / 2.0
        return np.array([[fx, 0, cx], [0, fx, cy], [0, 0, 1.0]])

    def project(self, points_world: np.ndarray) -> np.ndarray:
        """World points (N, 3) -> pixel coordinates (N, 2), using the view
        matrix and intrinsics directly (a simpler, equivalent route to the
        same answer as the GL clip-space pipeline, used by the CPU
        rasterizer and by pose refinement)."""
        view = self.view_matrix()
        r, t = view[:3, :3], view[:3, 3]
        cam_pts = points_world @ r.T + t  # (N, 3), camera looks down -Z
        z = -cam_pts[:, 2]
        z = np.where(np.abs(z) < 1e-9, 1e-9, z)
        k = self.intrinsics_matrix()
        px = k[0, 0] * cam_pts[:, 0] / z + k[0, 2]
        py = -k[1, 1] * cam_pts[:, 1] / z + k[1, 2]  # flip: image row grows downward
        return np.stack([px, py], axis=1)


def orbit_camera(
    width_px: int,
    height_px: int,
    distance_m: float,
    azimuth_rad: float,
    elevation_rad: float,
    focal_length_px: float,
    target_m: np.ndarray | None = None,
) -> Camera:
    """A camera on a sphere of radius `distance_m` around `target_m`,
    parameterized by azimuth/elevation -- the standard way this project
    generates a viewpoint for a rendered clip or a figure."""
    if target_m is None:
        target_m = np.zeros(3)
    eye = target_m + distance_m * np.array(
        [
            np.cos(elevation_rad) * np.cos(azimuth_rad),
            np.cos(elevation_rad) * np.sin(azimuth_rad),
            np.sin(elevation_rad),
        ]
    )
    return Camera(
        width_px=width_px,
        height_px=height_px,
        focal_length_px=focal_length_px,
        eye_m=eye,
        target_m=target_m,
    )
