"""Figure: two interiors, mirror images of each other, provably identical
moments. A steel spiral of slugs strung on a thin wire, and its
reflection through its own center of mass -- rendered as the bare
interior (no shell) so the chirality is visible, with a numeric readout
of the moments confirming they match to machine precision.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np
import trimesh

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import INK, SURFACE, apply_style, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.render.camera import Camera  # noqa: E402
from ballast.render.cpu_fallback import rasterize_cpu  # noqa: E402
from ballast.render.gl import GLRasterizer, gl_available  # noqa: E402
from ballast.render.meshes import cylinder_mesh, sphere_mesh  # noqa: E402

SURFACE_RGB = np.array([int(SURFACE[1:3], 16), int(SURFACE[3:5], 16), int(SURFACE[5:7], 16)]) / 255.0


def build_spiral_slugs(n_slugs: int = 14, radius_m: float = 0.045, turns: float = 2.2) -> np.ndarray:
    """A helical spiral of slugs, radius tapering toward both ends -- a
    shape whose mirror image is a visibly different (opposite-handed)
    spiral, not just a rotated copy of itself."""
    t = np.linspace(0, 1, n_slugs)
    theta = t * turns * 2 * np.pi
    r = radius_m * (0.25 + 0.75 * np.sin(np.pi * t))
    z = (t - 0.5) * 0.11
    return np.stack([r * np.cos(theta), r * np.sin(theta), z], axis=1)


def _segment_pose(p0: np.ndarray, p1: np.ndarray) -> tuple[np.ndarray, np.ndarray, float]:
    """Rotation + translation placing a z-axis-aligned cylinder between two points."""
    delta = p1 - p0
    length = float(np.linalg.norm(delta))
    z_axis = delta / (length + 1e-12)
    helper = np.array([0.0, 0.0, 1.0]) if abs(z_axis[2]) < 0.9 else np.array([1.0, 0.0, 0.0])
    x_axis = np.cross(helper, z_axis)
    x_axis /= np.linalg.norm(x_axis) + 1e-12
    y_axis = np.cross(z_axis, x_axis)
    rotation = np.stack([x_axis, y_axis, z_axis], axis=1)
    midpoint = (p0 + p1) / 2.0
    return rotation, midpoint, length


def render_spiral(positions: np.ndarray, camera: Camera, use_gl: bool, base_color: tuple, wire_color: tuple):
    canvas = np.ones((camera.height_px, camera.width_px, 3)) * SURFACE_RGB[None, None, :]
    depth_buf = np.full((camera.height_px, camera.width_px), np.inf)
    gl = GLRasterizer(camera.width_px, camera.height_px) if use_gl else None

    def draw(mesh, rotation, translation, color):
        if use_gl:
            rgb, mask = gl.render(mesh, rotation, translation, camera, base_color=color)
            canvas[mask] = rgb[mask] / 255.0
        else:
            mask, depth = rasterize_cpu(mesh, rotation, translation, camera)
            closer = mask & (depth < depth_buf)
            canvas[closer] = np.array(color)
            depth_buf[closer] = depth[closer]

    try:
        wire_mesh_cache = cylinder_mesh(0.0035, 1.0, n_segments=10)
        for p0, p1 in zip(positions[:-1], positions[1:], strict=True):
            rotation, midpoint, length = _segment_pose(p0, p1)
            scaled = trimesh.Trimesh(
                vertices=wire_mesh_cache.vertices * np.array([1.0, 1.0, length]),
                faces=wire_mesh_cache.faces,
                process=False,
            )
            scaled_mesh = type(wire_mesh_cache)(
                vertices=scaled.vertices, faces=scaled.faces, normals=wire_mesh_cache.normals
            )
            draw(scaled_mesh, rotation, midpoint, wire_color)

        n = len(positions)
        for i, p in enumerate(positions):
            t = i / (n - 1)
            color = tuple(np.array(base_color) * (0.55 + 0.45 * t) + np.array([0.0, 0.0, 0.0]) * (1 - t))
            radius = 0.007 + 0.006 * (0.3 + 0.7 * np.sin(np.pi * t))
            s = sphere_mesh(radius, n_lat=12, n_lon=18)
            draw(s, np.eye(3), p, color)
    finally:
        if gl is not None:
            gl.release()
    return canvas


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    positions = build_spiral_slugs()

    # The provably exact construction: point-reflect through the object's
    # OWN center of mass, x -> 2c - x, in continuous 3D space. This is NOT
    # the same as mirroring through a fixed world plane (negating one
    # coordinate): that only preserves the full inertia tensor if the
    # mirror plane happens to pass through the object's own center of mass
    # AND be one of its principal planes -- neither holds for an arbitrary
    # spiral. Point reflection through the object's own centroid holds
    # unconditionally, for any mass distribution.
    slug_mass = 1.0
    com = positions.mean(axis=0)
    mirrored = 2.0 * com - positions

    def point_mass_moments(pos: np.ndarray, mass: float) -> tuple[float, np.ndarray, np.ndarray]:
        m = mass * pos.shape[0]
        c = (mass * pos).sum(axis=0) / m
        rel = pos - c[None, :]
        s_com = mass * (rel[:, :, None] * rel[:, None, :]).sum(axis=0)
        i_com = np.trace(s_com) * np.eye(3) - s_com
        return m, c, i_com

    m1, c1, i1 = point_mass_moments(positions, slug_mass)
    m2, c2, i2 = point_mass_moments(mirrored, slug_mass)

    use_gl = gl_available()
    extent = np.linalg.norm(positions, axis=1).max() * 3.6
    cam = Camera(
        width_px=460,
        height_px=460,
        focal_length_px=420.0,
        eye_m=extent * np.array([np.cos(0.7) * np.cos(0.4), np.sin(0.7) * np.cos(0.4), np.sin(0.4)]),
        target_m=np.zeros(3),
    )

    img_left = render_spiral(positions, cam, use_gl, base_color=(0.16, 0.47, 0.84), wire_color=(0.75, 0.82, 0.9))
    img_right = render_spiral(mirrored, cam, use_gl, base_color=(0.92, 0.41, 0.2), wire_color=(0.92, 0.85, 0.78))

    import matplotlib.pyplot as plt

    fig, axes = plt.subplots(1, 2, figsize=(9, 5.0))
    for ax, img, title in zip(axes, [img_left, img_right], ["a steel spiral", "its mirror image"], strict=True):
        ax.imshow(img)
        ax.set_title(title, fontsize=13, color=INK)
        ax.axis("off")
    fig.suptitle(
        "Two different objects. Every motion of one looks exactly like a motion of the other.",
        fontsize=12.5,
        color=INK,
        y=1.0,
    )
    save(fig, str(out_dir / "chirality_twins.png"))

    print("mass          :", m1, m2, "  diff:", abs(m1 - m2))
    print("center of mass:", c1, c2, "  diff:", np.max(np.abs(c1 - c2)))
    print("inertia diff  :", np.max(np.abs(i1 - i2)))


if __name__ == "__main__":
    main()
