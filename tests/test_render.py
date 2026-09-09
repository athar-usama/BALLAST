"""Tests for the camera, meshes, and both rasterizers -- including the
golden-image cross-check between the moderngl and pure-NumPy paths that
the project plan calls for: two independent implementations of the same
projection agreeing on a silhouette is real evidence neither has a sign or
winding bug, and it is what lets the repository run identically whether or
not a usable GPU is present.
"""

from __future__ import annotations

import numpy as np
import pytest

from ballast.render.camera import Camera, orbit_camera
from ballast.render.cpu_fallback import rasterize_cpu
from ballast.render.gl import gl_available
from ballast.render.meshes import box_mesh, cylinder_mesh, mesh_for_shell, sphere_mesh

GL_OK = gl_available()


# --------------------------------------------------------------------------
# camera
# --------------------------------------------------------------------------


def test_project_places_a_point_above_target_higher_on_screen():
    """A world point directly above the camera's target should project to
    a SMALLER row index (higher on screen), matching image convention
    (row 0 at the top)."""
    cam = Camera(
        width_px=128,
        height_px=128,
        focal_length_px=200.0,
        eye_m=np.array([0.0, -1.0, 0.0]),
        target_m=np.zeros(3),
    )
    center_pixel = cam.project(np.array([[0.0, 0.0, 0.0]]))[0]
    above_pixel = cam.project(np.array([[0.0, 0.0, 0.05]]))[0]

    assert center_pixel[0] == pytest.approx(cam.width_px / 2, abs=1e-6)
    assert center_pixel[1] == pytest.approx(cam.height_px / 2, abs=1e-6)
    assert above_pixel[1] < center_pixel[1]
    assert above_pixel[0] == pytest.approx(center_pixel[0], abs=1e-6)


def test_project_matches_pinhole_geometry_for_a_lateral_offset():
    cam = Camera(
        width_px=128,
        height_px=128,
        focal_length_px=200.0,
        eye_m=np.array([0.0, -1.0, 0.0]),
        target_m=np.zeros(3),
    )
    offset_m = 0.1
    px = cam.project(np.array([[offset_m, 0.0, 0.0]]))[0]
    expected_dx = cam.focal_length_px * offset_m / 1.0  # distance to point is ~1 m along forward
    assert px[0] == pytest.approx(cam.width_px / 2 + expected_dx, rel=0.01)


def test_orbit_camera_stays_at_the_requested_distance():
    cam = orbit_camera(64, 64, distance_m=0.8, azimuth_rad=1.2, elevation_rad=0.3, focal_length_px=150.0)
    assert np.linalg.norm(cam.eye_m - cam.target_m) == pytest.approx(0.8)


# --------------------------------------------------------------------------
# meshes
# --------------------------------------------------------------------------


def test_box_mesh_vertices_at_correct_extent():
    mesh = box_mesh(half_extent_m=0.05)
    assert np.max(np.abs(mesh.vertices)) == pytest.approx(0.05)
    assert mesh.faces.max() < mesh.vertices.shape[0]


def test_sphere_mesh_vertices_on_the_sphere():
    mesh = sphere_mesh(radius_m=0.07, n_lat=10, n_lon=20)
    radii = np.linalg.norm(mesh.vertices, axis=1)
    assert np.allclose(radii, 0.07, atol=1e-9)
    assert np.allclose(np.linalg.norm(mesh.normals, axis=1), 1.0, atol=1e-6)


def test_cylinder_mesh_dimensions():
    mesh = cylinder_mesh(radius_m=0.05, height_m=0.1, n_segments=16)
    radial = np.linalg.norm(mesh.vertices[:, :2], axis=1)
    assert np.all(radial <= 0.05 + 1e-9)
    assert np.max(radial) == pytest.approx(0.05, abs=1e-9)
    assert np.max(mesh.vertices[:, 2]) == pytest.approx(0.05, abs=1e-9)
    assert np.min(mesh.vertices[:, 2]) == pytest.approx(-0.05, abs=1e-9)


def test_mesh_for_shell_matches_default_shape_dimensions():
    box = mesh_for_shell("box")
    assert np.max(np.abs(box.vertices)) == pytest.approx(0.08)
    sphere = mesh_for_shell("sphere")
    assert np.allclose(np.linalg.norm(sphere.vertices, axis=1), 0.07, atol=1e-9)


# --------------------------------------------------------------------------
# CPU rasterizer
# --------------------------------------------------------------------------


def test_cpu_rasterizer_sphere_silhouette_matches_pinhole_prediction():
    """A sphere of known radius at a known distance should project to a
    circular silhouette of the pinhole-predicted pixel radius."""
    radius_m, distance_m, focal_px = 0.07, 1.0, 200.0
    mesh = sphere_mesh(radius_m, n_lat=24, n_lon=48)
    cam = Camera(
        width_px=128,
        height_px=128,
        focal_length_px=focal_px,
        eye_m=np.array([0.0, -distance_m, 0.0]),
        target_m=np.zeros(3),
    )
    mask, depth = rasterize_cpu(mesh, np.eye(3), np.zeros(3), cam)

    expected_pixel_radius = focal_px * radius_m / distance_m
    expected_area = np.pi * expected_pixel_radius**2
    assert mask.sum() == pytest.approx(expected_area, rel=0.1)
    assert np.isfinite(depth[mask]).all()
    assert np.all(depth[mask] > 0)


def test_cpu_rasterizer_depth_is_nearest_surface():
    """For a sphere centered at the origin viewed from distance d, the
    nearest visible depth should be d - radius (the near side)."""
    radius_m, distance_m = 0.05, 0.5
    mesh = sphere_mesh(radius_m, n_lat=20, n_lon=40)
    cam = Camera(
        width_px=96,
        height_px=96,
        focal_length_px=150.0,
        eye_m=np.array([0.0, -distance_m, 0.0]),
        target_m=np.zeros(3),
    )
    mask, depth = rasterize_cpu(mesh, np.eye(3), np.zeros(3), cam)
    min_depth = depth[mask].min()
    assert min_depth == pytest.approx(distance_m - radius_m, rel=0.05)


def test_cpu_rasterizer_empty_when_object_behind_camera():
    mesh = sphere_mesh(0.05)
    cam = Camera(
        width_px=64,
        height_px=64,
        focal_length_px=100.0,
        eye_m=np.array([0.0, 1.0, 0.0]),
        target_m=np.zeros(3),
    )
    # object is at the origin, camera is on the +y side looking toward origin,
    # but place the mesh far behind the camera instead to confirm nothing draws
    mask, _depth = rasterize_cpu(mesh, np.eye(3), np.array([0.0, 5.0, 0.0]), cam)
    assert mask.sum() == 0


# --------------------------------------------------------------------------
# GL vs CPU golden-image cross-check
# --------------------------------------------------------------------------


@pytest.mark.skipif(not GL_OK, reason="no usable OpenGL context on this machine")
@pytest.mark.parametrize("shell_kind", ["box", "tin", "sphere"])
def test_gl_and_cpu_rasterizers_agree_on_silhouette(shell_kind):
    """The golden-image test: two independent rasterizer implementations
    should agree closely on which pixels are covered, for the same mesh,
    pose, and camera."""
    from ballast.render.gl import GLRasterizer

    mesh = mesh_for_shell(shell_kind)
    cam = orbit_camera(128, 128, distance_m=0.4, azimuth_rad=0.7, elevation_rad=0.4, focal_length_px=180.0)
    rotation = np.eye(3)
    translation = np.zeros(3)

    mask_cpu, _depth = rasterize_cpu(mesh, rotation, translation, cam)

    gl = GLRasterizer(128, 128)
    try:
        _rgb, mask_gl = gl.render(mesh, rotation, translation, cam)
    finally:
        gl.release()

    intersection = np.sum(mask_cpu & mask_gl)
    union = np.sum(mask_cpu | mask_gl)
    iou = intersection / union
    assert iou > 0.9, f"GL and CPU silhouettes should closely agree, IoU={iou}"
