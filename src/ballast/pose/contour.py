"""Silhouette-based pose refinement via a signed distance transform (SDT),
not a differentiable renderer.

The technical review behind this project's plan was explicit about why:
soft rasterization is slower, worse-conditioned, and drags in an entire
dependency class (a differentiable rendering backend) that this project
does not otherwise need. The PWP3D / SRT3D lineage instead precomputes the
signed distance transform of the OBSERVED silhouette once per frame, then
at each iteration renders a HARD silhouette from the current pose
hypothesis, back-projects its occluding-contour pixels into 3D points on
the mesh surface (using the rasterizer's own depth buffer -- no per-
triangle bookkeeping needed), and minimizes the observed SDT sampled at
those points' reprojection under a pose update. The rasterizer itself is
never differentiated; only the analytic projection of a FIXED set of 3D
landmark points is, and that Jacobian is 6 columns wide, so a numeric
central difference is exact enough and simpler than hand-deriving one.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.ndimage import binary_erosion, distance_transform_edt, map_coordinates

from ballast.render.camera import Camera
from ballast.render.cpu_fallback import rasterize_cpu
from ballast.render.meshes import Mesh


def signed_distance_transform(mask: np.ndarray) -> np.ndarray:
    """Signed distance (in pixels) to the silhouette boundary: negative
    inside the mask, positive outside, zero on the boundary. Computed once
    per observed frame, and reused across every Gauss-Newton iteration."""
    inside = distance_transform_edt(mask)
    outside = distance_transform_edt(~mask)
    return outside - inside


def _sample_sdt(sdt: np.ndarray, pixel_coords: np.ndarray) -> np.ndarray:
    """Bilinear sample of the SDT at (N, 2) pixel (x, y) coordinates."""
    coords = np.stack([pixel_coords[:, 1], pixel_coords[:, 0]], axis=0)  # map_coordinates wants (row, col)
    return map_coordinates(sdt, coords, order=1, mode="nearest")


def backproject_contour_points(
    mask: np.ndarray, depth: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: Camera
) -> np.ndarray:
    """Back-project a rendered mask's occluding contour into 3D points on
    the mesh surface, expressed in the BODY frame. Uses the rasterizer's
    own depth buffer, so no per-triangle tracking is needed -- this is
    what lets pose refinement work directly off (mask, depth), the same
    two outputs both rasterizers already produce.

    The pixels immediately inside a binary mask's boundary do NOT sit on
    the true (sub-pixel) edge: their own signed distance is systematically
    around -1, not 0 (a pixel-grid Euclidean distance transform gives a
    boundary-adjacent interior pixel a distance of about one pixel to the
    nearest exterior pixel). Naively treating those pixel centers as
    landmarks biases the whole fit by about a pixel outward. The fix used
    here is standard for signed-distance-based tracking: project each
    candidate pixel onto the true zero level by walking along the local
    SDT gradient by exactly `-sdt_value`, before doing the depth lookup.

    Known limitation: this single-gradient-step correction is exact for a
    point on a locally straight edge, but the SDT has a genuine kink at a
    silhouette CORNER (two edges meeting), where a single gradient
    direction overshoots -- by up to roughly a pixel at typical object
    sizes and resolutions in this project. For a small, few-cornered
    silhouette (a box viewed near face-on at ~50px across, say), the
    corner-influenced region can cover a large fraction of the total
    perimeter, not just an isolated few points near each vertex. This does
    not visibly harm pose fitting in practice (hundreds of contour points
    enter a single least-squares solve, and the corner residuals are not
    systematically one-sided), but it is a real, open thing to tighten --
    e.g. via a proper sub-pixel contour extraction (marching squares)
    instead of a single local-gradient step -- rather than a claim this
    function currently makes good on for every point.
    """
    interior_sdt = signed_distance_transform(mask)  # this mask's OWN sdt, for the sub-pixel correction below
    eroded = binary_erosion(mask, iterations=1, border_value=0)
    boundary = mask & ~eroded
    rows, cols = np.nonzero(boundary)
    if rows.size == 0:
        return np.zeros((0, 3))

    h, w = mask.shape
    rows_c = np.clip(rows, 1, h - 2)
    cols_c = np.clip(cols, 1, w - 2)
    grad_y = (interior_sdt[rows_c + 1, cols_c] - interior_sdt[rows_c - 1, cols_c]) / 2.0
    grad_x = (interior_sdt[rows_c, cols_c + 1] - interior_sdt[rows_c, cols_c - 1]) / 2.0
    grad_norm = np.hypot(grad_x, grad_y)
    grad_norm = np.where(grad_norm < 1e-9, 1.0, grad_norm)
    grad_x, grad_y = grad_x / grad_norm, grad_y / grad_norm

    sdt_val = interior_sdt[rows, cols]  # negative (roughly -1) for these interior-ring pixels
    px = cols.astype(np.float64) + 0.5 - sdt_val * grad_x
    py = rows.astype(np.float64) + 0.5 - sdt_val * grad_y

    # depth comes from the ORIGINAL (integer) pixel, not a bilinear sample at
    # the sub-pixel corrected location: interpolating the depth buffer there
    # would blend in a neighboring BACKGROUND pixel's depth (np.inf),
    # contaminating the result. Depth varies slowly and smoothly near a
    # silhouette edge relative to a sub-pixel lateral shift, so using the
    # pixel-center depth here is an accurate approximation.
    d = depth[rows, cols]
    finite = np.isfinite(d) & (d > 0)
    px, py, d = px[finite], py[finite], d[finite]
    if px.size == 0:
        return np.zeros((0, 3))

    fx = camera.focal_length_px
    cx, cy = camera.width_px / 2.0, camera.height_px / 2.0
    cam_x = (px - cx) * d / fx
    cam_y = -(py - cy) * d / fx
    cam_z = -d
    cam_pts = np.stack([cam_x, cam_y, cam_z], axis=1)

    view = camera.view_matrix()
    r_view, t_view = view[:3, :3], view[:3, 3]
    world_pts = (cam_pts - t_view) @ r_view  # world = R_view^T @ (cam - t_view)
    body_pts = (world_pts - translation) @ rotation  # body = R^T @ (world - t)
    return body_pts


def _axis_angle_to_matrix(rvec: np.ndarray) -> np.ndarray:
    """Exponential map so(3) -> SO(3), the local rotation update each
    Gauss-Newton step applies on top of the current pose."""
    theta = np.linalg.norm(rvec)
    if theta < 1e-12:
        return np.eye(3)
    axis = rvec / theta
    k = np.array([[0, -axis[2], axis[1]], [axis[2], 0, -axis[0]], [-axis[1], axis[0], 0]])
    return np.eye(3) + np.sin(theta) * k + (1 - np.cos(theta)) * (k @ k)


def _project_and_sample(
    body_pts: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: Camera, sdt: np.ndarray
) -> np.ndarray:
    world_pts = body_pts @ rotation.T + translation
    pixel_pts = camera.project(world_pts)
    return _sample_sdt(sdt, pixel_pts)


def _numeric_pose_jacobian(
    sdt: np.ndarray, body_pts: np.ndarray, rotation: np.ndarray, translation: np.ndarray, camera: Camera
) -> np.ndarray:
    """(N, 6) Jacobian of the SDT residual with respect to a local
    (rvec, translation) perturbation of the FIXED body-frame landmark set,
    by central differences. Six columns is cheap enough that autodiff
    would be overkill here."""
    eps = 1e-5
    n = body_pts.shape[0]
    jac = np.zeros((n, 6))

    def residual_at(rvec_delta, t_delta):
        r = _axis_angle_to_matrix(rvec_delta) @ rotation
        t = translation + t_delta
        return _project_and_sample(body_pts, r, t, camera, sdt)

    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        jac[:, i] = (residual_at(d, np.zeros(3)) - residual_at(-d, np.zeros(3))) / (2 * eps)
    for i in range(3):
        d = np.zeros(3)
        d[i] = eps
        jac[:, 3 + i] = (residual_at(np.zeros(3), d) - residual_at(np.zeros(3), -d)) / (2 * eps)
    return jac


@dataclass(frozen=True)
class PoseRefineResult:
    rotation: np.ndarray  # (3, 3)
    translation: np.ndarray  # (3,)
    covariance_6x6: np.ndarray  # local (rvec, translation) covariance from J^T J at the optimum
    n_iterations: int
    final_cost: float


def refine_pose_sdt(
    observed_mask: np.ndarray,
    mesh: Mesh,
    camera: Camera,
    rotation_init: np.ndarray,
    translation_init: np.ndarray,
    max_iter: int = 30,
    step_eps: float = 1e-7,
) -> PoseRefineResult:
    """Gauss-Newton refinement of a 6-DoF pose against an observed
    silhouette. Returns the refined pose plus a 6x6 covariance estimate
    (from `sigma^2 (J^T J)^-1`) that becomes the input to the Fisher-
    information observability atlas -- this covariance must come from an
    actual fit, not be assumed.

    Uses every extracted contour point every iteration, deliberately not a
    random subsample: since the Jacobian only reprojects a FIXED point set
    (no re-rendering involved, see `_numeric_pose_jacobian`), there is no
    cost benefit to subsampling, and re-drawing a fresh random subset each
    outer iteration was found to inject enough noise into the linearized
    problem to stall convergence -- effectively turning Gauss-Newton into
    something like noisy SGD.
    """
    sdt = signed_distance_transform(observed_mask)
    rotation, translation = rotation_init.copy(), translation_init.copy()

    cost_history = []
    n_iters_run = 0
    for _ in range(max_iter):
        mask, depth = rasterize_cpu(mesh, rotation, translation, camera)
        if mask.sum() < 10:
            break
        body_pts = backproject_contour_points(mask, depth, rotation, translation, camera)
        if body_pts.shape[0] < 6:
            break

        residual = _project_and_sample(body_pts, rotation, translation, camera, sdt)
        jac = _numeric_pose_jacobian(sdt, body_pts, rotation, translation, camera)

        jtj = jac.T @ jac + 1e-8 * np.eye(6)
        jtr = jac.T @ residual
        try:
            delta = np.linalg.solve(jtj, jtr)
        except np.linalg.LinAlgError:
            break

        rotation = _axis_angle_to_matrix(-delta[:3]) @ rotation
        translation = translation - delta[3:]
        n_iters_run += 1

        cost_history.append(float(residual @ residual))
        if np.linalg.norm(delta) < step_eps:
            break

    mask, depth = rasterize_cpu(mesh, rotation, translation, camera)
    body_pts = backproject_contour_points(mask, depth, rotation, translation, camera)
    final_residual = _project_and_sample(body_pts, rotation, translation, camera, sdt)
    final_cost = float(final_residual @ final_residual)

    if body_pts.shape[0] >= 6:
        jac = _numeric_pose_jacobian(sdt, body_pts, rotation, translation, camera)
        dof = max(jac.shape[0] - 6, 1)
        sigma2 = final_cost / dof
        try:
            cov = sigma2 * np.linalg.inv(jac.T @ jac + 1e-8 * np.eye(6))
        except np.linalg.LinAlgError:
            cov = np.full((6, 6), np.nan)
    else:
        cov = np.full((6, 6), np.nan)

    return PoseRefineResult(
        rotation=rotation,
        translation=translation,
        covariance_6x6=cov,
        n_iterations=n_iters_run,
        final_cost=final_cost,
    )


def rasterize_mask(mesh: Mesh, rotation: np.ndarray, translation: np.ndarray, camera: Camera) -> np.ndarray:
    """Convenience wrapper: silhouette only, via the CPU rasterizer (used
    both to generate a synthetic 'observed' frame in tests and by any
    initialization stage)."""
    mask, _depth = rasterize_cpu(mesh, rotation, translation, camera)
    return mask
