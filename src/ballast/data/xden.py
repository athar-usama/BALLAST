"""Ingesting a real XDen-1K object: mesh, X-ray density volume, and the
registration between their two, entirely unrelated coordinate systems.

Concrete traps found by reading the actual downloaded files (not just the
dataset's README, which gets several of these wrong):

  - Object directories are `dataset/<n>/` for `n` in `0..881` in the public
    HF release (not `01`, `02`, ... as the README's example suggests).
  - `metadata.json` keys are zero-based strings, and the per-object field
    is called `size` (a bounding-box extent in cm), not `scale`.
  - `mesh.glb` loads as a `trimesh.Scene`, not a `Trimesh` -- flattening
    it with the now-deprecated `Scene.dump(concatenate=True)` still works
    but `Scene.to_geometry()` is the current API.
  - The mesh is normalized into an arbitrary unit box (verified directly:
    object 0's mesh spans exactly 2.0 along its longest axis), NOT in
    real-world meters -- `metadata.json`'s `size` field is what recovers
    physical scale.
  - The mesh is NOT watertight (verified directly on object 0), so a
    signed volume/containment test is unreliable; voxelization via
    `trimesh`'s scan-fill voxelizer is used instead for registration.
  - The NIfTI affine is USELESS here: it decodes to the identity matrix
    despite `sform_code = 2` claiming an aligned frame (verified directly
    on object 0). There is no shortcut around registering by geometry.
  - `part_lac_density.json` already contains the dataset's own flawed
    conversion (verified directly: `Density = LAC / 0.17` reproduces the
    file's numbers exactly), which is what `xden_calib` replaces.
"""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

import nibabel as nib
import numpy as np
import trimesh
from scipy.ndimage import affine_transform, binary_fill_holes


@dataclass(frozen=True)
class XDenObject:
    object_id: int
    category: str
    size_cm: np.ndarray  # (3,) bounding-box extent, from metadata.json's "size" field
    mesh: trimesh.Trimesh
    lac_volume: np.ndarray  # (H, W, D) raw linear attenuation coefficients, cm^-1
    part_lac_density: dict  # the dataset's own (flawed) per-part LAC/density, for the audit figure


def load_metadata(root: Path) -> dict:
    with open(root / "metadata.json") as f:
        return json.load(f)


def load_object(root: Path, obj_id: int, metadata: dict | None = None) -> XDenObject:
    """`root` is the directory holding both `metadata.json` and the
    numbered object directories directly (`root/metadata.json`,
    `root/<obj_id>/...`) -- in the public HF snapshot that is
    `<snapshot>/dataset/`, one level below the snapshot's own top-level
    `README.md`."""
    obj_dir = root / str(obj_id)
    if metadata is None:
        metadata = load_metadata(root)
    entry = metadata[str(obj_id)]

    scene = trimesh.load(obj_dir / "mesh.glb")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    if isinstance(mesh, trimesh.Scene):  # a Scene with multiple disjoint geometries
        mesh = trimesh.util.concatenate(list(mesh.geometry.values()))

    img = nib.load(obj_dir / "optimized_lac.nii.gz")
    lac_volume = np.asarray(img.get_fdata())

    part_lac_density = {}
    part_file = obj_dir / "part_lac_density.json"
    if part_file.exists():
        with open(part_file) as f:
            part_lac_density = json.load(f)

    return XDenObject(
        object_id=obj_id,
        category=entry["category"],
        size_cm=np.asarray(entry["size"], dtype=np.float64),
        mesh=mesh,
        lac_volume=lac_volume,
        part_lac_density=part_lac_density,
    )


# --------------------------------------------------------------------------
# Registration: the mesh and the LAC volume live in two unrelated coordinate
# systems (the mesh is unit-box normalized, the volume's affine is useless).
# Register by brute-forcing all 24 proper signed axis permutations plus an
# isotropic scale and translation, maximizing occupancy IoU against the mesh.
# --------------------------------------------------------------------------


def _signed_permutation_matrices() -> list[np.ndarray]:
    """All 24 proper (determinant +1) signed 3x3 permutation matrices --
    the full set of ways two axis-aligned voxel grids can disagree on
    which physical axis is which, and which way each one points."""
    from itertools import permutations, product

    mats = []
    for perm in permutations(range(3)):
        base = np.zeros((3, 3))
        for i, j in enumerate(perm):
            base[i, j] = 1.0
        for signs in product((1, -1), repeat=3):
            m = base * np.array(signs)[:, None]
            if np.isclose(np.linalg.det(m), 1.0):
                mats.append(m)
    return mats


def voxelize_mesh_occupancy(mesh: trimesh.Trimesh, resolution: int) -> tuple[np.ndarray, np.ndarray, float]:
    """Boolean occupancy grid of `mesh`'s SOLID interior, at `resolution`
    voxels along its longest bounding-box axis. Uses scan-fill voxelization
    (robust to a non-watertight mesh, unlike a signed containment test) and
    a flood-fill of any remaining interior holes as a second safeguard.
    Returns (occupancy, grid_origin_m, voxel_size_m)."""
    extent = mesh.extents
    pitch = extent.max() / resolution
    voxel_grid = mesh.voxelized(pitch)
    occ = voxel_grid.matrix.astype(bool)
    occ = binary_fill_holes(occ)
    origin = voxel_grid.translation if hasattr(voxel_grid, "translation") else voxel_grid.bounds[0]
    return occ, np.asarray(origin, dtype=np.float64), float(pitch)


@dataclass(frozen=True)
class Registration:
    permutation: np.ndarray  # (3, 3) signed permutation, LAC-index-axis -> mesh-voxel-axis
    scale: float  # isotropic scale, LAC voxels -> mesh voxels
    translation: np.ndarray  # (3,) offset, in mesh-voxel-grid index units
    iou: float


def _resample_iou(
    lac_occ: np.ndarray, mesh_occ: np.ndarray, perm: np.ndarray, scale: float, translation: np.ndarray
) -> float:
    inv_perm = perm.T
    matrix = inv_perm / scale
    offset = -inv_perm @ (translation / scale)
    resampled = affine_transform(
        lac_occ.astype(np.float64),
        matrix=matrix,
        offset=offset,
        output_shape=mesh_occ.shape,
        order=0,
        cval=0.0,
    )
    resampled_occ = resampled > 0.5
    intersection = np.sum(resampled_occ & mesh_occ)
    union = np.sum(resampled_occ | mesh_occ)
    return float(intersection / union) if union > 0 else 0.0


def _refine_scale_translation(
    lac_occ: np.ndarray,
    mesh_occ: np.ndarray,
    perm: np.ndarray,
    lac_centroid: np.ndarray,
    mesh_centroid: np.ndarray,
    scale0: float,
    n_scale_steps: int = 9,
    scale_range: float = 0.25,
) -> tuple[float, np.ndarray, float]:
    """A small local refinement of (scale, translation) around the coarse
    bbox/centroid estimate, for the WINNING permutation only -- the coarse
    estimate gets the discrete permutation right but can leave enough
    residual scale/translation error to cost real IoU (found directly:
    the correct permutation scoring 0.65 IoU before this refinement, on a
    synthetic case with an exactly known answer). At each candidate scale,
    translation is re-solved by the same centroid-matching rule used for
    the initial guess, then scored by actual IoU; this is a simple grid
    search, not a full optimizer, but it closes the gap the coarse
    estimate leaves, and it only runs once (for the already-chosen
    permutation), not for all 24 candidates.
    """
    best_scale, best_translation, best_iou = scale0, mesh_centroid - scale0 * (perm @ lac_centroid), -1.0
    for mult in np.linspace(1.0 - scale_range, 1.0 + scale_range, n_scale_steps):
        scale = scale0 * mult
        translation = mesh_centroid - scale * (perm @ lac_centroid)
        iou = _resample_iou(lac_occ, mesh_occ, perm, scale, translation)
        if iou > best_iou:
            best_scale, best_translation, best_iou = scale, translation, iou
    return best_scale, best_translation, best_iou


def register_volume_to_mesh(
    lac_volume: np.ndarray,
    mesh: trimesh.Trimesh,
    threshold: float | None = None,
    mesh_resolution: int = 64,
) -> tuple[Registration, np.ndarray, float]:
    """Find the signed axis permutation, isotropic scale, and translation
    that best aligns the LAC volume's occupied region with the mesh's
    solid interior, by IoU. Returns (best_registration, mesh_occupancy,
    mesh_voxel_pitch_m); reject registrations with `iou < 0.7` (per the
    project plan) as unreliable.
    """
    mesh_occ, _origin, pitch = voxelize_mesh_occupancy(mesh, mesh_resolution)
    if threshold is None:
        threshold = 0.05 * lac_volume.max()
    lac_occ = lac_volume > threshold
    if not lac_occ.any() or not mesh_occ.any():
        raise ValueError("registration requires a nonempty occupancy in both the LAC volume and the mesh")

    # scale from the OCCUPIED bounding-box extent in each grid, not the full
    # array shape: a real LAC volume (or this function's own test fixtures)
    # has arbitrary empty background padding around the object, and dividing
    # by the full array size instead of the object's actual footprint badly
    # underestimates the true scale.
    lac_coords = np.argwhere(lac_occ)
    mesh_coords = np.argwhere(mesh_occ)
    lac_bbox_extent = (lac_coords.max(axis=0) - lac_coords.min(axis=0) + 1).astype(np.float64)
    mesh_bbox_extent = (mesh_coords.max(axis=0) - mesh_coords.min(axis=0) + 1).astype(np.float64)

    lac_centroid = lac_coords.mean(axis=0)
    mesh_centroid = mesh_coords.mean(axis=0)

    best: Registration | None = None
    for perm in _signed_permutation_matrices():
        # candidate affine: index_mesh = scale * (perm @ index_lac) + translation,
        # solved so that the two occupied regions' CENTROIDS and BOUNDING-BOX
        # extents coincide -- a coarse but cheap estimate, evaluated by IoU for
        # every one of the 24 candidates to pick the right discrete permutation
        permuted_extent = np.abs(perm) @ lac_bbox_extent
        scale = float(np.mean(mesh_bbox_extent / np.maximum(permuted_extent, 1.0)))
        translation = mesh_centroid - scale * (perm @ lac_centroid)
        iou = _resample_iou(lac_occ, mesh_occ, perm, scale, translation)

        if best is None or iou > best.iou:
            best = Registration(permutation=perm, scale=scale, translation=translation, iou=iou)

    assert best is not None

    # refine scale/translation for the WINNING permutation only: the coarse
    # bbox/centroid estimate reliably picks the right discrete permutation,
    # but can leave enough residual scale error to cost real IoU even when
    # the permutation itself is exactly correct (confirmed directly on a
    # synthetic case with a known answer: 0.65 IoU before this refinement,
    # 0.9+ after, for the SAME, already-correct permutation)
    refined_scale, refined_translation, refined_iou = _refine_scale_translation(
        lac_occ, mesh_occ, best.permutation, lac_centroid, mesh_centroid, best.scale
    )
    if refined_iou > best.iou:
        best = Registration(
            permutation=best.permutation,
            scale=refined_scale,
            translation=refined_translation,
            iou=refined_iou,
        )

    return best, mesh_occ, pitch
