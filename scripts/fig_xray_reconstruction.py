"""Figure: a real XDen-1K object, all the way through the pipeline. A pair
of pliers (steel jaws, the exact case the density-recalibration audit is
about) -- the raw X-ray attenuation slice, the dataset's own formula turned
into a density slice, and this project's NIST-XCOM recalibration, side by
side with the resulting mass numbers.
"""
from __future__ import annotations

import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
from figstyle import INK, INK_MUTED, apply_style, sequential_cmap, save  # noqa: E402

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.data.xden import (  # noqa: E402
    load_metadata,
    load_object,
    recalibrated_moments_from_object,
    register_volume_to_mesh,
)
from ballast.data.xden_calib import dataset_formula_density_kg_m3, recalibrate_density_kg_m3  # noqa: E402

XDEN_META_ROOT = REPO_ROOT / "data" / "xden1k"
XDEN_DATASET_ROOT = XDEN_META_ROOT / "dataset"
OBJECT_ID = 4  # Pliers -- ferrous jaws; registers to the mesh at 0.83 IoU, well above the 0.7 accept gate


def _profile_slice(volume: np.ndarray, occ: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """A slice through the object's LONGEST axis rather than a fixed
    middle index perpendicular to it -- pliers are thin and elongated, so
    a cross-section perpendicular to the long axis is a near-empty sliver
    (found directly: the first attempt showed almost nothing). Slice
    through the shortest of the two remaining axes instead, so the image
    plane contains the full length of the object."""
    coords = np.argwhere(occ)
    bbox_min, bbox_max = coords.min(axis=0), coords.max(axis=0)
    extents = bbox_max - bbox_min + 1
    axis_long = int(np.argmax(extents))
    other_axes = [a for a in range(3) if a != axis_long]
    axis_slice = min(other_axes, key=lambda a: extents[a])
    idx = int((bbox_min[axis_slice] + bbox_max[axis_slice]) // 2)

    index: list = [slice(None)] * 3
    index[axis_slice] = idx
    vol_slice = volume[tuple(index)]
    occ_slice = occ[tuple(index)]

    occ_coords = np.argwhere(occ_slice)
    pad = 4
    lo = np.maximum(occ_coords.min(axis=0) - pad, 0)
    hi = np.minimum(occ_coords.max(axis=0) + pad + 1, occ_slice.shape)
    crop = (slice(lo[0], hi[0]), slice(lo[1], hi[1]))
    return vol_slice[crop], occ_slice[crop]


def main() -> None:
    apply_style()
    out_dir = REPO_ROOT / "assets" / "figures"
    out_dir.mkdir(parents=True, exist_ok=True)

    metadata = load_metadata(XDEN_META_ROOT)
    obj = load_object(XDEN_DATASET_ROOT, OBJECT_ID, metadata=metadata)
    registration, mesh_occ, pitch = register_volume_to_mesh(obj.lac_volume, obj.mesh)
    result = recalibrated_moments_from_object(obj, registration, mesh_occ, pitch)

    print(f"category: {obj.category}   registration IoU: {registration.iou:.3f}")
    print(f"mass (dataset formula):  {result.mass_dataset_kg * 1000.0:.1f} g")
    print(f"mass (recalibrated):     {result.mass_recalibrated_kg * 1000.0:.1f} g")
    print(f"overestimate factor:     {result.mass_overestimate_factor:.2f}x")

    from scipy.ndimage import affine_transform

    perm, scale, translation = registration.permutation, registration.scale, registration.translation
    inv_perm = perm.T
    matrix = inv_perm / scale
    offset = -inv_perm @ (translation / scale)
    resampled_lac = affine_transform(
        obj.lac_volume, matrix=matrix, offset=offset, output_shape=mesh_occ.shape, order=1, cval=0.0
    )

    resampled_slice, occ_slice = _profile_slice(resampled_lac, mesh_occ)

    threshold = 0.02 * max(resampled_lac.max(), 1e-12)
    occupied = resampled_slice > threshold
    ds_density_slice = np.zeros_like(resampled_slice)
    ds_density_slice[occupied] = np.array(
        [dataset_formula_density_kg_m3(v) for v in resampled_slice[occupied]]
    )
    re_density_slice = np.zeros_like(resampled_slice)
    re_density_slice[occupied] = np.array(
        [recalibrate_density_kg_m3(v, obj.category) for v in resampled_slice[occupied]]
    )

    import matplotlib.pyplot as plt

    cmap = sequential_cmap()
    fig, axes = plt.subplots(1, 3, figsize=(12.5, 5.0))

    axes[0].imshow(np.where(occ_slice, resampled_slice, np.nan).T, cmap=cmap, origin="lower")
    axes[0].set_title("raw X-ray attenuation\n(linear attenuation coeff.)", fontsize=10.5, color=INK)

    vmax = max(ds_density_slice.max(), re_density_slice.max())
    axes[1].imshow(np.where(occ_slice, ds_density_slice, np.nan).T, cmap=cmap, origin="lower", vmax=vmax)
    axes[1].set_title("dataset's own formula\n$\\rho = \\mu / 0.17$", fontsize=10.5, color=INK)

    axes[2].imshow(np.where(occ_slice, re_density_slice, np.nan).T, cmap=cmap, origin="lower", vmax=vmax)
    axes[2].set_title("recalibrated\n(NIST XCOM, per material)", fontsize=10.5, color=INK)

    for ax in axes:
        ax.axis("off")
        ax.set_aspect("equal")

    fig.suptitle(
        f"Real XDen-1K pliers (object {OBJECT_ID}): the dataset's own density formula "
        f"overstates this object's mass {result.mass_overestimate_factor:.1f}x",
        fontsize=12,
        color=INK,
        y=0.98,
    )
    fig.text(
        0.5,
        0.02,
        f"registration IoU {registration.iou:.2f}   ·   "
        f"dataset-formula mass {result.mass_dataset_kg * 1000.0:.0f} g   ·   "
        f"recalibrated mass {result.mass_recalibrated_kg * 1000.0:.0f} g",
        ha="center",
        fontsize=9.5,
        color=INK_MUTED,
    )

    save(fig, str(out_dir / "xray_reconstruction.png"))
    print("done")


if __name__ == "__main__":
    main()
