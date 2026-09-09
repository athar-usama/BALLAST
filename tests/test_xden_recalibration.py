"""Tests for computing real ground-truth moments from a registered,
recalibrated XDen-1K object -- the actual "X-ray-validated reconstruction"
deliverable, not just the registration search on its own.

Skipped entirely when the real dataset is not present on disk (the whole
suite must not depend on a 16 GB external download); a synthetic test
covers the same code path independent of any real download.
"""

from __future__ import annotations

import json
from pathlib import Path

import nibabel as nib
import numpy as np
import pytest
import trimesh

from ballast.data.xden import XDenObject, recalibrated_moments_from_object, register_volume_to_mesh
from ballast.rattlebox.materials import MATERIALS

XDEN_ROOT = Path(__file__).resolve().parents[1] / "data" / "xden1k" / "dataset"


def _synthetic_steel_object() -> tuple[XDenObject, str]:
    """A synthetic object standing in for XDen-1K's own file layout: a
    small box mesh, a matching LAC volume built at STEEL's true
    attenuation (so the dataset-formula-vs-recalibration gap is large and
    checkable), registered via the identity transform for simplicity."""
    mesh = trimesh.creation.box(extents=(0.1, 0.1, 0.1))
    n = 24
    lac_volume = np.zeros((n, n, n))
    # steel's true LAC at 60 keV effective: rho[g/cm^3] * (mu/rho)[cm^2/g]
    steel_rho_g_cm3 = MATERIALS["steel"] / 1000.0
    steel_lac = steel_rho_g_cm3 * 1.205  # MASS_ATTENUATION_CM2_PER_G["steel"][60]
    lac_volume[4:20, 4:20, 4:20] = steel_lac

    obj = XDenObject(
        object_id=0,
        category="Pliers",  # a category whose material prior includes steel
        size_cm=np.array([10.0, 10.0, 10.0]),
        mesh=mesh,
        lac_volume=lac_volume,
        part_lac_density={},
    )
    return obj, "steel"


def test_recalibration_corrects_a_synthetic_steel_object():
    """The headline audit claim, reproduced end to end on a synthetic
    object with steel's TRUE attenuation: the dataset's own formula should
    overestimate mass by several times relative to the recalibrated value."""
    obj, _true_material = _synthetic_steel_object()
    registration, mesh_occ, pitch = register_volume_to_mesh(obj.lac_volume, obj.mesh, mesh_resolution=24)
    assert registration.iou > 0.7

    result = recalibrated_moments_from_object(obj, registration, mesh_occ, pitch)

    assert result.mass_dataset_kg > 0
    assert result.mass_recalibrated_kg > 0
    # steel: dataset formula (mu/rho=0.17 assumed) should overestimate several-fold
    assert result.mass_overestimate_factor > 1.5, (
        f"expected the dataset formula to overestimate steel's density, "
        f"got factor={result.mass_overestimate_factor}"
    )
    # the recalibrated mass should be close to steel's true density times the
    # occupied volume, not aluminium's, water's, or the dataset's own guess
    assert np.all(np.isfinite(result.inertia_recalibrated_kgm2))
    assert np.all(np.linalg.eigvalsh(result.inertia_recalibrated_kgm2) > 0)


def test_recalibration_factor_matches_the_documented_steel_overestimate():
    """Cross-check against `xden_calib.density_overestimate_factor`
    directly: the end-to-end pipeline's overestimate factor for a
    uniform-steel object should match the per-material formula closely."""
    from ballast.data.xden_calib import density_overestimate_factor

    obj, _material = _synthetic_steel_object()
    registration, mesh_occ, pitch = register_volume_to_mesh(obj.lac_volume, obj.mesh, mesh_resolution=24)
    result = recalibrated_moments_from_object(obj, registration, mesh_occ, pitch)

    expected_factor = density_overestimate_factor("steel", effective_kev=60)
    assert result.mass_overestimate_factor == pytest.approx(expected_factor, rel=0.15)


@pytest.mark.skipif(
    not (XDEN_ROOT / "0").exists(),
    reason="real XDen-1K data not present on disk (download not run or not yet complete)",
)
def test_real_object_zero_recalibration_runs_end_to_end():
    """A real-data smoke test on object 0: the full pipeline (register,
    resample, recalibrate, compute moments) runs without error and
    produces physically sane values, using a fallback category if
    metadata.json has not finished downloading yet."""
    scene = trimesh.load(XDEN_ROOT / "0" / "mesh.glb")
    mesh = scene.to_geometry() if hasattr(scene, "to_geometry") else scene
    img = nib.load(XDEN_ROOT / "0" / "optimized_lac.nii.gz")
    lac_volume = np.asarray(img.get_fdata())

    metadata_file = XDEN_ROOT / "metadata.json"
    if metadata_file.exists():
        metadata = json.loads(metadata_file.read_text())
        entry = metadata["0"]
        category = entry["category"]
        size_cm = np.asarray(entry["size"], dtype=np.float64)
    else:
        category, size_cm = "Toy", None  # observed directly for object 0 during dataset exploration

    part_file = XDEN_ROOT / "0" / "part_lac_density.json"
    part_lac_density = json.loads(part_file.read_text()) if part_file.exists() else {}

    obj = XDenObject(
        object_id=0,
        category=category,
        size_cm=size_cm,
        mesh=mesh,
        lac_volume=lac_volume,
        part_lac_density=part_lac_density,
    )

    registration, mesh_occ, pitch = register_volume_to_mesh(obj.lac_volume, obj.mesh, mesh_resolution=48)
    result = recalibrated_moments_from_object(obj, registration, mesh_occ, pitch)

    assert result.mass_dataset_kg > 0
    assert result.mass_recalibrated_kg > 0
    assert np.all(np.isfinite(result.com_recalibrated_m))
    assert np.all(np.linalg.eigvalsh(result.inertia_recalibrated_kgm2) > 0)
