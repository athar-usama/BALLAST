"""Precompute the data the HTML explorer embeds: for k = 0..10, the sharp
certified bound on a fixed interior query given only the FIRST k of the 10
raw moments (row 0 is total mass; rows 1-3 are the first-moment/COM-bearing
rows; rows 4-9 are the second-moment/inertia-bearing rows) -- an honest,
exactly-computed demonstration of T1 (the interval narrows monotonically
and every extremal interior stays bang-bang) that needs no video or
regime-specific machinery to show. k=0 is the fully uninformed case (shape
and material bound only, no motion at all); k=10 is everything a full set
of measured moments can ever pin down.

Output is a single JSON literal written into `explorer/index.html`,
replacing the `__EXPLORER_DATA__` placeholder, so the page stays one
self-contained file with no separate data fetch.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
from scipy.optimize import linprog

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))

from ballast.moments.bounds import half_space_query  # noqa: E402
from ballast.moments.operator import build_moment_operator  # noqa: E402
from ballast.moments.voxelgrid import VoxelGrid  # noqa: E402

N = 15
RHO_MAX = 8000.0


def main() -> None:
    grid = VoxelGrid(shape=(N, N, N), voxel_size_m=0.011, origin_m=np.array([-0.077, -0.077, -0.077]))
    centers = grid.centers()

    lump_center = np.array([0.022, -0.016, 0.011])
    dist = np.linalg.norm(centers - lump_center[None, :], axis=1)
    rho_true = np.where(dist < 0.032, RHO_MAX, 0.32 * RHO_MAX)
    r_outer = np.linalg.norm(grid.geometric_center_m() - centers, axis=1)
    outer_mask = r_outer <= 0.073
    rho_true = np.where(outer_mask, rho_true, 0.0)

    a = build_moment_operator(grid)
    b_full = a @ rho_true
    query = half_space_query(grid, normal=np.array([1.0, 0.0, 0.0]), offset_m=0.0)
    true_value = float(query @ rho_true)

    v = a.shape[1]
    bounds = [(0.0, RHO_MAX)] * v
    mid = N // 2

    steps = []
    for k in range(0, 11):
        a_k, b_k = a[:k], b_full[:k]
        res_max = linprog(-query, A_eq=a_k if k else None, b_eq=b_k if k else None, bounds=bounds, method="highs")
        res_min = linprog(query, A_eq=a_k if k else None, b_eq=b_k if k else None, bounds=bounds, method="highs")
        q_max, q_min = float(-res_max.fun), float(res_min.fun)

        min_slice = (res_min.x.reshape(grid.shape)[:, :, mid] > RHO_MAX / 2).astype(int).tolist()
        max_slice = (res_max.x.reshape(grid.shape)[:, :, mid] > RHO_MAX / 2).astype(int).tolist()
        steps.append(
            {
                "k": k,
                "qmin": q_min,
                "qmax": q_max,
                "min_slice": min_slice,
                "max_slice": max_slice,
            }
        )
        print(f"k={k:2d}  [{q_min:.4f}, {q_max:.4f}]  width={q_max - q_min:.4f}")

    true_slice = (rho_true.reshape(grid.shape)[:, :, mid] > RHO_MAX / 2).astype(int).tolist()
    outer_slice = outer_mask.reshape(grid.shape)[:, :, mid].astype(int).tolist()

    data = {
        "true_value": true_value,
        "no_motion_max": float(query.sum() * RHO_MAX),
        "true_slice": true_slice,
        "outer_slice": outer_slice,
        "grid_n": N,
        "steps": steps,
    }

    template_path = REPO_ROOT / "explorer" / "index.template.html"
    out_path = REPO_ROOT / "explorer" / "index.html"
    html = template_path.read_text(encoding="utf-8")
    placeholder = "/*__EXPLORER_DATA__*/"
    if placeholder not in html:
        raise RuntimeError(f"placeholder {placeholder!r} not found in {template_path}")
    html = html.replace(placeholder, json.dumps(data))
    out_path.write_text(html, encoding="utf-8")
    print(f"wrote {out_path} from {template_path}")


if __name__ == "__main__":
    main()
