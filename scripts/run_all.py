"""Reproduce every figure in the README from fixed seeds, in order.

Real-data figures (the XDen-1K reconstruction, the scale ladder's XDen-1K
and Bennu entries) are skipped automatically if `data/xden1k` has not been
downloaded (see `scripts/download_xden1k.py`) -- everything else runs from
this repository alone.
"""
from __future__ import annotations

import subprocess
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
SCRIPTS_DIR = Path(__file__).resolve().parent

FIGURE_SCRIPTS = [
    "fig_chirality_twins.py",
    "fig_dzhanibekov.py",
    "fig_sharp_bound.py",
    "fig_observability_atlas.py",
    "fig_active_interrogation.py",
    "fig_scale_ladder.py",
    "fig_bennu.py",
    "fig_xray_reconstruction.py",
]

XDEN_DEPENDENT = {"fig_xray_reconstruction.py", "fig_scale_ladder.py"}


def main() -> None:
    xden_available = (REPO_ROOT / "data" / "xden1k" / "metadata.json").exists()

    for script in FIGURE_SCRIPTS:
        if script in XDEN_DEPENDENT and not xden_available:
            print(f"skipping {script}: data/xden1k not found (see scripts/download_xden1k.py)")
            continue
        print(f"--- running {script} ---")
        result = subprocess.run([sys.executable, str(SCRIPTS_DIR / script)], cwd=REPO_ROOT)
        if result.returncode != 0:
            print(f"FAILED: {script}", file=sys.stderr)
            sys.exit(result.returncode)

    print("--- running test suite ---")
    result = subprocess.run([sys.executable, "-m", "pytest", "-q"], cwd=REPO_ROOT)
    sys.exit(result.returncode)


if __name__ == "__main__":
    main()
