"""A small material density table, used both to build synthetic Rattlebox
interiors and, later, as the corrected reference table for recalibrating
XDen-1K's density fields (which convert X-ray attenuation to density with a
single constant that is wrong by 2-6x for the ferrous tools that dominate
that dataset -- see the project plan). Densities are room-temperature,
kg/m^3, and are deliberately a short, physically real list rather than a
continuous range, matching the "few discrete materials" prior that C3
(prior-constrained interior solving) depends on to be a real constraint.
"""

from __future__ import annotations

# kg/m^3, standard reference values
MATERIALS: dict[str, float] = {
    "air": 1.2,
    "foam": 30.0,
    "cork": 240.0,
    "wood": 650.0,
    "plastic": 950.0,
    "water": 1000.0,
    "sand": 1600.0,
    "glass": 2500.0,
    "aluminium": 2700.0,
    "steel": 7850.0,
    "brass": 8500.0,
    "copper": 8960.0,
    "lead": 11340.0,
}

# NIST XCOM mass attenuation coefficients (cm^2/g) for common tube-voltage
# X-ray spectra, used to recalibrate XDen-1K's LAC-to-density conversion
# (which assumes a single mu/rho = 0.17, correct for water/polymers/
# aluminium but wrong by 2.2x at 100 keV and ~5.8x at ~55 keV effective
# spectrum for iron/steel, which dominates that dataset's tool categories)
MASS_ATTENUATION_CM2_PER_G: dict[str, dict[int, float]] = {
    "water": {100: 0.1707, 60: 0.2059},
    "plastic": {100: 0.1700, 60: 0.2059},  # polymer, close to water
    "aluminium": {100: 0.1704, 60: 0.2778},
    "iron": {100: 0.3717, 60: 1.205},
    "steel": {100: 0.3717, 60: 1.205},  # steel is mostly iron
    "copper": {100: 0.4584, 60: 1.593},
    "brass": {100: 0.4400, 60: 1.500},  # copper-zinc alloy, close to copper
    "lead": {100: 5.549, 60: 5.021},
}


def density_for(material: str) -> float:
    try:
        return MATERIALS[material]
    except KeyError as exc:
        raise KeyError(f"unknown material '{material}', known: {sorted(MATERIALS)}") from exc


def classify_by_attenuation(lac_cm_inv: float, rho_candidates: list[str], effective_kev: int = 60) -> str:
    """Given a reconstructed linear attenuation coefficient (cm^-1) and a
    shortlist of candidate materials (e.g. from a category prior), pick the
    material whose predicted LAC (density * mass-attenuation-coefficient)
    is closest. This is the corrected replacement for a single-constant
    mu/rho conversion."""
    best_material, best_err = None, float("inf")
    for material in rho_candidates:
        rho = MATERIALS.get(material)
        mu_over_rho = MASS_ATTENUATION_CM2_PER_G.get(material, {}).get(effective_kev)
        if rho is None or mu_over_rho is None:
            continue
        # MATERIALS is kg/m^3; mu/rho tables are in cm^2/g, i.e. expect g/cm^3
        rho_g_cm3 = rho / 1000.0
        predicted_lac = rho_g_cm3 * mu_over_rho
        err = abs(predicted_lac - lac_cm_inv)
        if err < best_err:
            best_material, best_err = material, err
    if best_material is None:
        raise ValueError(f"no valid candidates with attenuation data among {rho_candidates}")
    return best_material
