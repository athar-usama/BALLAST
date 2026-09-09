"""Recalibrating XDen-1K's density fields.

The dataset converts X-ray attenuation to density with a single constant,
`rho = mu / 0.17` (mu/rho = 0.17 cm^2/g), which is accurate for water,
polymers, and aluminium -- and wrong by 2.2x at 100 keV, and roughly 5.8x
at the ~55 keV effective spectrum of a 100 kV tube (the dataset's own
acquisition setting), for iron and steel. The dataset's own category list
is dominated by exactly the tools this breaks: pliers, hammers, scissors,
clamps. Under the published formula, a steel pair of pliers comes out
denser than lead.

This module replaces that single constant with a per-part classification
against `ballast.rattlebox.materials`' NIST-derived mass-attenuation
table, plus a mass sanity gate that flags the (rare, informative) parts
where even the corrected estimate looks implausible for the object's
category.
"""

from __future__ import annotations

from dataclasses import dataclass

from ballast.rattlebox.materials import MASS_ATTENUATION_CM2_PER_G, MATERIALS, classify_by_attenuation

# category -> plausible material shortlist, used as a weak prior for
# classification (an object's category narrows which materials are even
# physically plausible for its parts, without hard-coding an answer)
CATEGORY_MATERIAL_PRIORS: dict[str, list[str]] = {
    "Pliers": ["steel", "plastic", "aluminium"],
    "Hammer": ["steel", "wood", "plastic"],
    "Scissors": ["steel", "plastic"],
    "Toy": ["plastic", "wood", "aluminium"],
    "Brush": ["plastic", "wood"],
}
DEFAULT_MATERIAL_CANDIDATES = [
    "air",
    "foam",
    "wood",
    "plastic",
    "water",
    "glass",
    "aluminium",
    "steel",
    "lead",
]

# a category's typical total mass, in grams -- a loose sanity range, not a
# precise reference, used only to flag implausible reconstructions
CATEGORY_TYPICAL_MASS_G: dict[str, tuple[float, float]] = {
    "Pliers": (100.0, 350.0),
    "Hammer": (150.0, 700.0),
    "Scissors": (30.0, 200.0),
    "Toy": (10.0, 500.0),
    "Brush": (10.0, 150.0),
}


def dataset_formula_density_kg_m3(lac_cm_inv: float) -> float:
    """The dataset's OWN conversion, `rho[g/cm^3] = mu / 0.17`, reproduced
    exactly (including its units) so the recalibration below can be
    compared against it directly, in the README's density-audit figure."""
    rho_g_cm3 = lac_cm_inv / 0.17
    return rho_g_cm3 * 1000.0  # g/cm^3 -> kg/m^3


def recalibrate_density_kg_m3(lac_cm_inv: float, category: str, effective_kev: int = 60) -> float:
    """Classify the material producing this attenuation (using the
    category as a weak prior over candidates) and return the TABULATED
    density for that material, rather than the dataset's single-constant
    conversion."""
    candidates = CATEGORY_MATERIAL_PRIORS.get(category, DEFAULT_MATERIAL_CANDIDATES)
    material = classify_by_attenuation(lac_cm_inv, candidates, effective_kev=effective_kev)
    return MATERIALS[material]


def density_overestimate_factor(material: str, effective_kev: int = 60) -> float:
    """How much the dataset's own formula overestimates density for a
    given material, at a given effective X-ray energy -- the number this
    project's density-audit figure reports per material (2.2x at 100 keV,
    ~5.8x at ~55-60 keV for iron/steel, per the project plan)."""
    mu_over_rho = MASS_ATTENUATION_CM2_PER_G[material][effective_kev]
    rho_g_cm3 = MATERIALS[material] / 1000.0
    true_lac = rho_g_cm3 * mu_over_rho
    dataset_density = dataset_formula_density_kg_m3(true_lac)
    true_density = MATERIALS[material]
    return dataset_density / true_density


@dataclass(frozen=True)
class MassSanityResult:
    passed: bool
    implied_mass_g: float
    expected_range_g: tuple[float, float] | None


def mass_sanity_gate(implied_mass_g: float, category: str, tolerance_factor: float = 3.0) -> MassSanityResult:
    """A 3x check against category-typical mass: pass/fail plus the range
    used, so the pass rate across the dataset is itself a reportable
    number, not a silent filter."""
    expected = CATEGORY_TYPICAL_MASS_G.get(category)
    if expected is None:
        return MassSanityResult(passed=True, implied_mass_g=implied_mass_g, expected_range_g=None)
    lo, hi = expected
    passed = (implied_mass_g >= lo / tolerance_factor) and (implied_mass_g <= hi * tolerance_factor)
    return MassSanityResult(passed=passed, implied_mass_g=implied_mass_g, expected_range_g=expected)
