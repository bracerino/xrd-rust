"""XRD Calculator with Rust acceleration."""

from __future__ import annotations

import math
import os
from typing import TYPE_CHECKING

import numpy as np
import orjson

from pymatgen.analysis.diffraction.core import (
    AbstractDiffractionPatternCalculator,
    DiffractionPattern,
)
from pymatgen.symmetry.analyzer import SpacegroupAnalyzer

try:
    from .xrd_rust_accelerator import (
        get_unique_families_rust,
        calculate_xrd_intensities,
        merge_peaks,
        normalize_intensities,
    )
    HAS_RUST = True
except ImportError:
    HAS_RUST = False

if TYPE_CHECKING:
    from pymatgen.core import Structure

WAVELENGTHS = {
    "CuKa": 1.54184,
    "CuKa2": 1.54439,
    "CuKa1": 1.54056,
    "CuKb1": 1.39222,
    "MoKa": 0.71073,
    "MoKa2": 0.71359,
    "MoKa1": 0.70930,
    "MoKb1": 0.63229,
    "CrKa": 2.29100,
    "CrKa2": 2.29361,
    "CrKa1": 2.28970,
    "CrKb1": 2.08487,
    "FeKa": 1.93735,
    "FeKa2": 1.93998,
    "FeKa1": 1.93604,
    "FeKb1": 1.75661,
    "CoKa": 1.79026,
    "CoKa2": 1.79285,
    "CoKa1": 1.78896,
    "CoKb1": 1.63079,
    "AgKa": 0.560885,
    "AgKa2": 0.563813,
    "AgKa1": 0.559421,
    "AgKb1": 0.497082,
}

with open(
    os.path.join(os.path.dirname(__file__), "atomic_scattering_params.json"),
    "rb",
) as file:
    ATOMIC_SCATTERING_PARAMS = orjson.loads(file.read())


class XRDCalculatorRust(AbstractDiffractionPatternCalculator):

    AVAILABLE_RADIATION = tuple(WAVELENGTHS)

    def __init__(self, wavelength="CuKa", symprec: float = 0, debye_waller_factors=None):
        if isinstance(wavelength, (float, int)):
            self.wavelength = wavelength
        elif isinstance(wavelength, str):
            self.radiation = wavelength
            self.wavelength = WAVELENGTHS[wavelength]
        else:
            raise TypeError(f"wavelength must be float, int, or str")
        
        self.symprec = symprec
        self.debye_waller_factors = debye_waller_factors or {}
        
        if not HAS_RUST:
            raise ImportError(
                "Rust accelerator not installed. Install with:\n"
                "  cd xrd_rust_accelerator && maturin develop --release"
            )

    def get_pattern(self, structure: Structure, scaled=True, two_theta_range=(0, 90)):
        if self.symprec:
            finder = SpacegroupAnalyzer(structure, symprec=self.symprec)
            structure = finder.get_refined_structure()

        wavelength = self.wavelength
        lattice = structure.lattice
        is_hex = lattice.is_hexagonal()

        min_r, max_r = (
            (0, 2 / wavelength)
            if two_theta_range is None
            else [2 * math.sin(math.radians(t / 2)) / wavelength for t in two_theta_range]
        )

        recip_lattice = lattice.reciprocal_lattice_crystallographic
        recip_pts = recip_lattice.get_points_in_sphere([[0, 0, 0]], [0, 0, 0], max_r)
        if min_r:
            recip_pts = [pt for pt in recip_pts if pt[1] >= min_r]

        _zs = []
        _coeffs = []
        _frac_coords = []
        _occus = []
        _dw_factors = []

        for site in structure:
            for sp, occu in site.species.items():
                _zs.append(sp.Z)
                try:
                    c = ATOMIC_SCATTERING_PARAMS[sp.symbol]
                except KeyError:
                    raise ValueError(
                        f"No scattering coefficients for {sp.symbol}"
                    )
                _coeffs.append(c)
                _dw_factors.append(self.debye_waller_factors.get(sp.symbol, 0))
                _frac_coords.append(site.frac_coords.tolist())
                _occus.append(occu)

        hkls = []
        g_hkls = []
        d_hkls_list = []
        
        for hkl, g_hkl, ind, _ in sorted(
            recip_pts, 
            key=lambda i: (i[1], -i[0][0], -i[0][1], -i[0][2])
        ):
            hkl_rounded = [float(round(i)) for i in hkl]
            hkls.append(hkl_rounded)
            g_hkls.append(float(g_hkl))
            d_hkls_list.append(1.0 / g_hkl if g_hkl != 0 else 0.0)

        results = calculate_xrd_intensities(
            hkls=hkls,
            g_hkls=g_hkls,
            wavelength=wavelength,
            frac_coords=_frac_coords,
            atomic_numbers=_zs,
            scattering_coeffs=_coeffs,
            occupancies=_occus,
            dw_factors=_dw_factors,
        )

        two_thetas = []
        intensities = []
        hkls_int = []
        d_hkls_final = []

        for idx, (two_theta, intensity) in enumerate(results):
            if intensity > self.SCALED_INTENSITY_TOL:
                hkl = [int(h) for h in hkls[idx]]
                
                if is_hex:
                    hkl = [hkl[0], hkl[1], -hkl[0] - hkl[1], hkl[2]]
                
                two_thetas.append(two_theta)
                intensities.append(intensity)
                hkls_int.append(hkl)
                d_hkls_final.append(d_hkls_list[idx])

        if len(two_thetas) > 0:
            merged_data = merge_peaks(
                two_thetas,
                intensities,
                hkls_int,
                d_hkls_final,
                self.TWO_THETA_TOL,
            )
            two_thetas, intensities, hkls_groups, d_hkls_final = merged_data
            
            formatted_hkls = []
            for hkl_group in hkls_groups:
                hkl_tuples = [h for h in hkl_group]
                families = get_unique_families_rust(hkl_tuples)
                formatted = [
                    {"hkl": tuple(hkl), "multiplicity": mult}
                    for hkl, mult in families.items()
                ]
                formatted_hkls.append(formatted)
        else:
            formatted_hkls = []

        xrd = DiffractionPattern(
            list(two_thetas),
            list(intensities),
            formatted_hkls,
            list(d_hkls_final),
        )
        
        if scaled and len(intensities) > 0:
            xrd.normalize(mode="max", value=100)
        
        return xrd
