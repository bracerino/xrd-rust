from pymatgen.core import Structure
from xrd_rust_accelerator import XRDCalculatorRust

# Load structure and calculate powder XRD pattern
structure = Structure.from_file("structure.cif")
calc = XRDCalculatorRust(wavelength="Cuka")
pattern = calc.get_pattern(structure, scaled=False, two_theta_range=(5, 70))

# Save to file
with open("xrd_pattern.csv", 'w') as f:
    f.write("2theta,intensity,hkl\n")
    for i in range(len(pattern.x)):
        hkl = str([tuple(h['hkl']) for h in pattern.hkls[i]])
        f.write(f"{pattern.x[i]},{pattern.y[i]},{hkl}\n")
