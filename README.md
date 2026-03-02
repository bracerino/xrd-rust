# XRD-Rust
Compute powder X-ray diffraction (XRD) patterns using modified [pymatgen’s XRDCalculator](https://pymatgen.org/pymatgen.analysis.diffraction.html), with the performance-critical routines reimplemented in Rust, achieving an average ⚡ 4–6× speedup. The larger and more complex your structure (more peaks and atoms), the greater the speedup gains.

The XRD-Rust follows the same notation as the XRDCalculator, with the only change being the renamed class, XRDCalculatorRust. Due to this, it can be easily implemented into existing workflows that use the original pymatgen package for powder XRD pattern calculations. Simply import the XRD-Rust package and replace the class name (see also example below).



Benchmark results (2θ range = 2–60° (Mo radiation)) on two large crystallographic datasets demonstrate the following acceleration:

- COD (515 181 structures): ⚡ 6.1 ± 4.6× average speedup, up to 719× faster (1437 min (original pymatgen implementation) → 2 min (Rust-accelerated))

- MC3D (33 142 structures): ⚡ 4.7 ± 1.6× average speedup, up to 25× faster (34.9 s → 1.4 s)

Full benchmarking details are available at:
📖 https://arxiv.org/abs/2602.11709

If you like the package, please cite:
- For XRD-Rust (arXiv): LEBEDA, Miroslav, et al. Rust-accelerated powder X-ray diffraction simulation for high-throughput and machine-learning-driven materials science. arXiv preprint arXiv:2602.11709, 2026.  
- For pymatgen: ONG, Shyue Ping, et al. Python Materials Genomics (pymatgen): A robust, open-source python library for materials analysis. Computational Materials Science, 2013, 68: 314-319.  

## How to install it (tested on Python 3.12)
pip install xrd-rust

## Example: Calculate powder XRD pattern

```python
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
```
