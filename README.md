# XRD-Rust
Compute X-ray diffraction (XRD) patterns using modified [pymatgen’s XRDCalculator](https://pymatgen.org/pymatgen.analysis.diffraction.html), with the performance-critical routines reimplemented in Rust, achieving an average ⚡ 5–6× speedup. The larger and more complex your structure (more peaks and atoms), the greater the speedup gains.

Benchmark results on two large crystallographic datasets demonstrate the following acceleration:

- COD (515 181 structures): ⚡ 6.1 ± 4.6× average speedup, up to 719× faster (1437 min (original pymatgen implementation) → 2 min (Rust-accelerated))

- MC3D (33 142 structures): ⚡ 4.7 ± 1.6× average speedup, up to 25× faster (34.9 s → 1.4 s)

Full benchmarking details are available at:
https://arxiv.org/abs/2602.11709

## How to install it
Write into console:
pip install xrd-rust

 
