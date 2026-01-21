"""XRD Calculator with Rust acceleration."""

from .xrd_calculator_rust import (
    XRDCalculatorRust,
    create_xrd_calculator,
    WAVELENGTHS,
)

__version__ = "0.1.0"
__all__ = ["XRDCalculatorRust", "create_xrd_calculator", "WAVELENGTHS"]
