use pyo3::prelude::*;
use pyo3::types::{PyDict, PyList,PyTuple};
use std::collections::HashMap;
use std::f64::consts::PI;

/// Check if two HKL indices are permutations of each other
#[inline]
fn is_permutation(hkl1: &[i32], hkl2: &[i32]) -> bool {
    if hkl1.len() != hkl2.len() {
        return false;
    }
    
    let mut h1: Vec<i32> = hkl1.iter().map(|&x| x.abs()).collect();
    let mut h2: Vec<i32> = hkl2.iter().map(|&x| x.abs()).collect();
    
    h1.sort_unstable();
    h2.sort_unstable();
    
    h1 == h2
}

/// Get unique families of Miller indices
/// Returns a dict with TUPLES as keys (Python requires hashable keys)
#[pyfunction]
fn get_unique_families_rust(py: Python, hkls: Vec<Vec<i32>>) -> PyResult<PyObject> {
    let mut unique: HashMap<Vec<i32>, Vec<Vec<i32>>> = HashMap::new();

    for hkl1 in hkls {
        let mut found = false;

        for (key, value) in unique.iter_mut() {
            if is_permutation(&hkl1, key) {
                value.push(hkl1.clone());
                found = true;
                break;
            }
        }

        if !found {
            unique.insert(hkl1.clone(), vec![hkl1]);
        }
    }

    let result = PyDict::new(py);
    for (_, family) in unique {
        if let Some(max_hkl) = family.iter().max() {
            let tuple = PyTuple::new(py, max_hkl.iter());
            result.set_item(tuple, family.len())?;
        }
    }

    Ok(result.into())
}

/// Calculate atomic scattering factor
#[inline]
fn calculate_scattering_factor(z: i32, s_squared: f64, coeffs: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for coeff in coeffs {
        sum += coeff[0] * (-coeff[1] * s_squared).exp();
    }
    z as f64 - 41.78214 * s_squared * sum
}

/// Calculate structure factor for a given HKL
#[inline]
fn calculate_structure_factor(
    hkl: &[f64],
    frac_coords: &[Vec<f64>],
    atomic_numbers: &[i32],
    scattering_factors: &[f64],
    occupancies: &[f64],
    dw_corrections: &[f64],
) -> (f64, f64) {
    let mut real_part = 0.0;
    let mut imag_part = 0.0;
    
    for i in 0..frac_coords.len() {
        // Calculate g·r (dot product)
        let g_dot_r = hkl[0] * frac_coords[i][0] 
                    + hkl[1] * frac_coords[i][1] 
                    + hkl[2] * frac_coords[i][2];
        
        // Calculate exp(2πi·g·r) = cos(2πg·r) + i·sin(2πg·r)
        let angle = 2.0 * PI * g_dot_r;
        let cos_val = angle.cos();
        let sin_val = angle.sin();
        
        // Multiply by scattering factor, occupancy, and Debye-Waller correction
        let factor = scattering_factors[i] * occupancies[i] * dw_corrections[i];
        
        real_part += factor * cos_val;
        imag_part += factor * sin_val;
    }
    
    (real_part, imag_part)
}

/// Calculate XRD intensities for all HKL reflections (main bottleneck optimized)
#[pyfunction]
fn calculate_xrd_intensities(
    hkls: Vec<Vec<f64>>,
    g_hkls: Vec<f64>,
    wavelength: f64,
    frac_coords: Vec<Vec<f64>>,
    atomic_numbers: Vec<i32>,
    scattering_coeffs: Vec<Vec<Vec<f64>>>,
    occupancies: Vec<f64>,
    dw_factors: Vec<f64>,
) -> PyResult<Vec<(f64, f64)>> {
    let mut results = Vec::with_capacity(hkls.len());
    
    for (idx, hkl) in hkls.iter().enumerate() {
        let g_hkl = g_hkls[idx];
        
        if g_hkl == 0.0 {
            results.push((0.0, 0.0));
            continue;
        }
        
        // Calculate Bragg angle
        let sin_theta = wavelength * g_hkl / 2.0;
        if sin_theta > 1.0 {
            results.push((0.0, 0.0));
            continue;
        }
        
        let theta = sin_theta.asin();
        let s = g_hkl / 2.0;
        let s_squared = s * s;
        
        // Calculate atomic scattering factors for all atoms
        let mut scattering_factors = Vec::with_capacity(atomic_numbers.len());
        for i in 0..atomic_numbers.len() {
            let coeffs: Vec<[f64; 2]> = scattering_coeffs[i]
                .iter()
                .map(|c| [c[0], c[1]])
                .collect();
            let f = calculate_scattering_factor(atomic_numbers[i], s_squared, &coeffs);
            scattering_factors.push(f);
        }
        
        // Calculate Debye-Waller corrections
        let dw_corrections: Vec<f64> = dw_factors
            .iter()
            .map(|&dw| (-dw * s_squared).exp())
            .collect();
        
        // Calculate structure factor
        let (real, imag) = calculate_structure_factor(
            hkl,
            &frac_coords,
            &atomic_numbers,
            &scattering_factors,
            &occupancies,
            &dw_corrections,
        );
        
        // Calculate intensity
        let intensity = real * real + imag * imag;
        
        // Lorentz polarization factor
        let cos_theta = theta.cos();
        let sin_theta_sq = theta.sin().powi(2);
        let two_theta = 2.0 * theta;
        let lorentz_factor = (1.0 + two_theta.cos().powi(2)) / (sin_theta_sq * cos_theta);
        
        let final_intensity = intensity * lorentz_factor;
        let two_theta_deg = two_theta.to_degrees();
        
        results.push((two_theta_deg, final_intensity));
    }
    
    Ok(results)
}

/// Fast peak merging based on two-theta tolerance
#[pyfunction]
fn merge_peaks(
    two_thetas: Vec<f64>,
    intensities: Vec<f64>,
    hkls: Vec<Vec<i32>>,
    d_hkls: Vec<f64>,
    tolerance: f64,
) -> PyResult<(Vec<f64>, Vec<f64>, Vec<Vec<Vec<i32>>>, Vec<f64>)> {
    if two_thetas.is_empty() {
        return Ok((vec![], vec![], vec![], vec![]));
    }
    
    let mut merged_thetas = Vec::new();
    let mut merged_intensities = Vec::new();
    let mut merged_hkls = Vec::new();
    let mut merged_d_hkls = Vec::new();
    
    let mut current_theta = two_thetas[0];
    let mut current_intensity = intensities[0];
    let mut current_hkls = vec![hkls[0].clone()];
    let mut current_d = d_hkls[0];
    
    for i in 1..two_thetas.len() {
        if (two_thetas[i] - current_theta).abs() < tolerance {
            // Merge with current peak
            current_intensity += intensities[i];
            current_hkls.push(hkls[i].clone());
        } else {
            // Save current peak and start new one
            merged_thetas.push(current_theta);
            merged_intensities.push(current_intensity);
            merged_hkls.push(current_hkls);
            merged_d_hkls.push(current_d);
            
            current_theta = two_thetas[i];
            current_intensity = intensities[i];
            current_hkls = vec![hkls[i].clone()];
            current_d = d_hkls[i];
        }
    }
    
    merged_thetas.push(current_theta);
    merged_intensities.push(current_intensity);
    merged_hkls.push(current_hkls);
    merged_d_hkls.push(current_d);
    
    Ok((merged_thetas, merged_intensities, merged_hkls, merged_d_hkls))
}

/// Normalize intensities
#[pyfunction]
fn normalize_intensities(intensities: Vec<f64>, max_value: f64) -> PyResult<Vec<f64>> {
    if intensities.is_empty() {
        return Ok(vec![]);
    }
    
    let max_intensity = intensities
        .iter()
        .cloned()
        .fold(f64::NEG_INFINITY, f64::max);
    
    if max_intensity == 0.0 {
        return Ok(intensities);
    }
    
    let normalized: Vec<f64> = intensities
        .iter()
        .map(|&i| (i / max_intensity) * max_value)
        .collect();
    
    Ok(normalized)
}

/// Python module definition
#[pymodule]
fn xrd_rust_accelerator(_py: Python, m: &PyModule) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_unique_families_rust, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_xrd_intensities, m)?)?;
    m.add_function(wrap_pyfunction!(merge_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_intensities, m)?)?;
    Ok(())
}
