use pyo3::prelude::*;
use pyo3::types::{PyDict, PyTuple};
use std::collections::HashMap;
use std::f64::consts::PI;
use rayon::prelude::*;
use wide::f64x4;

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

#[pyfunction]
fn get_unique_families_rust(py: Python, hkls: Vec<Vec<i32>>) -> PyResult<Bound<'_, PyDict>> {
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
            let tuple = PyTuple::new(py, max_hkl.iter())?;
            result.set_item(tuple, family.len())?;
        }
    }
    Ok(result)
}

#[pyfunction]
fn generate_hkl_points(
    recip_matrix: Vec<Vec<f64>>,
    max_r: f64,
    min_r: f64,
) -> PyResult<Vec<(Vec<f64>, f64)>> {
    let a_star = (recip_matrix[0][0].powi(2)
                + recip_matrix[0][1].powi(2)
                + recip_matrix[0][2].powi(2)).sqrt();
    let b_star = (recip_matrix[1][0].powi(2)
                + recip_matrix[1][1].powi(2)
                + recip_matrix[1][2].powi(2)).sqrt();
    let c_star = (recip_matrix[2][0].powi(2)
                + recip_matrix[2][1].powi(2)
                + recip_matrix[2][2].powi(2)).sqrt();

    let max_h = if a_star > 0.0 { (max_r / a_star).ceil() as i32 + 1 } else { 1 };
    let max_k = if b_star > 0.0 { (max_r / b_star).ceil() as i32 + 1 } else { 1 };
    let max_l = if c_star > 0.0 { (max_r / c_star).ceil() as i32 + 1 } else { 1 };

    let mut points: Vec<(Vec<f64>, f64)> = Vec::new();

    for h in -max_h..=max_h {
        for k in -max_k..=max_k {
            for l in -max_l..=max_l {
                if h == 0 && k == 0 && l == 0 {
                    continue;
                }

                let gx = h as f64 * recip_matrix[0][0]
                       + k as f64 * recip_matrix[1][0]
                       + l as f64 * recip_matrix[2][0];
                let gy = h as f64 * recip_matrix[0][1]
                       + k as f64 * recip_matrix[1][1]
                       + l as f64 * recip_matrix[2][1];
                let gz = h as f64 * recip_matrix[0][2]
                       + k as f64 * recip_matrix[1][2]
                       + l as f64 * recip_matrix[2][2];

                let g_hkl = (gx * gx + gy * gy + gz * gz).sqrt();

                if g_hkl >= min_r && g_hkl <= max_r {
                    points.push((vec![h as f64, k as f64, l as f64], g_hkl));
                }
            }
        }
    }

    points.sort_by(|a, b| {
        a.1.partial_cmp(&b.1)
            .unwrap_or(std::cmp::Ordering::Equal)
            .then_with(|| (-a.0[0] as i64).cmp(&(-b.0[0] as i64)))
            .then_with(|| (-a.0[1] as i64).cmp(&(-b.0[1] as i64)))
            .then_with(|| (-a.0[2] as i64).cmp(&(-b.0[2] as i64)))
    });

    Ok(points)
}

#[inline]
fn calculate_scattering_factor(z: i32, s_squared: f64, coeffs: &[[f64; 2]]) -> f64 {
    let mut sum = 0.0;
    for coeff in coeffs {
        sum += coeff[0] * (-coeff[1] * s_squared).exp();
    }
    z as f64 - 41.78214 * s_squared * sum
}

#[inline]
fn calculate_structure_factor_scalar_soa(
    hkl: &[f64],
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    scattering_factors: &[f64],
    occupancies: &[f64],
    dw_corrections: &[f64],
) -> (f64, f64) {
    let mut real_part = 0.0;
    let mut imag_part = 0.0;
    let n = xs.len();

    for i in 0..n {
        let g_dot_r = hkl[0] * xs[i]
                    + hkl[1] * ys[i]
                    + hkl[2] * zs[i];
        let angle = 2.0 * PI * g_dot_r;
        let factor = scattering_factors[i] * occupancies[i] * dw_corrections[i];
        real_part += factor * angle.cos();
        imag_part += factor * angle.sin();
    }
    (real_part, imag_part)
}

fn calculate_structure_factor_simd_soa(
    hkl: &[f64],
    xs: &[f64],
    ys: &[f64],
    zs: &[f64],
    scattering_factors: &[f64],
    occupancies: &[f64],
    dw_corrections: &[f64],
) -> (f64, f64) {
    let n = xs.len();
    let chunks = n / 4;

    let hkl0   = f64x4::splat(hkl[0]);
    let hkl1   = f64x4::splat(hkl[1]);
    let hkl2   = f64x4::splat(hkl[2]);
    let two_pi = f64x4::splat(2.0 * PI);

    let mut real_sum = f64x4::ZERO;
    let mut imag_sum = f64x4::ZERO;

    for chunk in 0..chunks {
        let b = chunk * 4;

        let fx = f64x4::new([xs[b], xs[b+1], xs[b+2], xs[b+3]]);
        let fy = f64x4::new([ys[b], ys[b+1], ys[b+2], ys[b+3]]);
        let fz = f64x4::new([zs[b], zs[b+1], zs[b+2], zs[b+3]]);

        let g_dot_r = hkl0 * fx + hkl1 * fy + hkl2 * fz;
        let angle   = two_pi * g_dot_r;

        let sf  = f64x4::new([
            scattering_factors[b],   scattering_factors[b+1],
            scattering_factors[b+2], scattering_factors[b+3],
        ]);
        let occ = f64x4::new([
            occupancies[b],   occupancies[b+1],
            occupancies[b+2], occupancies[b+3],
        ]);
        let dw  = f64x4::new([
            dw_corrections[b],   dw_corrections[b+1],
            dw_corrections[b+2], dw_corrections[b+3],
        ]);

        let factor = sf * occ * dw;
        real_sum += factor * angle.cos();
        imag_sum += factor * angle.sin();
    }

    let mut real: f64 = real_sum.to_array().iter().sum();
    let mut imag: f64 = imag_sum.to_array().iter().sum();

    for i in (chunks * 4)..n {
        let g_dot_r = hkl[0] * xs[i] + hkl[1] * ys[i] + hkl[2] * zs[i];
        let angle   = 2.0 * PI * g_dot_r;
        let factor  = scattering_factors[i] * occupancies[i] * dw_corrections[i];
        real += factor * angle.cos();
        imag += factor * angle.sin();
    }

    (real, imag)
}

#[pyfunction]
fn calculate_xrd_intensities(
    hkls: Vec<Vec<f64>>,
    g_hkls: Vec<f64>,
    wavelength: f64,
    frac_coords_x: Vec<f64>,
    frac_coords_y: Vec<f64>,
    frac_coords_z: Vec<f64>,
    atomic_numbers: Vec<i32>,
    scattering_coeffs: Vec<Vec<Vec<f64>>>,
    occupancies: Vec<f64>,
    dw_factors: Vec<f64>,
    parallel: bool,
    num_threads: usize,
    use_simd: bool,
) -> PyResult<Vec<(f64, f64)>> {

    let frac_coords_x     = std::sync::Arc::new(frac_coords_x);
    let frac_coords_y     = std::sync::Arc::new(frac_coords_y);
    let frac_coords_z     = std::sync::Arc::new(frac_coords_z);
    let atomic_numbers    = std::sync::Arc::new(atomic_numbers);
    let scattering_coeffs = std::sync::Arc::new(scattering_coeffs);
    let occupancies       = std::sync::Arc::new(occupancies);
    let dw_factors        = std::sync::Arc::new(dw_factors);
    let g_hkls            = std::sync::Arc::new(g_hkls);

    let compute = {
        let frac_coords_x     = std::sync::Arc::clone(&frac_coords_x);
        let frac_coords_y     = std::sync::Arc::clone(&frac_coords_y);
        let frac_coords_z     = std::sync::Arc::clone(&frac_coords_z);
        let atomic_numbers    = std::sync::Arc::clone(&atomic_numbers);
        let scattering_coeffs = std::sync::Arc::clone(&scattering_coeffs);
        let occupancies       = std::sync::Arc::clone(&occupancies);
        let dw_factors        = std::sync::Arc::clone(&dw_factors);
        let g_hkls            = std::sync::Arc::clone(&g_hkls);

        move |idx: usize, hkl: &Vec<f64>| -> (f64, f64) {
            let g_hkl = g_hkls[idx];
            if g_hkl == 0.0 { return (0.0, 0.0); }

            let sin_theta = wavelength * g_hkl / 2.0;
            if sin_theta > 1.0 { return (0.0, 0.0); }

            let theta     = sin_theta.asin();
            let s         = g_hkl / 2.0;
            let s_squared = s * s;

            let mut scattering_factors = Vec::with_capacity(atomic_numbers.len());
            for i in 0..atomic_numbers.len() {
                let coeffs: Vec<[f64; 2]> = scattering_coeffs[i]
                    .iter()
                    .map(|c| [c[0], c[1]])
                    .collect();
                scattering_factors.push(
                    calculate_scattering_factor(atomic_numbers[i], s_squared, &coeffs)
                );
            }

            let dw_corrections: Vec<f64> = dw_factors
                .iter()
                .map(|&dw| (-dw * s_squared).exp())
                .collect();

            let (real, imag) = if use_simd {
                calculate_structure_factor_simd_soa(
                    hkl,
                    &frac_coords_x,
                    &frac_coords_y,
                    &frac_coords_z,
                    &scattering_factors,
                    &occupancies,
                    &dw_corrections,
                )
            } else {
                calculate_structure_factor_scalar_soa(
                    hkl,
                    &frac_coords_x,
                    &frac_coords_y,
                    &frac_coords_z,
                    &scattering_factors,
                    &occupancies,
                    &dw_corrections,
                )
            };

            let intensity      = real * real + imag * imag;
            let cos_theta      = theta.cos();
            let sin_theta_sq   = theta.sin().powi(2);
            let two_theta      = 2.0 * theta;
            let lorentz_factor = (1.0 + two_theta.cos().powi(2))
                               / (sin_theta_sq * cos_theta);

            (two_theta.to_degrees(), intensity * lorentz_factor)
        }
    };

    let results: Vec<(f64, f64)> = if parallel {
        let pool = rayon::ThreadPoolBuilder::new()
            .num_threads(num_threads)
            .build()
            .unwrap();
        pool.install(|| {
            hkls.par_iter()
                .enumerate()
                .map(|(idx, hkl)| compute(idx, hkl))
                .collect()
        })
    } else {
        hkls.iter()
            .enumerate()
            .map(|(idx, hkl)| compute(idx, hkl))
            .collect()
    };

    Ok(results)
}

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

    let mut merged_thetas      = Vec::new();
    let mut merged_intensities = Vec::new();
    let mut merged_hkls        = Vec::new();
    let mut merged_d_hkls      = Vec::new();

    let mut current_theta     = two_thetas[0];
    let mut current_intensity = intensities[0];
    let mut current_hkls      = vec![hkls[0].clone()];
    let mut current_d         = d_hkls[0];

    for i in 1..two_thetas.len() {
        if (two_thetas[i] - current_theta).abs() < tolerance {
            current_intensity += intensities[i];
            current_hkls.push(hkls[i].clone());
        } else {
            merged_thetas.push(current_theta);
            merged_intensities.push(current_intensity);
            merged_hkls.push(current_hkls);
            merged_d_hkls.push(current_d);
            current_theta     = two_thetas[i];
            current_intensity = intensities[i];
            current_hkls      = vec![hkls[i].clone()];
            current_d         = d_hkls[i];
        }
    }

    merged_thetas.push(current_theta);
    merged_intensities.push(current_intensity);
    merged_hkls.push(current_hkls);
    merged_d_hkls.push(current_d);

    Ok((merged_thetas, merged_intensities, merged_hkls, merged_d_hkls))
}

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
    Ok(intensities.iter().map(|&i| (i / max_intensity) * max_value).collect())
}

#[pymodule]
fn xrd_rust_accelerator(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_function(wrap_pyfunction!(get_unique_families_rust, m)?)?;
    m.add_function(wrap_pyfunction!(generate_hkl_points, m)?)?;
    m.add_function(wrap_pyfunction!(calculate_xrd_intensities, m)?)?;
    m.add_function(wrap_pyfunction!(merge_peaks, m)?)?;
    m.add_function(wrap_pyfunction!(normalize_intensities, m)?)?;
    Ok(())
}