/// Finite Differencing Routines
/// Provides routines associated with finding derivatives or
/// jacobians via finite differencing (here we use the center difference)
///
/// per numerical recipes chpt 5 pg 230 we use
/// h \approx \sqrt(e_f) * x_c where x_c is the curvature scale
/// typically we assume x_c = x unless x is near 0 then we want to use
/// another value
// === Begin Imports ===
// std library
use std::f64::EPSILON;

// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixN, VectorN};

// === End Imports ===

// Finds jacobian matrix via finite differencing
pub fn fdiff_jacobian<F, N: Dim + DimName>(
    fxn: &F,
    y: &VectorN<f64, N>,
    x: &VectorN<f64, N>,
) -> MatrixN<f64, N>
where
    F: Fn(&VectorN<f64, N>) -> VectorN<f64, N>,
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, N, N>,
{
    // Approximately cube root of ULP precision
    const H_FACTOR: f64 = EPSILON; // 1.4901161193847656e-8_f64;
    const Z_LIM: f64 = 1e-16_f64;

    // Initialize a vector for differences
    let shift_vals = VectorN::<f64, N>::from_iterator(x.iter().map(|val| {
        if val * H_FACTOR < Z_LIM{
            let temp = val + val.abs() * H_FACTOR;
            temp - val
        } else {
            EPSILON
        }
    }));

    // Pre-initialize values
    let mut diff: VectorN<f64, N> = VectorN::<f64, N>::zeros();
    let mut columns: Vec<VectorN<f64, N>> = Vec::new();
    let mut fxn_shift_p: VectorN<f64, N>;
    let mut fxn_shift_m: VectorN<f64, N>;

    for m in 0..x.len() {
        diff.fill(0.0);
        diff[m] = shift_vals[m];
        fxn_shift_p = fxn(&(x + &diff));
        fxn_shift_m = fxn(&(x - &diff));
        columns.push((&fxn_shift_p - &fxn_shift_m) / (2.0 * shift_vals[m]));
    }
    MatrixN::<f64, N>::from_columns(&columns)
}

// Finds jacobian matrix via finite differencing
pub fn fdiff_jacobian_2<F, N: Dim + DimName>(
    fxn: &F,
    y: &VectorN<f64, N>,
    x: &VectorN<f64, N>,
) -> MatrixN<f64, N>
where
    F: Fn(&VectorN<f64, N>) -> VectorN<f64, N>,
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, N, N>,
{
    let dim = x.len();
    let mut mat = MatrixN::<f64, N>::repeat(0.0);
    let mut temp: f64;
    let mut xh_p: VectorN<f64, N> = x.clone();
    let mut xh_m: VectorN<f64, N> = x.clone();
    let mut h: f64;
    let mut f_p: VectorN<f64, N>;
    let mut f_m: VectorN<f64, N>;
    for jdx in 0..dim {
        temp = x[jdx];
        h = EPSILON * temp.abs();
        if h == 0.0 {
            h = EPSILON;
        }
        xh_p[jdx] = temp + h;
        xh_m[jdx] = temp - h;
        h = xh_p[jdx] - temp;
        f_p = fxn(&xh_p);
        f_m = fxn(&xh_m);
        xh_p[jdx] = temp;
        xh_m[jdx] = temp;
        for idx in 0..dim {
            mat[(idx, jdx)] = (f_p[idx] - f_m[idx]) / (2.0 * h);
        }
    }
    mat
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::{Matrix2, Vector2};

    #[test]
    fn test_jacobian() {
        let y = Vector2::new(1.0, 2.0);
        let x = |z: &Vector2<f64>| z.component_mul(&y);
        let z_0 = Vector2::new(2.0, 2.0);
        let y_0 = x(&z_0);
        let jac = fdiff_jacobian(&x, &y_0, &z_0);
        const TOL: f64 = 1.0e-8_f64;
        let solution = Matrix2::new(1.0, 0.0, 0.0, 2.0);
        for idx in 0..4 {
            assert!((jac[idx] - solution[idx]).abs() < TOL);
        }
    }
}
