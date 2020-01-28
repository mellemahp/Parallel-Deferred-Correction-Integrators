/// Newton Raphson Solver
///
/// A multi-dimensional generalized newton-raphson root finder
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, MatrixN, VectorN, U1};

// === End Imports ===

// Basic newton-raphson method using finite differencing
pub fn newton_raphson_fdiff<F, N: Dim + DimName + DimMin<N> + DimSub<U1>>(
    fxn: F,
    x_0: VectorN<f64, N>,
    acc: f64,
) -> Result<VectorN<f64, N>, &'static str>
where
    F: Fn(&VectorN<f64, N>) -> VectorN<f64, N>,
    DefaultAllocator: Allocator<f64, N>
        + Allocator<f64, N, N>
        + Allocator<f64, <N as DimMin<N>>::Output, N>
        + Allocator<f64, <N as DimMin<N>>::Output>
        + Allocator<f64, N, <N as DimMin<N>>::Output>
        + Allocator<f64, <<N as DimMin<N>>::Output as DimSub<U1>>::Output>,
    <N as DimMin<N>>::Output: DimName,
    <N as DimMin<N>>::Output: DimSub<U1>,
{
    const MAX_ITER: i32 = 100;
    const INV_TOL: f64 = 1.0_e-8_f64;

    // pre-initialize variables
    let mut fk = fxn(&x_0);
    let mut jac_inv: MatrixN<f64, N> = fdiff_jacobian(&fxn, &fk, &x_0)
        .pseudo_inverse(INV_TOL)
        .unwrap();
    let mut x_new: VectorN<f64, N>;
    let mut x_last = x_0.clone();

    // Iterate to victory!
    for _j in 0..MAX_ITER {
        x_new = &x_last - jac_inv * fk;
        if (&x_new - &x_last).norm() < acc {
            return Ok(x_new);
        }
        x_last = x_new.clone();
        fk = fxn(&x_new);
        jac_inv = fdiff_jacobian(&fxn, &fk, &x_new)
            .pseudo_inverse(INV_TOL)
            .unwrap();
    }
    return Err("Maximum Number of Iterations Reached");
}

// per numerical recipes chpt 5 pg 230 we use
// h \approx \sqrt(e_f) * x_c where x_c is the curvature scale
// typically we assume x_c = x unless x is near 0 then we want to use
// another value
fn fdiff_jacobian<F, N: Dim + DimName>(
    fxn: &F,
    y: &VectorN<f64, N>,
    x: &VectorN<f64, N>,
) -> MatrixN<f64, N>
where
    F: Fn(&VectorN<f64, N>) -> VectorN<f64, N>,
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, N, N>,
{
    // Approximately sqrt of ULP precision
    const H_FACTOR: f64 = 1.0e-7_f64;
    // Used to check if X is near zero
    const Z_TOL: f64 = 1.0e-10;
    // Alternative step if X is near zero
    const Z_FACTOR: f64 = 1.0e-14;

    // Initialize a vector for differences
    let shift_vals = VectorN::<f64, N>::from_iterator(x.iter().map(|val| {
        if *val > Z_TOL {
            val * H_FACTOR
        } else {
            Z_FACTOR
        }
    }));

    // Pre-initialize values
    let mut diff: VectorN<f64, N> = VectorN::<f64, N>::zeros();
    let mut columns: Vec<VectorN<f64, N>> = Vec::new();
    let mut fxn_shift: VectorN<f64, N>;

    for m in 0..x.len() {
        diff.fill(0.0);
        diff[m] = shift_vals[m];
        fxn_shift = fxn(&(x + &diff));
        columns.push((&fxn_shift - y) / shift_vals[m]);
    }

    MatrixN::<f64, N>::from_columns(&columns)
}

#[cfg(test)]
mod tests {
    use super::*;
    use na::{Matrix2, Vector1, Vector2};

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

    #[test]
    fn test_newton_1d() {
        let i_guess = Vector1::new(1.0);
        let fxn = |x: &Vector1<f64>| Vector1::new(x[0].powf(3.0) + 3.0 * x[0] - 7.0);
        let ans =
            newton_raphson_fdiff(fxn, i_guess, 1.0e-6_f64).expect("Couldn't converge to solution");

        // truth (from wolfram)
        let sol = 1.406287579960534691140831;
        const TOL: f64 = 1.0e-6_f64;
        assert!((ans[0] - sol).abs() < TOL);
    }

    #[test]
    fn test_newton_2d() {
        let i_guess = Vector2::new(0.0, 0.0);
        let fxn = |x: &Vector2<f64>| {
            Vector2::new(
                x[0] + 0.5 * (x[0] - x[1]).powf(3.0) - 1.0,
                0.5 * (x[1] - x[0]).powf(3.0) + x[1],
            )
        };

        // value found using scipy.optimize.root
        let ans =
            newton_raphson_fdiff(fxn, i_guess, 1.0e-6_f64).expect("Couldn't converge to solution");

        let python_sol = Vector2::new(0.8411639, 0.1588361);
        const TOL: f64 = 1.0e-7_f64;
        for idx in 0..2 {
            assert!((ans[idx] - python_sol[idx]).abs() < TOL);
        }
    }
}
