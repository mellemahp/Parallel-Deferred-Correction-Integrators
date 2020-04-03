/// Euler Integrators
///
/// Provides both fwd and backwards euler integrators!
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, VectorN, U1};

// local imports
use crate::newton_raphson::newton_raphson_fdiff;

// === End Imports ===

pub fn fwd_euler<N: Dim + DimName>(
    t: f64,
    y0: &VectorN<f64, N>,
    fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    tn: f64,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    y0 + (tn - t) * fxn(t, y0)
}

pub fn bwd_euler<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
    t: f64,
    y0: &VectorN<f64, N>,
    fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    tn: f64,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>
        + Allocator<f64, N, N>
        + Allocator<f64, <N as DimMin<N>>::Output, N>
        + Allocator<f64, <N as DimMin<N>>::Output>
        + Allocator<f64, N, <N as DimMin<N>>::Output>
        + Allocator<f64, <<N as DimMin<N>>::Output as DimSub<U1>>::Output>,
    <N as DimMin<N>>::Output: DimName,
    <N as DimMin<N>>::Output: DimSub<U1>,
{
    const CONV_TOL: f64 = 1.0e-7_f64; // tolerance for convergence of newton iteration
    let y1_hat = fwd_euler(t, y0, fxn, tn);
    let root_problem = |yn: &VectorN<f64, N>| yn - y0 - (tn - t) * fxn(tn, yn);
    newton_raphson_fdiff(root_problem, y1_hat, CONV_TOL).unwrap()
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::test_fxns::one_d::{
        one_d_dynamics, one_d_solution, ONE_D_INIT_TIME, ONE_D_INIT_VAL,
    };
    //use itertools_num;

    #[test]
    fn test_fwd_one_step() {
        let t1 = 1.00001;
        let out = fwd_euler(ONE_D_INIT_TIME, &ONE_D_INIT_VAL, one_d_dynamics, t1);
        // Calculated ans using scipy integrate
        const TOL: f64 = 1.0e-5_f64;
        assert!((out[0] - one_d_solution(t1)[0]).abs() < TOL);
        println!("ESTIMATE | {:?}", out);
        println!("TRUTH! {:?} | ", one_d_solution(t1));
    }
    /*
        //#[test]
        fn test_fwd_iterative() {
            const TOL: f64 = 1.5;
            let mut y = 0.0;
            let mut t = 0.0;

            for tn in itertools_num::linspace(0.0, 1.0, 1000) {
                y = fwd_euler(t, y, test_ode1, tn);
                t = tn;
            }
            let diff = (-88.62652713 - y).abs();

            assert!(diff < TOL);
        }

        //#[test]
        fn test_bwd_one_step() {
            let t0 = 0.0;
            let t1 = 1.0;
            let y0 = 0.0;
            let _out = bwd_euler(t0, y0, test_ode1, t1);
        }

        //#[test]
        fn test_bwd_iterative() {
            const TOL: f64 = 1.5;

            let mut y = 0.0;
            let mut t = 0.0;

            for tn in itertools_num::linspace(0.0, 1.0, 1000) {
                y = bwd_euler(t, y, test_ode1, tn);
                t = tn;
            }
            let diff = (-88.62652713 - y).abs();

            assert!(diff < TOL);
        }

        // TODO Fix this
        fn test_fwd_diff() {
            const TOL: f64 = 1e-6;
            let diff1 = (2.0 - fwd_diff(&test_x2, test_x2(1.0), 1.0)).abs();
            let diff2 = (12.0 - fwd_diff(&test_x3, test_x3(2.0), 2.0)).abs();
            let diff3 = (10.75 - fwd_diff(&test_x4, test_x4(1.5), 1.5)).abs();
            assert!(diff1 < TOL);
            assert!(diff2 < TOL);
            assert!(diff3 < TOL);
        }
    */
}
