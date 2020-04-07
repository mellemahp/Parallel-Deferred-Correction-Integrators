/// Linsearch
///
/// A multi-dimensional linear search routine from pg 479 of
/// numerical recipes
///
///
///
// === Begin Imports ===
// std library imports
use std::f64::EPSILON;

// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// === End Imports ===

// Newton raphson method using Broydens method
// see: https://en.wikipedia.org/wiki/Broyden%27s_method
//
pub fn linsrch_w_backtracking<F, N: Dim + DimName>(
    x_old: &VectorN<f64, N>,
    f_old: f64,
    grad: &VectorN<f64, N>,
    p: &VectorN<f64, N>,
    stepmax: f64,
    fxn: F,
) -> Result<(VectorN<f64, N>, VectorN<f64, N>, f64), &'static str>
where
    F: Fn(&VectorN<f64, N>) -> (VectorN<f64, N>, f64),
    DefaultAllocator: Allocator<f64, N>,
{
    const MAX_STEPS: usize = 100;
    const ALPHA: f64 = 1e-4_f64;
    const TOLX: f64 = EPSILON;

    // pre-initialize variables
    let dim = x_old.len();
    let mut p = p.clone();

    // shrink step if attempted step is too big
    let sum = p.norm();
    if sum > stepmax {
        for idx in 0..dim {
            p[idx] *= stepmax / sum;
        }
    }

    // compute slope for search
    let mut slope = 0.0;
    for idx in 0..dim {
        slope += grad[idx] * &p[idx];
    }
    if slope > 0.0 {
        if slope < EPSILON {
            let x = x_old.clone();
            let (f_vec, f_new) = fxn(&x);
            return Ok((x, f_vec, f_new));
        }
        println!("SLOPE {:?}", slope);
        //slope = -slope;
        return Err("Roundoff problem in linsrch_w_backtracking");
    }

    // compute lambda min
    let mut test = 0.0;
    let mut temp: f64;
    for idx in 0..dim {
        temp = p[idx].abs() / &x_old[idx].abs().max(1.0);
        if temp > test {
            test = temp;
        }
    }

    // pre-initialize values for loop
    let alamin = TOLX / test;
    let mut alam = 1.0;
    let mut tmplam: f64;
    let mut rhs1: f64;
    let mut rhs2: f64;
    let mut a: f64;
    let mut b: f64;
    let mut disc: f64;
    let mut alam2: f64 = 0.0;
    let mut f_2: f64 = 0.0;
    let x_old = x_old.clone();

    // main loop!
    for _ in 0..MAX_STEPS {
        let x_new = &x_old + alam * &p;
        let (f_vec, f_new) = fxn(&x_new.clone());
        // convergence on del_x
        if alam < alamin {
            return Ok((x_new.clone(), f_vec, f_new));
        // sufficient function decrease
        } else if f_new <= f_old + ALPHA * alam * slope {
            return Ok((x_new.clone(), f_vec, f_new));
        //backtrack
        } else {
            if alam == 1.0 {
                tmplam = -slope / (2.0 * (f_new - f_old - slope));
            } else {
                rhs1 = f_new - f_old - alam * slope;
                rhs2 = f_2 - f_old - alam2 * slope;
                a = (rhs1 / (alam * alam) - rhs2 / (alam2 * alam2)) / (alam - alam2);
                b = (-alam2 * rhs1 / (alam * alam) + alam * rhs2 / (alam2 * alam2))
                    / (alam - alam2);
                if a == 0.0 {
                    tmplam = -slope / (2.0 * b);
                } else {
                    disc = b * b - 3.0 * a * slope;
                    if disc < 0.0 {
                        tmplam = 0.5 * alam;
                    } else if b <= 0.0 {
                        tmplam = (-b * disc.sqrt()) / (3.0 * a);
                    } else {
                        tmplam = -slope / (b + disc.sqrt());
                    }
                }
                // lambda <= 0.5 lambda_1
                if tmplam > 0.5 * alam {
                    tmplam = 0.5 * alam
                }
            }
        }
        alam2 = alam;
        f_2 = f_new;
        // lambda >= 0.1 lambda_1
        alam = tmplam.max(0.1 * alam);
    }
    Err("Maximum number of steps reached in Linear search with backtracking")
}
