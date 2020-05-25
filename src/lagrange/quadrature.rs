/// Legendre Quadrature (quadrature)
///
/// Defines the weight "stencils" for Legendre Polynomial Quadrature
///
/// Each weight w_dd_i is found by integrating:
/// w_dd_0 = \int_{x_0}^x 1 dx
/// w_dd_j = \int_{x_0}^x \prod_{i=1}^{j-1} (t - t_i) dx
///
/// Currently only weights up to order 4 are provided. Higher order
/// templates can be derived using a symbolic solver such as wolfram alpha
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// standard library
use std::collections::VecDeque;

// === Begin Imports ===

pub fn get_x_pow(x_0: f64, x: f64, order: usize) -> Vec<f64> {
    let mut x_pow: Vec<f64> = Vec::with_capacity(order);
    for i in 1..=order + 1 {
        x_pow.push(x.powi(i as i32) - x_0.powi(i as i32))
    }
    x_pow
}

// ---------------------- DIVIDED DIFFERENCE STYLE -------------------------------------
pub fn w_dd_0(x_pow: &Vec<f64>) -> f64 {
    x_pow[0]
}

pub fn w_dd_1(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    0.5 * x_pow[1] - times[0] * x_pow[0]
}

pub fn w_dd_2(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 6.0
        * (6.0 * times[0] * times[1] * x_pow[0] - 3.0 * (times[0] + times[1]) * x_pow[1]
            + 2.0 * x_pow[2])
}

pub fn w_dd_3(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 12.0
        * (-12.0 * times[0] * times[1] * times[2] * x_pow[0]
            + 6.0 * (times[0] * (times[1] + times[2]) + times[1] * times[2]) * x_pow[1]
            - 4.0 * (times[0] + times[1] + times[2]) * x_pow[2]
            + 3.0 * x_pow[3])
}

pub fn w_dd_4(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 60.0
        * (60.0 * times[0] * times[1] * times[2] * times[3] * x_pow[0]
            - 30.0
                * (times[0] * times[1] * (times[2] + times[3])
                    + times[0] * times[2] * times[3]
                    + times[1] * times[2] * times[3])
                * x_pow[1]
            + 20.0
                * (times[0] * (times[1] + times[2] + times[3])
                    + times[1] * (times[2] + times[3])
                    + times[2] * times[3])
                * x_pow[2]
            - 15.0 * (times[0] + times[1] + times[2] + times[3]) * x_pow[3]
            + 12.0 * x_pow[4])
}
// -------------------------------------------------------------------------------------

// ------------------------- Classic Lagrange Style ------------------------------------
pub fn get_weights(times: &VecDeque<f64>) -> Vec<Vec<f64>> {
    // pre-initialize weights
    let mut weights: Vec<Vec<f64>> = (0..4).map(|_| Vec::with_capacity(4)).collect();
    // TODO add check that vecDeque size is correct
    for i in 0..4 {
        weights[i] = w_n3(
            times[i % 4],
            times[(i + 1) % 4],
            times[(i + 2) % 4],
            times[(i + 3) % 4],
        )
    }
    weights
}

// Third order weights stencil
pub fn w_n3(a: f64, b: f64, c: f64, d: f64) -> Vec<f64> {
    let factor = 1.0 / (12.0 * (a - b) * (a - c) * (a - d));
    vec![
        -12.0 * b * c * d,
        6.0 * (b * (c + d) + c * d),
        -4.0 * (b + c + d),
        3.0,
    ]
    .iter()
    .map(|x| x * factor)
    .collect()
}

pub fn specific_weights(x_pows: Vec<f64>, weights: &Vec<Vec<f64>>) -> Vec<f64> {
    weights
        .iter()
        .map(|w| w.iter().zip(x_pows.iter()).map(|(w, x)| x * w).sum())
        .collect()
}

pub fn lagrange_quad_third_order<N: Dim + DimName>(
    x_0: f64,
    x: f64,
    weights: &Vec<Vec<f64>>,
    vals: &VecDeque<VectorN<f64, N>>,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Compute the specific weights for the integral bounds
    let x_pows = get_x_pow(x_0, x, 3);
    let spec_weights = specific_weights(x_pows, weights);
    spec_weights
        .iter()
        .zip(vals.iter())
        .map(|(w, y)| *w * y)
        .sum()
}

// -------------------------------------------------------------------------------------

pub fn lagrange_quad_fourth_order<N: Dim + DimName>(
    x_0: f64,
    x: f64,
    times: &VecDeque<f64>,
    div_diffs: &Vec<VectorN<f64, N>>,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let x_pows = get_x_pow(x_0, x, 4);
    let weights = vec![
        w_dd_0(&x_pows),
        w_dd_1(&x_pows, &times),
        w_dd_2(&x_pows, &times),
        w_dd_3(&x_pows, &times),
        w_dd_4(&x_pows, &times),
    ];
    div_diffs
        .iter()
        .zip(weights.iter())
        .map(|(w, y)| *y * w)
        .sum()
}

// Tests
#[cfg(test)]
mod tests {
    extern crate nalgebra as na;
    use na::Vector1;

    use super::*;
    use crate::lagrange::div_diff::divided_diff;

    #[test]
    fn test_x_pow() {
        // test trivial cases
        let x_0: f64 = 0.0;
        let x: f64 = 1.0;
        let ans = get_x_pow(x_0, x, 4);

        for i in 0..ans.len() {
            assert_eq!(ans[i], 1.0);
        }
        let x_0: f64 = 4.0;
        let x: f64 = 4.0;
        let ans = get_x_pow(x_0, x, 4);

        for i in 0..ans.len() {
            assert_eq!(ans[i], 0.0);
        }

        // test one hand calculated one
        let x_0: f64 = 2.0;
        let x: f64 = 3.0;
        let ans = get_x_pow(x_0, x, 3);
        let truth = vec![1.0, 5.0, 19.0, 65.0];
        for i in 0..ans.len() {
            assert_eq!(ans[i], truth[i]);
        }
    }

    #[test]
    fn validate_weights_dd() {
        // Checks that weights match integral from wolfram alpha
        let x_0 = 1.0;
        let x = 5.0;
        let times = VecDeque::from(vec![5.0, 3.0, 2.5, 1.0]);

        // Computed using wolfram alpha
        let w_dd_trues = vec![4.0, -8.0, 5.333333333333333, -8.0, -8.533333333333333];
        const TOL: f64 = 1.0e-10;

        let x_pows = get_x_pow(x_0, x, 4);
        let w_dd_est = vec![
            w_dd_0(&x_pows),
            w_dd_1(&x_pows, &times),
            w_dd_2(&x_pows, &times),
            w_dd_3(&x_pows, &times),
            w_dd_4(&x_pows, &times),
        ];

        for idx in 0..=4 {
            assert!((w_dd_trues[idx] - w_dd_est[idx]).abs() < TOL);
        }
    }

    #[test]
    fn test_integration_lin_dd() {
        // tests the integration by legendre quadrature against known integrals
        // Linear model: y = 2x
        let lin_times = VecDeque::from(vec![5.0, 4.0, 3.0, 2.0]);
        let lin_pts = VecDeque::from(vec![
            Vector1::new(10.0),
            Vector1::new(8.0),
            Vector1::new(6.0),
            Vector1::new(4.0),
        ]);
        let lin_div_diff = divided_diff(&lin_pts, &lin_times);
        let true_lin_area = Vector1::new(21.0);
        let est_lin_area = lagrange_quad_fourth_order(2.0, 5.0, &lin_times, &lin_div_diff);

        const TOL: f64 = 1.0e-10;
        assert!((true_lin_area[0] - est_lin_area[0]).abs() < TOL);
    }

    #[test]
    fn test_integration_quad_dd() {
        // Quadratic: y = x^2
        let quad_times = VecDeque::from(vec![5.0, 4.0, 3.0, 2.0]);
        let quad_pts = VecDeque::from(vec![
            Vector1::new(25.0),
            Vector1::new(16.0),
            Vector1::new(9.0),
            Vector1::new(4.0),
        ]);
        let quad_div_diff = divided_diff(&quad_pts, &quad_times);
        let true_quad_area = Vector1::new(39.0);
        let est_quad_area = lagrange_quad_fourth_order(2.0, 5.0, &quad_times, &quad_div_diff);

        const TOL: f64 = 1.0e-10;
        assert!((true_quad_area[0] - est_quad_area[0]).abs() < TOL);
    }

    #[test]
    fn test_integration_cubic_dd() {
        // cubic: y = x^3 + x^2 + 1
        let cubic_times = VecDeque::from(vec![5.0, 4.0, 2.0, 1.0]);
        let cubic_pts = VecDeque::from(vec![
            Vector1::new(151.0),
            Vector1::new(81.0),
            Vector1::new(13.0),
            Vector1::new(3.0),
        ]);
        let cubic_div_diff = divided_diff(&cubic_pts, &cubic_times);
        let true_cubic_area = Vector1::new(128.724);
        let est_cubic_area = lagrange_quad_fourth_order(2.0, 4.5, &cubic_times, &cubic_div_diff);

        const TOL: f64 = 1.0e-4;
        assert!((true_cubic_area[0] - est_cubic_area[0]).abs() < TOL);
    }

    #[test]
    fn test_integration_lin_classic() {
        // tests the integration by legendre quadrature against known integrals
        // Linear model: y = 2x
        let lin_times = VecDeque::from(vec![5.0, 4.0, 3.0, 2.0]);
        let lin_pts = VecDeque::from(vec![
            Vector1::new(10.0),
            Vector1::new(8.0),
            Vector1::new(6.0),
            Vector1::new(4.0),
        ]);
        let lin_weights = get_weights(&lin_times);
        let true_lin_area = Vector1::new(21.0);
        let est_lin_area = lagrange_quad_third_order(2.0, 5.0, &lin_weights, &lin_pts);

        const TOL: f64 = 1.0e-10;
        assert!((true_lin_area[0] - est_lin_area[0]).abs() < TOL);
    }

    #[test]
    fn test_integration_quad_classic() {
        // Quadratic: y = x^2
        let quad_times = VecDeque::from(vec![5.0, 4.0, 3.0, 2.0]);
        let quad_pts = VecDeque::from(vec![
            Vector1::new(25.0),
            Vector1::new(16.0),
            Vector1::new(9.0),
            Vector1::new(4.0),
        ]);
        let quad_weights = get_weights(&quad_times);
        let true_quad_area = Vector1::new(39.0);
        let est_quad_area = lagrange_quad_third_order(2.0, 5.0, &quad_weights, &quad_pts);

        const TOL: f64 = 1.0e-10;
        assert!((true_quad_area[0] - est_quad_area[0]).abs() < TOL);
    }

    #[test]
    fn test_integration_cubic_classic() {
        let cubic_times = VecDeque::from(vec![5.0, 4.0, 2.0, 1.0]);
        let cubic_pts = VecDeque::from(vec![
            Vector1::new(151.0),
            Vector1::new(81.0),
            Vector1::new(13.0),
            Vector1::new(3.0),
        ]);
        let cubic_weights = get_weights(&cubic_times);
        let true_cubic_area = Vector1::new(128.724);
        let est_cubic_area = lagrange_quad_third_order(2.0, 4.5, &cubic_weights, &cubic_pts);

        const TOL: f64 = 1.0e-4;
        assert!((true_cubic_area[0] - est_cubic_area[0]).abs() < TOL);
    }
}
