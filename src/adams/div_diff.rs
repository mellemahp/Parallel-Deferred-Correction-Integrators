extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

use std::collections::VecDeque;

// TODO Double Check directions
// Divided diff from:
// https://www.math.usm.edu/lambers/mat460/fall09/lecture17.pdf
pub fn divided_diff<N: Dim + DimName>(
    points: &VecDeque<VectorN<f64, N>>,
    times: &VecDeque<f64>,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let n = times.len();
    // initialize empty finite difference table
    let mut diffs = vec![vec![VectorN::<f64, N>::zeros(); n]; n];
    let mut times: Vec<f64> = times.clone().into_iter().collect();
    times.reverse();
    for i in (0..n).rev() {
        diffs[i][0] = points[i].clone();
    }
    for j in 1..n {
        for i in 0..n - j {
            diffs[i][j] = (&diffs[i + 1][j - 1] - &diffs[i][j - 1]) / (times[i + j] - times[i])
        }
    }
    diffs[0].clone()
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use na::Vector1;

    //#[test]
    fn test_divided_difference_1() {
        // Case testing against geeksforgeeks.com
        // NOTE: because of the way the divided diff works for
        // the adams predictor this vector is flipped from the
        // one in the geeks for geeks site
        let pts = VecDeque::from(vec![
            Vector1::new(16.0),
            Vector1::new(14.0),
            Vector1::new(13.0),
            Vector1::new(12.0),
        ]);
        let times = VecDeque::from(vec![5.0, 6.0, 9.0, 11.0]);

        let ans = divided_diff(&pts, &times);

        let diff_true = vec![
            Vector1::new(12.0),
            Vector1::new(1.0),
            Vector1::new(-1.0 / 6.0),
            Vector1::new(1.0 / 20.0),
        ];

        const TOL: f64 = 1.0e-10;
        for idx in 0..diff_true.len() {
            assert!((ans[idx] - diff_true[idx]) < Vector1::new(TOL));
        }
    }
}
