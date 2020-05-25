/// Divided Differences (div_diff)
///
/// Provides a method to find the Lagrange Polynomial Coefficients via divided differencing
///
/// Divided diff algorithm from:
/// https://www.math.usm.edu/lambers/mat460/fall09/lecture17.pdf
/// https://www.uio.no/studier/emner/matnat/math/MAT-INF1100/h10/kompendiet/kap9.pdf
///
/// # Background
/// Given the following data:
/// points: [f(t_n), f(t_{n-1}), f(t_{n-2})]
/// times:  [t_n, t_{n-1}, t_{n-2}]
///
/// The Lagrange polynomial fit (backwards fit) for the data above is:
///     
/// p_n(t) = \sum_{j=0}^m c_j N_j(x) \\
/// N_j(x) = \prod_{k=0}^{j-1}(x - x_k) \quad j=1,...,n
///
/// The coefficients $c_j$ above are defined by the divided differences:
/// c_j = f[t_n, ..., t_{n-j}]
///
/// where the divided difference is defined as:
/// f[t_0] = f(t_0)
/// f[t_0, t_1] = (f[t_1] - f[t_0]) / (t_1 - t_0)
/// f[t_0, t_2, t_2] = f[t_1, t_2] - f[]
///
/// The definition of the divided difference above clearly lends itself to a recursive method of computation.
/// To find the divided differences for a lagrange polynomial we construct a table the
/// has the following structure (the example here would be for a 3rd order Lagrange polynomial):
///
/// |------------|---------------------|------------------------------|-----------------------------------|
/// | f[t_n]     | f[t_n, t_{n-1}]     | f[t_n, t_{n-1}, t_{n-2}]     | f[t_n, t_{n-1}, t_{n-2}, t_{n-3}] |
/// |------------|---------------------|------------------------------|-----------------------------------|
/// | f[t_{n-1}] | f[t_{n-1}, t_{n-2}] | f[t_{n-1}, t_{n-2}, t_{n-3}] | Nil                               |
/// |------------|---------------------|------------------------------|-----------------------------------|
/// | f[t_{n-2}] | f[t_{n-2}, t_{n-3}] | Nil                          | Nil                               |
/// |------------|---------------------|------------------------------|-----------------------------------|
/// | f[t_{n-3}] | Nil                 | Nil                          | Nil                               |
/// |------------|---------------------|------------------------------|-----------------------------------|
///
/// Our final answer for the lagrange coefficients is the top row of this table. Note that all values in this
/// table are generically vectors.
///
/// Note: because we are going from $f[t_n] -> f[t_{n-m}]$ this is called the backwards divided difference
///
/// # Updating Divided differences
/// One of the nicest parts of using the divided differences formulation of the Lagrange polynomial is that
/// updating the fit with a new point becomes fairly easy. Most importantly we only need the previous set of
/// divided differences and times and the new point and time. For example lets say we want to move our third
/// order fit from above:
///
/// |------------|---------------------|------------------------------|-----------------------------------|
/// | f[t_n]     | f[t_n, t_{n-1}]     | f[t_n, t_{n-1}, t_{n-2}]     | f[t_n, t_{n-1}, t_{n-2}, t_{n-3}] |
/// |------------|---------------------|------------------------------|-----------------------------------|
///
/// To encompass a new point f(t_{n+1}) at time t_{n+1}. So we form a new table
/// |------------|---------------------|------------------------------|-------------------------------------|
/// | f[t_{n+1}] | f[t_{n+1}, t_{n}]   | f[t_{n+1}, t_{n}, t_{n-1}]   | f[t_{n+1}, t_{n}, t_{n-1}, t_{n-2}] |
/// |------------|---------------------|------------------------------|-------------------------------------|
/// | f[t_n]     | f[t_n, t_{n-1}]     | f[t_n, t_{n-1}, t_{n-2}]     | f[t_n, t_{n-1}, t_{n-2}, t_{n-3}]   |
/// |------------|---------------------|------------------------------|-------------------------------------|
///
/// Notice how every point on the top row can can be computed with only the previous set of divided differences
/// The algorithm for computing these new divided differences (f[t_{n+1}]) is:
///
/// f[t_{n+1},...,t_{n+1-j}] = (f[t_n,...,t_{n-j}] - f[t_{n+1},..., t_{n+1-j}]) / (t_{n+1-j} - t_{n+1})
///
/// This is implemented below to provide the rapid computation of a moving Lagrange polynomial fit
///
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// standard library
use std::collections::VecDeque;

// === Begin Imports ===

/// Computes a set of Divided difference weights from a set of Points and times
/// This is a vector function where all weights are functions
///
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
    // insert first column
    for i in 0..n {
        diffs[i][0] = points[i].clone();
    }
    // recurse to victory
    for j in 1..n {
        for i in 0..n - j {
            diffs[i][j] = (&diffs[i + 1][j - 1] - &diffs[i][j - 1]) / (times[i + j] - times[i])
        }
    }
    diffs[0].clone()
}

/// Updates divided difference weights one point and time forward
/// This function maintains the order of the original divided difference fit
/// and allows users to move the "window" of the Lagrange polynomial fit one
/// timestep
///
pub fn update_diff<N: Dim + DimName>(
    old_diffs: Vec<VectorN<f64, N>>,
    times: &VecDeque<f64>,
    nxt_point: &VectorN<f64, N>,
    nxt_time: f64,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let n = old_diffs.len();
    let mut new_diffs = vec![VectorN::<f64, N>::zeros(); n];
    new_diffs[0] = nxt_point.clone();

    for i in 1..n {
        new_diffs[i] = (&old_diffs[i - 1] - &new_diffs[i - 1]) / (times[i - 1] - nxt_time)
    }

    new_diffs
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use na::Vector1;

    #[test]
    fn test_divided_difference() {
        // Case testing against geeksforgeeks.com
        // NOTE: remember this is a _backwards_ difference
        //
        let pts = VecDeque::from(vec![
            Vector1::new(16.0),
            Vector1::new(14.0),
            Vector1::new(13.0),
            Vector1::new(12.0),
        ]);
        let times = VecDeque::from(vec![11.0, 9.0, 6.0, 5.0]);

        let ans = divided_diff(&pts, &times);

        let diff_true = vec![
            Vector1::new(16.0),
            Vector1::new(1.0),
            Vector1::new(2.0 / 15.0),
            Vector1::new(1.0 / 20.0),
        ];

        const TOL: f64 = 1.0e-10;
        for idx in 0..diff_true.len() {
            assert!((ans[idx] - diff_true[idx]) < Vector1::new(TOL));
        }
    }

    #[test]
    fn test_update_difference() {
        // This test was computed by hand
        // Generate initial divided difference
        let pts = VecDeque::from(vec![
            Vector1::new(16.0),
            Vector1::new(14.0),
            Vector1::new(13.0),
            Vector1::new(12.0),
        ]);
        let times = VecDeque::from(vec![11.0, 9.0, 6.0, 5.0]);

        let old_diffs = divided_diff(&pts, &times);
        let nxt_time = 12.0;
        let nxt_point = Vector1::new(18.0);

        let diff_true = vec![
            Vector1::new(18.0),
            Vector1::new(2.0),
            Vector1::new(1.0 / 3.0),
            Vector1::new(1.0 / 30.0),
        ];
        let ans = update_diff(old_diffs, &times, &nxt_point, nxt_time);

        assert!(diff_true.len() == 4);
        const TOL: f64 = 1.0e-10;
        for idx in 0..diff_true.len() {
            assert!((ans[idx] - diff_true[idx]) < Vector1::new(TOL));
        }
    }
}
