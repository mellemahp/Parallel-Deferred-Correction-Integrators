/// Butcher Tableau for both implicit and Explicit RK methods
///
/// For an s-stage RK method the Butcher Tableau contains:
///     c = [c_1, c_2, ..., c_s]^T
///     b = [b_1, b_2, ..., c_s]
///     A = [a_ik] (s x s matrix)
///
/// Typically the Tableau is represented in the following form
///
/// c_1 | a_11  a_12  ...  a_1s
/// c_2 | a_21  a_22  ...  a_2s
///  .  |
/// c_s | a_s1  a_s2  ...  a_ss
/// ----|----------------------
///     | b_1   b_2   ...  b_s
///
/// A tableau can define one of 4 types of RK integrators
///
/// (1) Explicit:  a_ik = 0  for k > i, k = 1,2, ..., s -> A is strictly lower Triangular
/// (2) Semi-Explicit: a_ik = 0  for k >= i, k = 1, 2, ..., s -> A is triangular
/// (3) Implicit Method: a_ik =/= 0 for some k >= i, k = 1, 2, ..., s -> A is not lower triangular
/// (4) Diagonally-implicit method: a_ii = const for i = 1, 2, ... , s
///
/// NOTE: This library does not currently support Diagonally-implicit methods
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, MatrixMN, VectorN};

// Types of Runge-Kutta at tableau could define
#[derive(Debug, Clone, PartialEq)]
pub enum RkType {
    Explicit,
    SemiImplicit,
    Implicit,
}

#[derive(Debug, Clone, PartialEq)]
pub struct Tableau<D: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    pub a_vals: MatrixMN<f64, D, D>,
    pub c_vals: VectorN<f64, D>,
    pub b_vals: VectorN<f64, D>,
}

/// Butcher Tableau for Embedded RK methods
///
/// For an s-stage RK method the Butcher Tableau contains:
///     c  = [c_1, c_2, ..., c_s]^T
///     b  = [b_1, b_2, ..., b_s]
///     b' = [b'_1, b'_2, ..., b'_s]
///     A  = [a_ik] (s x s matrix)
///
/// A tableau for this system is typically written as:
///
//     c_1 | a_11  a_12  ...  a_1s
//     c_2 | a_21  a_22  ...  a_2s
//      .  |
//     c_s | a_s1  a_s2  ...  a_ss
//     ----|----------------------
//         | b_1   b_2   ...  b_s
//         | b'_1  b'_2  ...  b'_s
///
/// Note: The B' values give the actual solution, while the B values give only
/// the correction term used to compute error. Typically the returned values are
/// of lower order than the correction value for stability.
///
/// Note: This library currently support only explicit embedded methods
/// I.E. a_ik = 0  for k > i, k = 1,2, ..., s -> A is strictly lower Triangular
/// TODO: Add implicit embeddings some time
///
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedTableau<D: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    pub a_vals: MatrixMN<f64, D, D>,
    pub c_vals: VectorN<f64, D>,
    pub b_vals: VectorN<f64, D>,
    pub b_hat_vals: VectorN<f64, D>,
}

/// Checks that row-sum condition holds for a given tableau
/// NOTE: As discussed (here)[https://ntrs.nasa.gov/archive/nasa/casi.ntrs.nasa.gov/20160005923.pdf]
///     the row-sum condition does not need to hold for all tableaus. So, for generality we do not
///     check for that condition through types or on build. Instead the following is provided to check
///     the row-sum condition if the user needs
///
pub fn check_row_sum<D: DimName + Dim>(tab: &Tableau<D>) -> bool
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    let (rows, cols) = tab.a_vals.shape();
    let mut row_sum: f64;
    for r_idx in 0..rows - 1 {
        row_sum = 0.0;
        for c_idx in 0..cols - 1 {
            row_sum += tab.a_vals[(r_idx, c_idx)]
        }
        if row_sum != tab.c_vals[r_idx] {
            return false;
        }
    }
    true
}

#[cfg(test)]
mod tests {
    use super::*;
    use nalgebra::{Matrix4, Vector4};

    #[test]
    fn tableau_manual() {
        let a_vals = Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let c_vals = Vector4::new(0.0, 0.5, 0.5, 1.0);
        let b_vals = Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let _b = Tableau {
            a_vals,
            b_vals,
            c_vals,
        };
    }

    #[test]
    fn row_sum() {
        let a_vals = Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let c_vals = Vector4::new(0.0, 0.5, 0.5, 1.0);
        let b_vals = Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let b = Tableau {
            a_vals,
            b_vals,
            c_vals,
        };
        assert_eq!(true, check_row_sum(&b))
    }
}
