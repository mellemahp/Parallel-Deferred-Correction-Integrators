/// Runge Kutta Steppers
///
/// The process of building and using an RK stepper is roughly as follows
///     # Contruct Butcher Tableau
///     let t = Tableau{ a_vals, b_vals, c_vals };
///     # Create an Integrator Config using the tableau
///     let config = RKIntegConfig::new(t);
///     # This creates an RK-# Integrator config. If you now want to use that integrator in a specific problem
///     integrator: RKInteg = config.initialize(.. intial_conditions)
///     let y = integrator.step(t, h);
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// Local imports
use super::common::StepSimple;
use super::fixed::FixedStep;
use super::tableaus::{RkType, Tableau};

// === End Imports ===

#[derive(Debug, Clone, PartialEq)]
pub struct RKStepper<D: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    name: &'static str,
    tableau: Tableau<D>,
    stages: usize,
    rktype: RkType,
}

impl<D: DimName + Dim> RKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    pub fn new(s: &'static str, t: Tableau<D>) -> Result<Self, ()> {
        let mut ut = t.a_vals.upper_triangle();
        ut.fill_diagonal(0.0);
        match (t.a_vals.trace() == 0.0, ut.iter().sum::<f64>() == 0.0) {
            (true, true) => Ok(RKStepper {
                stages: t.b_vals.len(),
                name: s,
                tableau: t,
                rktype: RkType::Explicit,
            }),
            (_, true) => unimplemented!("No Semi-Implict Integration provided yet"),
            _ => unimplemented!("No Implicit Embedded integration provided yet"),
        }
    }
}
/// Basic step for all runge-kutta follows the following equation
///     y_{n+1} = y_n * h * Sum_{i=1}^s b_i k_i
/// where
///     k_i = f(x_n + c_i * h, y_n + h * Sum_{j=1}^s a_{ij} k_j)
///
impl<D: DimName + Dim> StepSimple for RKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    fn step<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
    ) -> VectorN<f64, N>
    where
        DefaultAllocator: Allocator<f64, N>,
    {
        match self.rktype {
            RkType::Explicit => {
                let mut ks = Vec::new();
                for i in 0..self.stages {
                    let ka_sum: VectorN<f64, N> = ks
                        .iter()
                        .enumerate()
                        .map(|(j, k)| self.tableau.a_vals[(i, j)] * k)
                        .fold(VectorN::<f64, N>::zeros(), |sum, val| sum + val);
                    ks.push(fxn(
                        t_0 + step * self.tableau.c_vals[i],
                        &(y_0 + step * ka_sum),
                    ));
                }
                let sum_bi_ki: VectorN<f64, N> = self
                    .tableau
                    .b_vals
                    .iter()
                    .enumerate()
                    .map(|(i, b)| *b * &ks[i])
                    .fold(VectorN::<f64, N>::zeros(), |sum, val| sum + val);

                y_0 + step * sum_bi_ki
            }
            _ => unimplemented!("Only Explicit Embedded methods currently supported"),
        }
    }
}

impl<D: DimName + Dim> FixedStep for RKStepper<D> where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>
{
}
// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use na::{Matrix3, Matrix4, Vector1, Vector3, Vector4};

    fn test_dyn(x: f64, _y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * x - 3.0)
    }

    fn test_dyn_2(x: f64, y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * x) - 3.0 * y
    }

    #[test]
    fn integ_test_1() {
        // 4th order test
        let a_vals = Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let c_vals = Vector4::new(0.0, 0.5, 0.5, 1.0);
        let b_vals = Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let b = Tableau {
            a_vals,
            c_vals,
            b_vals,
        };
        let r = RKStepper::new("RK4", b).unwrap();
        let ans = r.step(test_dyn, 0.0, &Vector1::new(0.0), 1.0);

        assert!((ans[0] + 0.5).abs() < 1e-7);
    }

    #[test]
    fn integ_test_2() {
        let a_vals = Matrix4::new(
            0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
        );
        let c_vals = Vector4::new(0.0, 0.5, 0.5, 1.0);
        let b_vals = Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0);
        let b = Tableau {
            a_vals,
            c_vals,
            b_vals,
        };
        let r = RKStepper::new("RK4", b).unwrap();
        let ans = r.step(test_dyn_2, 0.0, &Vector1::new(0.0), 1.0);
        assert_eq!(ans[0], 1.875);
    }

    #[test]
    fn integ_test_3() {
        let a_vals = Matrix3::new(0.0, 0.0, 0.0, 1.0 / 3.0, 0.0, 0.0, 0.0, 2.0 / 3.0, 0.0);
        let c_vals = Vector3::new(0.0, 1.0 / 3.0, 2.0 / 3.0);
        let b_vals = Vector3::new(1.0 / 4.0, 0.0, 3.0 / 4.0);
        let b = Tableau {
            a_vals,
            c_vals,
            b_vals,
        };
        let r = RKStepper::new("RK3", b).unwrap();
        let ans = r.step(test_dyn_2, 0.0, &Vector1::new(0.0), 0.5);
        assert_eq!(ans[0], 0.3125)
    }
}
