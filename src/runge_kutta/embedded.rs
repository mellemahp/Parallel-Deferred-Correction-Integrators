/// Embedded Runge Kutta Steppers
///
/// The process of building and using an RK stepper is roughly as follows
///     # Contruct Butcher Tableau
///     let t = EmbeddedTableau{ a_vals, b_vals, b_hat_vals, c_vals };
///     # Create an Integrator Config using the tableau
///     let config = EmbeddedRKStepperConfig::new(t);
///     # This creates an RK-# Integrator config. If you now want to use that integrator in a specific problem
///     let integrator = config.initialize(.. intial_conditions);
///     let step_result = integrator.step(t, h);
///     let y = step_result.value;
///     let err = step_result.error;
///
/// The Embedded rk formula was checked against the paper:
///  "Embedded Runge-Kutta formulae with stable Equilibrium states"
///  by Higham and Hall
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// local imports
use super::adaptive::AdaptiveStep;
use super::common::{RkOrder, StepResult, StepWithError};
use super::tableaus::{EmbeddedTableau, RkType};
// === End Imports ===

// Main Structure for embedded runge kutta fehlberg stepper
#[derive(Debug, Clone, PartialEq)]
pub struct EmbeddedRKStepper<D: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    name: &'static str,
    tableau: EmbeddedTableau<D>,
    stages: usize,
    rktype: RkType,
}

impl<D: DimName + Dim> EmbeddedRKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    pub fn new(s: &'static str, t: EmbeddedTableau<D>) -> Result<Self, ()> {
        let mut ut = t.a_vals.upper_triangle();
        ut.fill_diagonal(0.0);
        match (t.a_vals.trace() == 0.0, ut.iter().sum::<f64>() == 0.0) {
            (true, true) => Ok(Self {
                stages: t.c_vals.len(),
                name: s,
                tableau: t,
                rktype: RkType::Explicit,
            }),
            (_, true) => unimplemented!("No Semi-Implict Integration provided yet"),
            _ => unimplemented!("No Implicit Embedded integration provided yet"),
        }
    }
}

impl<D: DimName + Dim> StepWithError for EmbeddedRKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    // defaults are atol = 1e-3, rtol = 1e-6 (copied from scipy defaults)
    fn step<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
        atol: &VectorN<f64, N>,
        rtol: f64,
    ) -> StepResult<N>
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
                let sum_b_hat_i_ki: VectorN<f64, N> = self
                    .tableau
                    .b_hat_vals
                    .iter()
                    .enumerate()
                    .map(|(i, b)| *b * &ks[i])
                    .fold(VectorN::<f64, N>::zeros(), |sum, val| sum + val);

                let y_n = y_0 + step * sum_bi_ki;
                let y_hat_n = y_0 + step * sum_b_hat_i_ki;
                // Pulled from pg 913 of Numerical Recipes (eq 17.2.7-9)
                let error = VectorN::<f64, N>::from_iterator(
                    (&y_hat_n - &y_n).iter().enumerate().map(|(idx, delta)| {
                        delta / (atol[idx] + rtol * y_n[idx].abs().max(y_hat_n[idx].abs()))
                    }),
                )
                .norm();
                StepResult {
                    dyn_eval: fxn(t_0 + step, &y_hat_n),
                    value: y_hat_n,
                    error,
                }
            }
            _ => unimplemented!("Only Explicit Embedded methods currently supported"),
        }
    }
}
impl<D: DimName + Dim> RkOrder for EmbeddedRKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    fn order(&self) -> usize {
        self.stages
    }
}

impl<D: DimName + Dim> AdaptiveStep for EmbeddedRKStepper<D> where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>
{
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use na::{Matrix4, Matrix6, Vector1, Vector4, Vector6};

    fn test_dyn(x: f64, y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * x - 3.0 * y[0])
    }

    fn test_dyn_2(x: f64, y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * x - 3.0 * y[0] + y[0].powf(2.0))
    }

    #[test]
    /// Tests Fehlberg's method 4(5) "True" values
    /// "True" values calculated by hand
    /// More information here:
    /// http://maths.cnam.fr/IMG/pdf/RungeKuttaFehlbergProof.pdf
    fn test_fehlberg45() {
        let a_vals = Matrix6::new(
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.25,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            3.0 / 32.0,
            9.0 / 32.0,
            0.0,
            0.0,
            0.0,
            0.0,
            1932.0 / 2197.0,
            -7200.0 / 2197.0,
            7296.0 / 2197.0,
            0.0,
            0.0,
            0.0,
            439.0 / 216.0,
            -8.0,
            3680.0 / 513.0,
            -845.0 / 4104.0,
            0.0,
            0.0,
            -8.0 / 27.0,
            2.0,
            -3544.0 / 2565.0,
            1859.0 / 4104.0,
            -11.0 / 40.0,
            0.0,
        );
        let c_vals = Vector6::new(0.0, 0.25, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5);
        let b_vals = Vector6::new(
            16.0 / 135.0,
            0.0,
            6656.0 / 12825.0,
            28561.0 / 56430.0,
            -9.0 / 50.0,
            2.0 / 55.0,
        );
        let b_hat_vals = Vector6::new(
            25.0 / 216.0,
            0.0,
            1408.0 / 2565.0,
            2197.0 / 4104.0,
            -1.0 / 5.0,
            0.0,
        );
        let tab = EmbeddedTableau {
            a_vals,
            b_vals,
            b_hat_vals,
            c_vals,
        };

        let integ = EmbeddedRKStepper::new("Fehlberg 4(5) method", tab).unwrap();
        // Initial values
        let t_0 = 0.0;
        let y_0 = Vector1::new(0.0);
        let h = 0.1;

        // Hand computed values
        let act_val: f64 = 0.022674519230769238;
        let atol = Vector1::new(1e-6_f64);
        let rtol = 1e-3_f64;
        let step_val = integ.step(test_dyn, t_0, &y_0, h, &atol, rtol);
        const TOL_VAL: f64 = 1e-8;
        assert!((act_val - step_val.value[0]).abs() < TOL_VAL);
    }

    #[test]
    /// Test the Bogacki-Shampine integration technique
    /// "True" values calculated by hand
    /// More information here:
    /// https://en.wikipedia.org/wiki/Bogacki%E2%80%93Shampine_method
    fn test_bogacki_shampine() {
        let a_vals = Matrix4::new(
            0.0,
            0.0,
            0.0,
            0.0,
            0.5,
            0.0,
            0.0,
            0.0,
            0.0,
            0.75,
            0.0,
            0.0,
            2.0 / 9.0,
            1.0 / 3.0,
            4.0 / 9.0,
            0.0,
        );
        let c_vals = Vector4::new(0.0, 0.5, 0.75, 1.0);
        let b_hat_vals = Vector4::new(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0);
        let b_vals = Vector4::new(7.0 / 24.0, 0.25, 1.0 / 3.0, 1.0 / 8.0);
        let tab = EmbeddedTableau {
            a_vals,
            b_vals,
            b_hat_vals,
            c_vals,
        };
        let integ = EmbeddedRKStepper::new("Bogacki-Shampine 2(3) Method", tab).unwrap();
        // Initial values
        let t_0 = 0.0;
        let y_0 = Vector1::new(0.0);

        const TOL_VAL: f64 = 1e-8;
        let atol = Vector1::new(1e-6_f64);
        let rtol = 1e-3_f64;

        // test against value computed from Scipy's RK23 method
        let python_step = 0.0176027714722467136521100;
        let python_val = 0.00076101;
        let step_val_2 = integ.step(test_dyn, t_0, &y_0, python_step, &atol, rtol);
        assert!((python_val - step_val_2.value[0]).abs() < TOL_VAL);

        // another test against python for a slightly more complex function
        let python_step = 9.999999999999999e-05_f64;
        let python_val = 0.0000000249975000000156198;
        let step_val_3 = integ.step(test_dyn_2, t_0, &y_0, python_step, &atol, rtol);
        assert!((python_val - step_val_3.value[0]).abs() < TOL_VAL);
    }
}
