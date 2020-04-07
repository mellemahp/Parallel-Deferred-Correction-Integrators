/// Fixed step Runge Kutta Steppers
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use super::common::{IntegOptions, IntegResult, StepSimple};
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// === End Imports ===

pub trait FixedStep: StepSimple {
    // Note: Step should be strictly positive. If it is negative
    // it will be changed to a positive value.
    fn integrate<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: VectorN<f64, N>,
        dt: f64,
        step: f64,
        integ_opts: IntegOptions<N>,
    ) -> Result<IntegResult<N>, &'static str>
    where
        DefaultAllocator: Allocator<f64, N>,
    {
        // extract options
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);
        if step.abs() < min_step_size {
            return Err("Requested Step size is smaller than minimum step size");
        }

        // initialize results
        let mut results = IntegResult::new(t_0, y_0);
        let t_end = t_0 + dt;
        let backward: bool = dt < 0.0;
        let mut step = step.abs();
        if backward {
            step = -step;
        }

        while results.t != t_end {
            // Ensures integrator does not over-step the goal
            if (backward && step.abs() > (t_end - results.t).abs())
                || (!backward && step > (t_end - results.t))
            {
                step = t_end - results.t;
            }
            let res = self.step(fxn, results.t, results.last_y(), step);
            results.add_val(step, res.value);
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::rk_simp::RK2;
    use nalgebra::Vector1;

    fn test_dyn(x: f64, y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * x - 3.0 * y[0])
    }

    #[test]
    fn test_fixed_integ_rk3() {
        let ans = RK2.integrate(
            test_dyn,
            0.0,
            Vector1::new(0.0),
            10.0,
            0.0001,
            IntegOptions::default(),
        );
        println!("{:?}", ans);
    }
}
