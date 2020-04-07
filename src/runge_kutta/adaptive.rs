/// Adaptive step size Integrators based on Runge-Kutta-Fehlberg Method
///
///
extern crate nalgebra as na;
use super::common::{IntegOptions, IntegResult, RkOrder, StepResult, StepWithError};
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

pub enum StepValid {
    Accept(f64),
    Refine(f64),
}

pub trait AdaptiveStep: StepWithError + RkOrder {
    /// Returns whether step needs refinement and what ideal step would be if it does
    /// safety factor a few percent less than unity is recommended in
    /// https://people.cs.clemson.edu/~dhouse/courses/817/papers/adaptive-h-c16-2.pdf
    /// Also see GMAT math specification
    fn revise_step(&self, error: f64, step: f64) -> StepValid {
        const SF: f64 = 0.9; // recommended by GMAT
        let new_step = SF * step * (1.0 / error).powf(1.0 / (self.order() as f64 - 1.0));
        if error > 1.0 {
            StepValid::Refine(new_step)
        } else {
            StepValid::Accept(new_step)
        }
    }

    fn integrate<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: VectorN<f64, N>,
        step: f64,
        integ_opts: IntegOptions<N>,
    ) -> Result<IntegResult<N>, &'static str>
    where
        DefaultAllocator: Allocator<f64, N>,
    {
        // Unwrap Options to defaults
        let atol = integ_opts
            .atol
            .unwrap_or(VectorN::<f64, N>::repeat(1e-3_f64));
        let rtol = integ_opts.rtol.unwrap_or(1e-6_f64);
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);

        let mut results = IntegResult::new(t_0, y_0);
        let t_end = t_0 + step;
        let mut sub_step = step;
        let mut step_res: StepResult<N>;
        let mut step_revision: StepValid;
        let backward: bool = step < 0.0;

        while results.t != t_end {
            // Ensures integrator does not over-step the goal
            if (backward && sub_step.abs() > (t_end - results.t).abs())
                || (!backward && sub_step > (t_end - results.t))
            {
                sub_step = t_end - results.t;
            } else if sub_step.abs() < min_step_size {
                return Err("Step size is below minimum allowable step size");
            };

            step_res = self.step(fxn, results.t, results.last_y(), sub_step, &atol, rtol);
            step_revision = self.revise_step(step_res.error, sub_step);

            match step_revision {
                StepValid::Accept(nxt_step) => {
                    results.add_val(sub_step, step_res.value);
                    sub_step = nxt_step;
                }
                StepValid::Refine(nxt_step) => {
                    sub_step = nxt_step;
                }
            }
        }
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::embedded::EmbeddedRKStepper;
    use crate::runge_kutta::rk_embed::DOPRI78;
    use crate::runge_kutta::tableaus::EmbeddedTableau;
    use nalgebra::{Matrix4, Vector1, Vector4};

    use crate::test_fxns::two_d::{two_d_dynamics, two_d_solution, IT_2_D, IV_2_D};
    use na::Vector2;

    fn test_dyn(t: f64, y: &Vector1<f64>) -> Vector1<f64> {
        Vector1::new(5.0 * t - 3.0 * y[0])
    }

    #[test]
    // Tests the bogacki-shampine method against the Python implementation provided
    // in scipy
    fn rk23_test() {
        let rk32 = EmbeddedRKStepper::new(
            "Bogacki-Shampine 3(2)",
            EmbeddedTableau {
                a_vals: Matrix4::new(
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
                ),
                c_vals: Vector4::new(0.0, 0.5, 0.75, 1.0),
                b_hat_vals: Vector4::new(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
                b_vals: Vector4::new(7.0 / 24.0, 0.25, 1.0 / 3.0, 1.0 / 8.0),
            },
        )
        .unwrap();

        let ans = rk32
            .integrate(
                test_dyn,
                0.0,
                Vector1::new(0.0),
                10.0,
                IntegOptions::default(),
            )
            .unwrap();
        let python_t: f64 = 10.0;
        let python_y: f64 = 16.1105506676843397428910975577;
        println!("PYTHON | t: {:?}, y: {:?}", python_t, python_y);
        println!("t: {:?} | y: {:?}", ans.t, ans.last_y());
        println!("DIFF: {:?}", (python_y - ans.last_y()[0]).abs());
        const TOL_VAL: f64 = 1e-3;
        assert!((python_y - ans.last_y()[0]).abs() < TOL_VAL);
    }

    #[test]
    fn test_dopri78_2d() {
        let time_end = 1.0;
        let dt = time_end - IT_2_D;
        let options = IntegOptions::default();
        println!("STARTING 8th order integ");
        let ans = DOPRI78
            .integrate(two_d_dynamics, IT_2_D, *IV_2_D, dt, options)
            .unwrap();

        let tol_val = Vector2::repeat(1e-6);
        let diff = (two_d_solution(time_end) - ans.last_y()).abs();
        println!("{:?}", diff);
        assert!(diff < tol_val);
    }
}
