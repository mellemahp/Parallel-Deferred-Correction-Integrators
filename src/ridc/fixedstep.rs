/// Fixed time-step Revisionist Iterated Differed Corrector (RIDC) [adaptive]
///
/// Performs parallel integration of IVP using a multiple order correction
/// method with a fixed time step
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, VectorN, U1};

// local imports
use super::base::{RIDCIntegratorBase, RIDCIntegratorFixed};
use super::common::{IVPSolData, IVPSolMsg, IntegOptionsParallel};
use crate::lagrange::quadrature::{get_weights, get_x_pow, specific_weights};
use crate::runge_kutta::base::RKStepper;
use crate::runge_kutta::common::{IntegResult, StepSimple};

// Standard library imports
use std::collections::VecDeque;
use std::marker::Send;

// === End Imports ===

impl<D: DimName + Dim> RIDCIntegratorBase for RKStepper<D> where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>
{
}

impl<D: DimName + Dim> RIDCIntegratorFixed for RKStepper<D>
where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>,
{
    fn parallel_integrator<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
        mut dt: f64,
        integ_opts: IntegOptionsParallel<N>,
    ) -> Result<IntegResult<N>, &'static str>
    where
        DefaultAllocator: Allocator<f64, N>
            + Allocator<f64, U1, N>
            + Allocator<f64, N, N>
            + Allocator<f64, <N as DimMin<N>>::Output, N>
            + Allocator<f64, <N as DimMin<N>>::Output>
            + Allocator<f64, N, <N as DimMin<N>>::Output>
            + Allocator<f64, <<N as DimMin<N>>::Output as DimSub<U1>>::Output>,
        <N as DimMin<N>>::Output: DimName,
        <N as DimMin<N>>::Output: DimSub<U1>,
        <DefaultAllocator as Allocator<f64, N>>::Buffer: Send + Sync,
    {
        // Unwrap Options to defaults
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);
        let poly_order = integ_opts.poly_order.unwrap_or(3); // ONLY 3 is currently supported
        let corrector_order = integ_opts.corrector_order.unwrap_or(1);
        let restart_length = integ_opts.restart_length.unwrap_or(100);
        let corr_conv_tol = integ_opts.convergence_tol.unwrap_or(1.0e-10_f64);

        if dt.abs() < min_step_size {
            return Err("Requested Step size is smaller than minimum step size");
        }

        // Initialize results struct and other integration variables
        let mut results = IntegResult::new(t_0, y_0.clone());
        let t_end = t_0 + step;
        let backward: bool = step < 0.0;
        if backward {
            dt = -dt;
        }

        // initialize vals
        let mut y_last = y_0.clone();
        let first_dyn_eval = &fxn(t_0, y_0);
        let mut counter = 1;

        // spawn threads
        let (root_tx, root_rx) = self.spawn_correctors(
            corrector_order,
            poly_order,
            fxn,
            t_0,
            y_0,
            first_dyn_eval,
            corr_conv_tol,
        );

        // flag to prevent infinite looping while collecting results
        let mut just_restarted = false;

        // Used to generate the weights for the quadrature. Quadrature uses a backwards
        // form of the lagrange polynomial interpolation
        let mut times_rev: VecDeque<f64> = VecDeque::from(vec![t_0]);
        times_rev.reserve_exact(poly_order);

        while results.t != t_end {
            // Ensures integrator does not over-step the goal
            if (backward && dt.abs() > (t_end - results.t).abs())
                || (!backward && dt > (t_end - results.t))
            {
                dt = t_end - results.t;
            }

            if counter < poly_order + 1 {
                just_restarted = false;
                // evaluate function
                let step_res = self.step(fxn, results.t, &y_last, dt);

                // Update all times
                results.t += dt;
                results.times.push(results.t);
                times_rev.push_front(results.t);

                // send initialization point to corrector
                root_tx
                    .send(IVPSolMsg::PROCESS(IVPSolData {
                        y_nxt: step_res.value.clone(),
                        dy_nxt: step_res.dyn_eval.clone(),
                        t_nxt: results.t.clone(),
                        weights: None,
                    }))
                    .expect("Could not send Message from [ROOT]");

                y_last = step_res.value;
                counter += 1;
            } else if (counter % restart_length == 0) && !(just_restarted) {
                // stop and wait for other threads to catch up
                self.collect_results(&root_rx, &mut results)?;
                y_last = results.states[results.states.len() - 1].clone();
                just_restarted = true;
            } else {
                just_restarted = false;
                let step_res = self.step(fxn, results.t, &y_last, dt);

                results.t += dt;

                let x_pows = get_x_pow(
                    results.times[results.times.len() - 1],
                    results.t,
                    poly_order,
                );
                results.times.push(results.t);

                // rotate times into times vector
                times_rev.pop_back().expect("Could not pop old time");
                times_rev.push_front(results.t);

                // send estimate to the corrector
                root_tx
                    .send(IVPSolMsg::PROCESS(IVPSolData {
                        y_nxt: step_res.value.clone(),
                        dy_nxt: step_res.dyn_eval.clone(),
                        t_nxt: results.t.clone(),
                        weights: Some(specific_weights(x_pows, &get_weights(&times_rev))),
                    }))
                    .expect("Could not send Message from [ROOT]");

                y_last = step_res.value;
                counter += 1;
            }
        }
        self.collect_results(&root_rx, &mut results)?;
        self.poison(root_tx, root_rx)?;
        Ok(results)
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::one_d::{
        one_d_dynamics, one_d_solution, ONE_D_INIT_TIME, ONE_D_INIT_VAL,
    };
    use crate::test_fxns::two_d::{two_d_dynamics, two_d_solution, IT_2_D, IV_2_D};
    use na::{Vector1, Vector2};

    #[test]
    fn test_ridc_1d() {
        println!("STARTING FIXED STEP RIDC TEST");
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;
        let step_size = 0.5;
        let options = IntegOptionsParallel::default();
        let ans = RK4
            .parallel_integrator(
                one_d_dynamics,
                ONE_D_INIT_TIME,
                &ONE_D_INIT_VAL,
                dt,
                step_size,
                options,
            )
            .unwrap();

        let tol_val = Vector1::new(1e-3);
        let diff = (one_d_solution(time_end) - ans.last_y()).abs();
        println!("DIFF 1d | {:?}", diff);
        assert!(diff < tol_val);
    }

    #[test]
    fn test_ridc_2d() {
        let time_end = 5.0;
        let dt = time_end - IT_2_D;
        let step = 0.5;
        let options = IntegOptionsParallel::default();
        let ans = RK4
            .parallel_integrator(two_d_dynamics, IT_2_D, &IV_2_D, dt, step, options)
            .unwrap();
        let tol_val = Vector2::repeat(1e-7);
        let diff = (two_d_solution(time_end) - ans.last_y()).abs();
        println!("DIFF 2d | {:?}", diff);
        assert!(diff < tol_val);
    }
}
