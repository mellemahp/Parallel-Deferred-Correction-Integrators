/// Adaptive time-step Revisionist Iterated Differed Corrector (RIDC) [adaptive]
///
/// Performs parallel integration of IVP using a multiple order correction
/// method
///
/// << ADD MORE >>
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, VectorN, U1};

// local imports
use super::common::{IVPSolData, IVPSolMsg, IntegOptionsParallel};
use super::corrector::Corrector;
use crate::lagrange::quadrature::{get_weights, get_x_pow, specific_weights};
use crate::runge_kutta::adaptive::{AdaptiveStep, StepValid};
use crate::runge_kutta::common::{IntegResult, StepResult};
use crate::runge_kutta::embedded::EmbeddedRKStepper;

// Standard library imports
use std::collections::VecDeque;
use std::marker::Send;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

// === End Imports ===

pub trait RIDCAdaptiveInteg: AdaptiveStep {
    fn parallel_integrator<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
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
        let atol = integ_opts
            .atol
            .unwrap_or(VectorN::<f64, N>::repeat(1e-6_f64));
        let rtol = integ_opts.rtol.unwrap_or(1e-3_f64);
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);
        let poly_order = integ_opts.poly_order.unwrap_or(3); // ONLY 3 is currently supported
        let corrector_order = integ_opts.corrector_order.unwrap_or(1);
        let restart_length = integ_opts.restart_length.unwrap_or(100);
        let corr_conv_tol = integ_opts.convergence_tol.unwrap_or(1.0e-10_f64);

        // Initialize results struct and other integration variables
        let mut results = IntegResult::new(t_0, y_0.clone());
        let t_end = t_0 + step;
        let mut sub_step = step;
        let mut step_res: StepResult<N>;
        let mut step_revision: StepValid;
        let backward: bool = step < 0.0;

        // Spawn all channels
        let mut channels: Vec<(Sender<IVPSolMsg<N>>, Receiver<IVPSolMsg<N>>)> = Vec::new();
        channels.push(mpsc::channel());
        for _ in 0..corrector_order {
            channels.push(mpsc::channel());
        }

        // root channels used by the predictor thread
        let channels_root = channels.pop().unwrap();
        let root_tx = channels_root.0;
        let mut last_rx = channels_root.1;

        // generate threads
        let mut thread_handles: Vec<thread::JoinHandle<Result<(), &'static str>>> = Vec::new();
        for i in 0..corrector_order {
            let chan = channels.pop().unwrap();
            let mut corrector = Corrector::new(
                poly_order,
                fxn,
                &y_0,
                &fxn(t_0, &y_0),
                t_0,
                last_rx,
                chan.0,
                i as u32,
                corr_conv_tol,
            );
            last_rx = chan.1;
            let handler = thread::Builder::new()
                .name(format!("THREAD {}", i))
                .spawn(move || corrector.run())
                .unwrap();
            thread_handles.push(handler);
        }
        let root_rx = last_rx;

        // initialize vals
        let mut y_last = y_0.clone();
        let mut counter = 1;
        // Used to generate the weights for the quadrature. Quadrature uses a backwards
        // form of the lagrange polynomial interpolation
        let mut times_rev: VecDeque<f64> = VecDeque::from(vec![t_0]);
        times_rev.reserve_exact(poly_order);

        // start the RK integrator
        while results.t != t_end {
            // Ensures integrator does not over-step the goal
            if (backward && sub_step.abs() > (t_end - results.t).abs())
                || (!backward && sub_step > (t_end - results.t))
            {
                sub_step = t_end - results.t;
            } else if sub_step.abs() < min_step_size {
                return Err("Step size is below minimum allowable step size");
            };

            step_res = self.step(fxn, results.t, &y_last, sub_step, &atol, rtol);
            step_revision = self.revise_step(step_res.error, sub_step);

            match step_revision {
                StepValid::Accept(nxt_step) => {
                    if counter < poly_order + 1 {
                        // Update all times
                        results.t += sub_step;
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
                        sub_step = nxt_step;
                        counter += 1;
                    } else if counter % restart_length == 0 {
                        results.t += sub_step;
                        let x_pows = get_x_pow(
                            results.times[results.times.len() - 1],
                            results.t,
                            poly_order,
                        );
                        results.times.push(results.t);

                        // rotate times into times vector
                        times_rev.pop_back().expect("Could not pop old time");
                        times_rev.push_front(results.t);

                        root_tx
                            .send(IVPSolMsg::PROCESS(IVPSolData {
                                y_nxt: step_res.value.clone(),
                                dy_nxt: step_res.dyn_eval.clone(),
                                t_nxt: results.t.clone(),
                                weights: Some(specific_weights(x_pows, &get_weights(&times_rev))),
                            }))
                            .expect("Could not send Message from [ROOT]");
                        sub_step = nxt_step;
                        counter += 1;

                        while results.states.len() < counter {
                            match root_rx.recv() {
                                Ok(msg) => match msg {
                                    IVPSolMsg::PROCESS(data) => {
                                        results.states.push(data.y_nxt);
                                    }
                                    IVPSolMsg::TERMINATE => {
                                        return Err("Somehow the root thread recieved a terminate command. Cry yourself to sleep");
                                    }
                                },
                                Err(_) => {
                                    return Err("Something went terribly wrong");
                                }
                            };
                        }
                        y_last = results.states[results.states.len() - 1].clone();
                    } else {
                        results.t += sub_step;

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
                        sub_step = nxt_step;
                        counter += 1;
                    }
                }
                StepValid::Refine(nxt_step) => {
                    sub_step = nxt_step;
                }
            }
        }
        if results.times.len() != results.states.len() {
            while results.times.len() != results.states.len() {
                match root_rx.recv() {
                    Ok(msg) => match msg {
                        IVPSolMsg::PROCESS(data) => {
                            results.states.push(data.y_nxt);
                        }
                        IVPSolMsg::TERMINATE => {
                            return Err("Somehow the root thread recieved a terminate command. Cry yourself to sleep");
                        }
                    },
                    Err(err) => {
                        println!("{:?}", err);
                        return Err("Something went terribly wrong");
                    }
                };
            }
        }
        root_tx
            .send(IVPSolMsg::TERMINATE)
            .expect("Could not send poison pill msg from [ROOT]");

        Ok(results)
    }
}

impl<D: DimName + Dim> RIDCAdaptiveInteg for EmbeddedRKStepper<D> where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>
{
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::{RK32, RKF45};
    use crate::test_fxns::{
        one_d_dynamics, one_d_solution, two_body_dyn, two_d_dynamics, two_d_solution,
        KeplerianState, IT_2_D, IV_2_D, ONE_D_INIT_TIME, ONE_D_INIT_VAL,
    };
    use na::{Vector1, Vector2, Vector6};
    use std::time::Instant;

    //#[test]
    fn test_ridc_1d() {
        let time_end = 5.0;
        let dt = time_end - ONE_D_INIT_TIME;
        let options = IntegOptionsParallel::default();
        let ans = RK32
            .parallel_integrator(
                one_d_dynamics,
                ONE_D_INIT_TIME,
                &ONE_D_INIT_VAL,
                dt,
                options,
            )
            .unwrap();

        let tol_val = Vector1::new(1e-3);
        let diff = (one_d_solution(time_end) - ans.last_y()).abs();
        println!("{:?}", diff);
        assert!(diff < tol_val);
    }

    #[test]
    fn test_ridc_2d() {
        let time_end = 5.0;
        let dt = time_end - IT_2_D;
        let options = IntegOptionsParallel::default();
        let ans = RK32
            .parallel_integrator(two_d_dynamics, IT_2_D, &IV_2_D, dt, options)
            .unwrap();
        let tol_val = Vector2::repeat(1e-7);
        let diff = (two_d_solution(time_end) - ans.last_y()).abs();
        assert!(diff < tol_val);
    }

    // Generates data for plotting the Acccuracy vs CPU time
    //#[test]
    fn test_time_to_acc_2d() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];

        let time_end = 5.0;
        let dt = time_end - IT_2_D;
        //let par_options = IntegOptionsParallel::default();

        let parallel = RK32.clone();
        let regular_32 = RK32.clone();

        // test reg runge kutta integrators
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector2::repeat(*acc)),
                rtol: Some(*acc * 10e3_f64),
                min_step: Some(1e-14_f64),
            };
            let start = Instant::now();
            let ans_reg = regular_32
                .integrate(
                    two_d_dynamics,
                    IT_2_D,
                    IV_2_D.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (two_d_solution(time_end) - ans_reg.last_y()).norm();
            println!("[RK32 2D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
        // test cash Karp 45
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector2::repeat(*acc)),
                rtol: Some(*acc * 10e3_f64),
                min_step: Some(1e-10_f64),
            };
            let start = Instant::now();
            let ans_reg = RKF45
                .clone()
                .integrate(
                    two_d_dynamics,
                    IT_2_D,
                    IV_2_D.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (two_d_solution(time_end) - ans_reg.last_y()).norm();
            println!("[RK45 2D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }

        // test cash Karp 45
        for n in 1..10 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector2::repeat(1e-6_f64)),
                rtol: Some(1e-3_f64),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(n),
                restart_length: Some(100),
                convergence_tol: Some(1e-10_f64),
            };
            let start = Instant::now();
            let ans_par = RK32
                .parallel_integrator(two_d_dynamics, IT_2_D, &IV_2_D, dt, par_options.clone())
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (two_d_solution(time_end) - ans_par.last_y()).norm();
            println!("[PARALLEL 2D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
    }

    // Generates data for plotting the Acccuracy vs CPU time
    //#[test]
    fn test_time_to_acc_1d() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];
        let time_end = 5.0;
        let dt = time_end - ONE_D_INIT_TIME;

        let parallel = RK32.clone();
        let regular_32 = RK32.clone();

        // test reg runge kutta integrators
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector1::repeat(*acc)),
                rtol: Some(*acc * 10e3_f64),
                min_step: Some(1e-14_f64),
            };
            let start = Instant::now();
            let ans_reg = regular_32
                .integrate(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    ONE_D_INIT_VAL.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
            println!("[RK32] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
        // test cash Karp 45
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector1::repeat(*acc)),
                rtol: Some(*acc * 10e3_f64),
                min_step: Some(1e-10_f64),
            };
            let start = Instant::now();
            let ans_reg = RKF45
                .clone()
                .integrate(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    ONE_D_INIT_VAL.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
            println!("[Rk45 1D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }

        // test cash Karp 45
        for n in 1..10 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector1::repeat(1e-6_f64)),
                rtol: Some(1e-3_f64),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(n),
                restart_length: Some(100),
                convergence_tol: Some(1e-10_f64),
            };
            let start = Instant::now();
            let ans_par = RK32
                .parallel_integrator(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    &ONE_D_INIT_VAL,
                    dt,
                    par_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
            println!("[PARALLEL 1D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
    }

    #[test]
    fn test_time_to_acc_kep_0001() {
        // INitial state
        let kep_init = KeplerianState::from_peri_rad(8000.0, 0.001, 0.0, 0.0, 0.0, 0.0);
        let cart_init = kep_init.into_cartesian();

        // Initialize integrators
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];

        let regular_32 = RK32.clone();
        let time_end = 3500.0;
        let dt = time_end;

        // test reg runge kutta integrators
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector6::repeat(*acc)),
                rtol: Some(*acc),
                min_step: Some(1e-12_f64),
            };
            let start = Instant::now();
            let ans_reg = regular_32
                .integrate(
                    two_body_dyn,
                    0.0,
                    cart_init.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff =
                (kep_init.propagate_to_time(time_end).into_cartesian() - ans_reg.last_y()).norm();

            println!("[RK32] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
        // test cash Karp 45
        for acc in &accs {
            let reg_options = IntegOptions {
                atol: Some(Vector6::repeat(*acc)),
                rtol: Some(*acc),
                min_step: Some(1e-12_f64),
            };
            let start = Instant::now();
            let ans_reg = RKF45
                .clone()
                .integrate(
                    two_body_dyn,
                    0.0,
                    cart_init.clone(),
                    dt,
                    reg_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff =
                (kep_init.propagate_to_time(time_end).into_cartesian() - ans_reg.last_y()).norm();
            println!("[Rk45 1D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }

        // test cash Karp 45
        for n in 1..3 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector6::repeat(1e-6_f64)),
                rtol: Some(1e-6_f64),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(n),
                restart_length: Some(100),
                convergence_tol: Some(1e-8_f64),
            };
            let start = Instant::now();
            let ans_par = RK32
                .parallel_integrator(
                    two_body_dyn,
                    0.0,
                    &cart_init.clone(),
                    dt,
                    par_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff =
                (kep_init.propagate_to_time(time_end).into_cartesian() - ans_par.last_y()).norm();
            println!("[PARALLEL 2D] TIME: {:?} | ACC_OUT: {:?}", dur, diff);
        }
    }
}
