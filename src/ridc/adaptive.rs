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
use crate::lagrange::quadrature::{get_weights, get_x_pow, specific_weights};
use crate::newton_raphson::newton_raphson_fdiff;
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

// Integrator Traits
#[derive(Debug, Clone, PartialEq)]
pub struct IntegOptionsParallel<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Absolute tolerance to use for RK predictor
    pub atol: Option<VectorN<f64, N>>,
    // Relative tolerance to use for RK predictor
    pub rtol: Option<f64>,
    // Minimum step allowed for RK predictor
    pub min_step: Option<f64>,
    // Order of polynomial fit for correctors to use
    pub poly_order: Option<usize>,
    // Number of corrections to apply. Corresponds to number of additional threads spawned
    pub corrector_order: Option<usize>,
    // Number of steps to take before restarting the integrator
    pub restart_length: Option<usize>,
}
impl<N: DimName + Dim> IntegOptionsParallel<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    pub fn default() -> Self {
        Self {
            atol: None,
            rtol: None,
            min_step: None,
            poly_order: None,
            corrector_order: None,
            restart_length: None,
        }
    }
}

pub trait RIDCInteg: AdaptiveStep {
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
            .unwrap_or(VectorN::<f64, N>::repeat(1e-4_f64));
        let rtol = integ_opts.rtol.unwrap_or(1e-7_f64);
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);
        let poly_order = integ_opts.poly_order.unwrap_or(3); // ONLY 3 is currently supported
        let corrector_order = integ_opts.corrector_order.unwrap_or(poly_order);
        let restart_length = integ_opts.restart_length.unwrap_or(50);

        //println!("[ROOT] INITIALIZED DEFAULTS");

        // Initialize results struct and other integration variables
        let mut results = IntegResult::new(t_0, y_0.clone());
        let t_end = t_0 + step;
        let mut sub_step = step;
        let mut step_res: StepResult<N>;
        let mut step_revision: StepValid;
        let backward: bool = step < 0.0;

        //println!("[ROOT] INITIALIZED RESULTS");

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

        //println!("[ROOT] INITIALIZED CHANNELS");

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
            );
            last_rx = chan.1;
            let handler = thread::Builder::new()
                .name(format!("THREAD {}", i))
                .spawn(move || corrector.run())
                .unwrap();
            thread_handles.push(handler);
        }
        let root_rx = last_rx;

        // println!("[ROOT] INITIALIZED THREADS");

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
                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: None,
                        }));
                        y_last = step_res.value;
                        sub_step = nxt_step;

                        println!("[ROOT] GENERATED INIT PT {}", counter);

                        counter += 1;
                    } else if counter % restart_length == 0 {
                        println!(
                            "[ROOT] HIT A RESTART. TIME TO COLLECT THINGS! | {}",
                            counter
                        );

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

                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: Some(specific_weights(x_pows, &get_weights(&times_rev))),
                        }));
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

                        println!(
                            "[ROOT] GENERATED REG PT {} | t: {} | s: {:?}",
                            counter, results.t, step_res.value
                        );

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
                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: Some(specific_weights(x_pows, &get_weights(&times_rev))),
                        }));
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
                            println!("[ROOT] Collected a point! Up to {}", results.states.len());
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
        println!("Somehow got here....");
        root_tx.send(IVPSolMsg::TERMINATE);
        //for i in 0..thread_handles.len() {
        //    thread_handles.pop().unwrap().join();
        //}
        Ok(results)
    }
}

impl<D: DimName + Dim> RIDCInteg for EmbeddedRKStepper<D> where
    DefaultAllocator: Allocator<f64, D> + Allocator<f64, D, D>
{
}

struct Corrector<N: Dim + DimName + DimMin<N> + DimSub<U1>>
where
    DefaultAllocator: Allocator<f64, N>
        + Allocator<f64, N, N>
        + Allocator<f64, <N as DimMin<N>>::Output, N>
        + Allocator<f64, <N as DimMin<N>>::Output>
        + Allocator<f64, N, <N as DimMin<N>>::Output>
        + Allocator<f64, <<N as DimMin<N>>::Output as DimSub<U1>>::Output>,
    <N as DimMin<N>>::Output: DimName,
    <N as DimMin<N>>::Output: DimSub<U1>,
    <DefaultAllocator as Allocator<f64, N>>::Buffer: Send + Sync,
{
    // Order, M, of the polynomial fit to use for quadrature. Requires M+1 points
    poly_order: usize,
    // Dynamics function used for the initial value problem
    dynamics: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    // Corrected Estimates of the IVP solutions
    y_ests: VecDeque<VectorN<f64, N>>,
    // Evaluations of the Dynamics function at the final corrected estimate
    fxn_evals: VecDeque<VectorN<f64, N>>,
    // Times at which function evals occur
    times: VecDeque<f64>,
    // Handle for recieving messages from channel
    rx: Receiver<IVPSolMsg<N>>,
    // Handle for sending messages on channel
    tx: Sender<IVPSolMsg<N>>,
    // Thread Number. An ID for helping with debugging
    id: u32,
}

impl<N: Dim + DimName + DimMin<N> + DimSub<U1>> Corrector<N>
where
    DefaultAllocator: Allocator<f64, N>
        + Allocator<f64, N, N>
        + Allocator<f64, <N as DimMin<N>>::Output, N>
        + Allocator<f64, <N as DimMin<N>>::Output>
        + Allocator<f64, N, <N as DimMin<N>>::Output>
        + Allocator<f64, <<N as DimMin<N>>::Output as DimSub<U1>>::Output>,
    <N as DimMin<N>>::Output: DimName,
    <N as DimMin<N>>::Output: DimSub<U1>,
    <DefaultAllocator as Allocator<f64, N>>::Buffer: Send + Sync,
{
    fn new(
        poly_order: usize,
        dynamics: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        y_0: &VectorN<f64, N>,
        dy_0: &VectorN<f64, N>,
        t_0: f64,
        rx: Receiver<IVPSolMsg<N>>,
        tx: Sender<IVPSolMsg<N>>,
        id: u32,
    ) -> Self {
        let mut y_ests: VecDeque<VectorN<f64, N>> = VecDeque::from(vec![y_0.clone()]);
        y_ests.reserve_exact(poly_order);
        let mut fxn_evals: VecDeque<VectorN<f64, N>> = VecDeque::from(vec![dy_0.clone()]);
        fxn_evals.reserve_exact(poly_order);
        let mut times: VecDeque<f64> = VecDeque::from(vec![t_0]);
        times.reserve_exact(poly_order);

        Corrector {
            poly_order,
            dynamics,
            y_ests,
            fxn_evals,
            times,
            rx,
            tx,
            id,
        }
    }

    fn run(&mut self) -> Result<(), &'static str> {
        // initialization loop
        loop {
            let data = match self.rx.recv() {
                Ok(msg) => match msg {
                    IVPSolMsg::PROCESS(data) => data,
                    IVPSolMsg::TERMINATE => {
                        break;
                    }
                },
                Err(_) => {
                    break;
                }
            };
            println!("[THREAD {}] RECIEVED INIT PT{}", self.id, self.y_ests.len());
            match self.initialize(data) {
                Ok(i) => match i {
                    0 => continue,
                    1 => break,
                    _ => break,
                },
                Err(_) => break,
            }
        }
        loop {
            let data = match self.rx.recv() {
                Ok(msg) => match msg {
                    IVPSolMsg::PROCESS(data) => data,
                    IVPSolMsg::TERMINATE => {
                        break;
                    }
                },
                Err(_) => {
                    break;
                }
            };
            self.correct(data)?;
        }
        Ok(())
    }

    fn initialize(&mut self, data: IVPSolData<N>) -> Result<u32, &'static str> {
        self.y_ests.push_front(data.y_nxt);
        self.fxn_evals.push_front(data.dy_nxt);
        self.times.push_front(data.t_nxt);

        if self.y_ests.len() == self.poly_order + 1 {
            self.first_correction()?;
            Ok(1)
        } else {
            Ok(0)
        }
    }

    fn first_correction(&mut self) -> Result<(), &'static str> {
        const CONV_TOL: f64 = 1.0e-10_f64;

        let gen_weights = get_weights(&self.times);

        let l = self.poly_order + 1;
        for i in 1..l {
            // set correction time interval
            let t_0 = self.times[l - i];
            let t_n = self.times[l - i - 1];
            let dt = t_n - t_0;

            // Generate quadrature solution over the selected interval
            let x_pows = get_x_pow(t_0, t_n, self.poly_order);
            let spec_weights = specific_weights(x_pows, &gen_weights);
            let quadrature: VectorN<f64, N> = spec_weights
                .iter()
                .zip(self.fxn_evals.iter())
                .map(|(w, y)| *w * y)
                .sum();

            // set up and solve implicit solution
            let root_problem = |y_n: &VectorN<f64, N>| {
                y_n - (&self.y_ests[l - i] + dt * (self.dynamics)(t_n, y_n)
                    - dt * &self.fxn_evals[l - i - 1]
                    + &quadrature)
            };

            let root_sol =
                newton_raphson_fdiff(root_problem, self.y_ests[l - i].clone(), CONV_TOL).unwrap();

            self.y_ests[l - i - 1] = root_sol;
            self.fxn_evals[l - i - 1] = (self.dynamics)(t_n, &self.y_ests[l - i - 1]);

            let data_new = IVPSolMsg::PROCESS(IVPSolData {
                y_nxt: self.y_ests[l - i - 1].clone(),
                dy_nxt: self.fxn_evals[l - i - 1].clone(),
                t_nxt: t_n,
                weights: None,
            });
            self.tx.send(data_new);
            println!("[THREAD {}] Sent first corr {}", self.id, i);
        }
        Ok(())
    }

    fn correct(&mut self, data: IVPSolData<N>) -> Result<(), &'static str> {
        // convergence tolerance
        const CONV_TOL: f64 = 1.0e-10_f64;

        println!("[THREAD {}] GOT TO A NORMAL CORRECTION", self.id);

        // rotate out the oldest point before adding new ones to avoid re-allocation
        self.y_ests.pop_back().expect("Could not append new state");
        self.fxn_evals
            .pop_back()
            .expect("Could not append new dynamics evaluation");
        self.times.pop_back().expect("Could not append new time");

        // add the new points from the IVP message
        self.y_ests.push_front(data.y_nxt);
        self.fxn_evals.push_front(data.dy_nxt);
        self.times.push_front(data.t_nxt);

        // compute correction
        let quadrature: VectorN<f64, N> = data
            .weights
            .clone()
            .unwrap()
            .iter()
            .zip(self.fxn_evals.iter())
            .map(|(w, y)| *w * y)
            .sum();

        let dt = self.times[0] - self.times[1];
        let root_problem = |y_n: &VectorN<f64, N>| {
            y_n - (&self.y_ests[1] + dt * (self.dynamics)(self.times[0], y_n)
                - dt * &self.fxn_evals[0]
                + &quadrature)
        };

        self.y_ests[0] = newton_raphson_fdiff(root_problem, self.y_ests[0].clone(), CONV_TOL)
            .expect("Couldn't converge to solution");

        println!("[THREAD {}] SURVIVED THE SCARY ROOT PROBLEM!", self.id);

        // re-evaluate the dynamics function
        self.fxn_evals[0] = (self.dynamics)(self.times[0], &self.y_ests[0]);

        println!("[THREAD {}] Re-evaluated the function", self.id);

        let data_new = IVPSolMsg::PROCESS(IVPSolData {
            y_nxt: self.y_ests[0].clone(),
            dy_nxt: self.fxn_evals[0].clone(),
            t_nxt: self.times[0].clone(),
            weights: data.weights,
        });
        self.tx.send(data_new);
        println!("[THREAD {}] Sent a correction!", self.id);

        Ok(())
    }
}

enum IVPSolMsg<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    PROCESS(IVPSolData<N>),
    TERMINATE,
}

#[derive(Debug)]
struct IVPSolData<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    y_nxt: VectorN<f64, N>,
    dy_nxt: VectorN<f64, N>,
    t_nxt: f64,
    weights: Option<Vec<f64>>,
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::rk_embed::RK32;
    use crate::test_fxns::{two_d_dynamics, two_d_solution, IT_2_D, IV_2_D};

    #[test]
    fn test_ridc() {
        let time_end = 5.0;
        let dt = time_end + IT_2_D;
        let options = IntegOptionsParallel::default();
        let ans = RK32
            .parallel_integrator(two_d_dynamics, IT_2_D, &IV_2_D, dt, options)
            .unwrap();
        println!("EST | {:?}", ans.clone().states);
        println!("TRUE | {:?}", two_d_solution(time_end));
        println!();
        for i in 0..ans.times.len() {
            println!(
                "{} DIFF | T: {} | S: {:?}",
                i,
                ans.clone().times[i],
                two_d_solution(ans.clone().times[i]) - ans.clone().states[i]
            );
        }
    }
}
