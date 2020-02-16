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
    fn parallel_integrater<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
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
            .unwrap_or(VectorN::<f64, N>::repeat(1e-3_f64));
        let rtol = integ_opts.rtol.unwrap_or(1e-6_f64);
        let min_step_size = integ_opts.min_step.unwrap_or(1e-10_f64);
        let poly_order = integ_opts.poly_order.unwrap_or(3); // ONLY 3 is currently supported
        let corrector_order = integ_opts.corrector_order.unwrap_or(poly_order);
        let restart_length = integ_opts.restart_length.unwrap_or(100);

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
                &VectorN::<f64, N>::zeros(),
                t_0,
                last_rx,
                chan.0,
            );
            last_rx = chan.1;
            let handler = thread::spawn(move || corrector.run());
            thread_handles.push(handler);
        }
        let root_rx = last_rx;

        // start the integrator
        let mut y_last = y_0.clone();
        let mut counter = 1;
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
                        results.t += sub_step;
                        results.times.push(results.t);
                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: None,
                        }));
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

                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: Some(specific_weights(
                                x_pows,
                                &get_weights(&VecDeque::from(results.times.clone())),
                            )),
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
                        let x_pows = get_x_pow(
                            results.times[results.times.len() - 1],
                            results.t,
                            poly_order,
                        );
                        results.times.push(results.t);
                        root_tx.send(IVPSolMsg::PROCESS(IVPSolData {
                            y_nxt: step_res.value.clone(),
                            // TODO this should really not be re-computed here
                            dy_nxt: fxn(results.t, &step_res.value),
                            t_nxt: results.t.clone(),
                            weights: Some(specific_weights(
                                x_pows,
                                &get_weights(&VecDeque::from(results.times.clone())),
                            )),
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
        }
        root_tx.send(IVPSolMsg::TERMINATE);
        for i in 0..thread_handles.len() {
            thread_handles.pop().unwrap().join();
        }
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
    // Evaluations of the Dynamics function at the final corrected estimate
    fxn_evals: VecDeque<VectorN<f64, N>>,
    // Corrected Estimates of the IVP solutions
    fxn_ests: VecDeque<VectorN<f64, N>>,
    // Times at which function evals occur
    times: VecDeque<f64>,
    // Handle for recieving messages from channel
    rx: Receiver<IVPSolMsg<N>>,
    // Handle for sending messages on channel
    tx: Sender<IVPSolMsg<N>>,
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
    ) -> Self {
        let mut fxn_evals: VecDeque<VectorN<f64, N>> = VecDeque::from(vec![y_0.clone()]);
        fxn_evals.reserve_exact(poly_order);
        let mut fxn_ests: VecDeque<VectorN<f64, N>> = VecDeque::from(vec![dy_0.clone()]);
        fxn_ests.reserve_exact(poly_order);
        let mut times: VecDeque<f64> = VecDeque::from(vec![t_0]);
        times.reserve_exact(poly_order);

        Corrector {
            poly_order,
            dynamics,
            fxn_evals,
            fxn_ests,
            times,
            rx,
            tx,
        }
    }

    fn run(&mut self) -> Result<(), &'static str> {
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
            if self.fxn_ests.len() < self.poly_order + 1 {
                self.initialize(data)?;
            } else {
                self.correct(data)?;
            }
        }
        Ok(())
    }

    fn initialize(&mut self, data: IVPSolData<N>) -> Result<(), &'static str> {
        self.fxn_ests.push_front(data.y_nxt);
        self.fxn_evals.push_front(data.dy_nxt);
        self.times.push_front(data.t_nxt);

        if self.fxn_ests.len() == self.poly_order + 1 {
            self.first_correction()?;
        }
        Ok(())
    }

    fn first_correction(&mut self) -> Result<(), &'static str> {
        const CONV_TOL: f64 = 1.0e-7_f64;

        let gen_weights = get_weights(&self.times);
        let l = self.times.len();
        for i in 0..l {
            // Find new quadrature
            let t_0 = self.times[l - i - 2];
            let t_n = self.times[l - i - 1];
            let dt = t_n - t_0;
            let x_pows = get_x_pow(t_0, t_n, self.poly_order);
            let spec_weights = specific_weights(x_pows, &gen_weights);
            let quadrature: VectorN<f64, N> = spec_weights
                .iter()
                .zip(self.fxn_evals.iter())
                .map(|(w, y)| *w * y)
                .sum();

            let root_problem = |y_n: &VectorN<f64, N>| {
                y_n - (&self.fxn_ests[l - i - 1]
                    - dt * (self.dynamics)(t_n, y_n)
                    - dt * &self.fxn_evals[l - i - 1]
                    + &quadrature)
            };
            self.fxn_ests[l - i - 1] =
                newton_raphson_fdiff(root_problem, self.fxn_ests[l - i - 1].clone(), CONV_TOL)
                    .unwrap();
            self.fxn_evals[l - i - 1] = (self.dynamics)(t_n, &self.fxn_ests[l - i - 1]);

            let data_new = IVPSolMsg::PROCESS(IVPSolData {
                y_nxt: self.fxn_ests[l - i - 1].clone(),
                dy_nxt: self.fxn_evals[l - i - 1].clone(),
                t_nxt: t_n,
                weights: None,
            });
            self.tx.send(data_new);
        }
        Ok(())
    }

    fn correct(&mut self, data: IVPSolData<N>) -> Result<(), &'static str> {
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
mod tests {}
