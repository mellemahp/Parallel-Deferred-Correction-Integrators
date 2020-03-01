/// RIDC Correctors
///
/// Defines the corrector for the RIDC method. Each corrector is spawned
/// in a separate thread and uses a polynomial interpolation and Deferred correction
/// to generate corrected solutions.
///
/// Estimates are read in to the corrector via an mpsc channel and are processed in
/// FIFO order. Each point is corrected and then that correction is on an mpsc channel
/// to either a predictor (for the final correction level in an RIDC integrator) or
/// to the next level of correction
///
/// Note: Please look at either the adaptive step or fixed step RIDC predictors for usage
/// information
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, VectorN, U1};

// local imports
use super::common::{IVPSolData, IVPSolMsg};
use crate::lagrange::quadrature::{get_weights, get_x_pow, specific_weights};
use crate::newton_raphson::newton_raphson_broyden;

// Standard library imports
use std::collections::VecDeque;
use std::marker::Send;
use std::sync::mpsc::{Receiver, Sender};

// === End Imports ===

// PROFILING
use std::time::Instant;
// END PROFILING

pub struct Corrector<N: Dim + DimName + DimMin<N> + DimSub<U1>>
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
    // Order, M, of the polynomial fit to use for quadrature. Requires M+1 points
    pub poly_order: usize,
    // Dynamics function used for the initial value problem
    dynamics: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    // Corrected Estimates of the IVP solutions
    y_ests: VecDeque<VectorN<f64, N>>,
    // Evaluations of the Dynamics function at the final corrected estimate
    fxn_evals: VecDeque<VectorN<f64, N>>,
    // Times at which function evals occur
    times: VecDeque<f64>,
    // Handle for recieving messages from channel
    pub rx: Receiver<IVPSolMsg<N>>,
    // Handle for sending messages on channel
    pub tx: Sender<IVPSolMsg<N>>,
    // Thread Number. An ID for helping with debugging
    pub id: u32,
    // Convergence tolerance for Newton solver used in the backward euler step
    convergence_tol: f64,
}

impl<N: Dim + DimName + DimMin<N> + DimSub<U1>> Corrector<N>
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
    pub fn new(
        poly_order: usize,
        dynamics: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        y_0: &VectorN<f64, N>,
        dy_0: &VectorN<f64, N>,
        t_0: f64,
        rx: Receiver<IVPSolMsg<N>>,
        tx: Sender<IVPSolMsg<N>>,
        id: u32,
        convergence_tol: f64,
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
            convergence_tol,
        }
    }

    pub fn run(&mut self) -> Result<(), &'static str> {
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
            match self.initialize(data) {
                Ok(i) => match i {
                    0 => continue,
                    1 => break,
                    _ => break,
                },
                Err(_) => break,
            }
        }
        // Main processing loop
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

            let root_sol = newton_raphson_broyden(
                root_problem,
                self.y_ests[l - i].clone(),
                self.convergence_tol,
            )
            .unwrap();

            self.y_ests[l - i - 1] = root_sol;
            self.fxn_evals[l - i - 1] = (self.dynamics)(t_n, &self.y_ests[l - i - 1]);

            let data_new = IVPSolMsg::PROCESS(IVPSolData {
                y_nxt: self.y_ests[l - i - 1].clone(),
                dy_nxt: self.fxn_evals[l - i - 1].clone(),
                t_nxt: t_n,
                weights: None,
            });
            self.tx
                .send(data_new)
                .expect("Could Not Send message from thread!");
        }
        Ok(())
    }

    fn correct(&mut self, data: IVPSolData<N>) -> Result<(), &'static str> {
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

        // === PROFILING CODE
        let now = Instant::now();
        // === END PROFILING CODE
        let root_problem = |y_n: &VectorN<f64, N>| {
            y_n - (&self.y_ests[1] + dt * (self.dynamics)(self.times[0], y_n)
                - dt * &self.fxn_evals[0]
                + &quadrature)
        };

        self.y_ests[0] =
            newton_raphson_broyden(root_problem, self.y_ests[0].clone(), self.convergence_tol)
                .expect("Couldn't converge to solution");

        // === PROFILING CODE
        println!("DURATION: {:?}", now.elapsed().as_micros());
        // === END PROFILING COD

        // re-evaluate the dynamics function
        self.fxn_evals[0] = (self.dynamics)(self.times[0], &self.y_ests[0]);

        let data_new = IVPSolMsg::PROCESS(IVPSolData {
            y_nxt: self.y_ests[0].clone(),
            dy_nxt: self.fxn_evals[0].clone(),
            t_nxt: self.times[0].clone(),
            weights: data.weights,
        });
        self.tx
            .send(data_new)
            .expect("Could not send message from thread");

        Ok(())
    }
}
