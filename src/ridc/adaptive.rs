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
use crate::runge_kutta::rk_embed::RK32;

// Standard library imports
use std::collections::VecDeque;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;

// === End Imports ===

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
mod tests {
    use super::*;
}
