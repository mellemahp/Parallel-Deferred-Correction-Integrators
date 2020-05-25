/// Base Process Trait for Revisionist Iterated Differed Corrector (RIDC)
///
/// Performs parallel integration of IVP using a multiple order correction
/// method. The base implementations provided here are used for both fixed and
/// Adaptive step RIDC
///
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimMin, DimName, DimSub, VectorN, U1};

// local imports
use super::common::{IVPSolMsg, IntegOptionsParallel};
use super::corrector::Corrector;
use crate::runge_kutta::adaptive::AdaptiveStep;
use crate::runge_kutta::common::IntegResult;
use crate::runge_kutta::fixed::FixedStep;

// Standard library imports
use std::marker::Send;
use std::sync::mpsc;
use std::sync::mpsc::{Receiver, Sender};
use std::thread;
use std::time::Instant;

// === End Imports ===
pub trait RIDCIntegratorAdaptive: AdaptiveStep + RIDCIntegratorBase {
    fn parallel_integrator<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        // Dynamics function to integrate
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        // Initial time
        t_0: f64,
        // Initial state
        y_0: &VectorN<f64, N>,
        // Time to step to. IE duration of integration
        step: f64,
        // Integration options for solving IVP. See common.rs
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
        <DefaultAllocator as Allocator<f64, N>>::Buffer: Send + Sync;
}

pub trait RIDCIntegratorFixed: FixedStep + RIDCIntegratorBase {
    fn parallel_integrator<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        // Dynamics function to integrate
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        // Initial time
        t_0: f64,
        // Initial state
        y_0: &VectorN<f64, N>,
        // Time to step to. IE duration of integration
        step: f64,
        // Time step to use for fixed step integration
        dt: f64,
        // Integration options for solving IVP. See common.rs
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
        <DefaultAllocator as Allocator<f64, N>>::Buffer: Send + Sync;
}

pub trait RIDCIntegratorBase {
    // Timeout for how long to allow for thread shutdown in ms
    const SHUTDOWN_TIMEOUT_SEC: u128 = 100;

    // Generates all corrector threads for RIDC
    fn spawn_correctors<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        // Number of corrector threads to spawn
        corrector_order: usize,
        // Size of Polynomial fit to use for Stencil (stencil size = Poly Order + 1)
        poly_order: usize,
        // Dynamics function to use for integration problem
        dyn_fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        // Initial time to start all correctors at
        itime: f64,
        // Initial state to initialize all correctors with
        istate: &VectorN<f64, N>,
        // First dynamics evaluation using the initial state (istate) and initial time (itime)
        idyn: &VectorN<f64, N>,
        // Convergence tolerance to use for newton solver in corrector
        corr_conv_tol: f64,
    ) -> (Sender<IVPSolMsg<N>>, Receiver<IVPSolMsg<N>>)
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
        // Spawn all channels
        let mut channels: Vec<(Sender<IVPSolMsg<N>>, Receiver<IVPSolMsg<N>>)> = Vec::new();
        channels.push(mpsc::channel());
        for _ in 0..corrector_order {
            channels.push(mpsc::channel());
        }

        // set up root channels used by the predictor thread
        let channels_root = channels.pop().unwrap();
        let root_tx = channels_root.0;
        let mut last_rx = channels_root.1;

        // generate corrector threads
        let mut thread_handles: Vec<thread::JoinHandle<Result<(), &'static str>>> = Vec::new();
        for i in 0..corrector_order {
            let chan = channels.pop().unwrap();
            let mut corrector = Corrector::new(
                poly_order,
                dyn_fxn,
                istate,
                idyn,
                itime,
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
        (root_tx, root_rx)
    }

    // Initiates the deployment of poison pill, starting shutdown of threads
    fn poison<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        // Transmit channel for main process
        root_tx: Sender<IVPSolMsg<N>>,
        // Receiver channel for main process
        root_rx: Receiver<IVPSolMsg<N>>,
    ) -> Result<(), &'static str>
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
        root_tx
            .send(IVPSolMsg::TERMINATE)
            .expect("Could not send poison pill msg from [ROOT]");

        let time = Instant::now();
        while time.elapsed().as_micros() < Self::SHUTDOWN_TIMEOUT_SEC {
            match root_rx.recv() {
                Ok(msg) => match msg {
                    IVPSolMsg::PROCESS(_) => continue,
                    IVPSolMsg::TERMINATE => return Ok(()),
                },
                Err(_) => {
                    return Err(
                        "Something went wrong while attempting to receive messages on root rx",
                    );
                }
            };
        }
        Err("Shutdown Process Timed out")
    }

    fn collect_results<N: Dim + DimName + DimMin<N> + DimSub<U1>>(
        &self,
        // Root receiver channel to listen for results on
        root_rx: &Receiver<IVPSolMsg<N>>,
        // Results object to add results to
        results: &mut IntegResult<N>,
    ) -> Result<(), &'static str>
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
        while results.states.len() < results.times.len() {
            match root_rx.recv() {
                Ok(msg) => match msg {
                    IVPSolMsg::PROCESS(data) => {
                        results.states.push(data.y_nxt);
                    }
                    IVPSolMsg::TERMINATE => {
                        return Err(
                            "The root thread recieved a terminate command without `poison()`.",
                        );
                    }
                },
                Err(_) => {
                    return Err("Something went terribly wrong while collecting results");
                }
            };
        }
        Ok(())
    }
}
