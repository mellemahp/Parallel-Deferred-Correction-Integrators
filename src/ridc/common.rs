/// Common Structures for RIDC integrators (ridc/common)
///
///
///
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// === End Imports ===

// Integrator Traits
#[derive(Debug, Clone, PartialEq)]
pub struct IntegOptionsParallel<N: Dim + DimName>
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
    // Tolerance to use for the convergence of the Corrector Newton Solver
    pub convergence_tol: Option<f64>,
}
impl<N: Dim + DimName> IntegOptionsParallel<N>
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
            convergence_tol: None,
        }
    }
}

pub enum IVPSolMsg<N: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N>,
{
    PROCESS(IVPSolData<N>),
    TERMINATE,
}

#[derive(Debug)]
pub struct IVPSolData<N: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Next solution estimate to correct
    pub y_nxt: VectorN<f64, N>,
    // Dynamics Function eval f(t_nxt, y_nxt)
    pub dy_nxt: VectorN<f64, N>,
    // Time at which solution estimate and were evaluated
    pub t_nxt: f64,
    // Pre-computed quadrature weights for integrating from time t0 to t_nxt
    pub weights: Option<Vec<f64>>,
}
