/// common structs and Traits for multiple Runge-Kutta integrators
///
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// === End Imports ===

#[derive(Debug, Clone, PartialEq)]
pub struct IntegResult<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // List of times of solutions
    pub times: Vec<f64>,
    // List of solutions associated with times
    pub states: Vec<VectorN<f64, N>>,
    // Current time of integrator
    pub t: f64,
}

impl<N: DimName + Dim> IntegResult<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    pub fn new(t_0: f64, y_0: VectorN<f64, N>) -> Self {
        IntegResult {
            times: vec![t_0],
            states: vec![y_0],
            t: t_0,
        }
    }

    pub fn last_y(&self) -> &VectorN<f64, N> {
        &self.states[self.states.len() - 1]
    }

    pub fn add_val(&mut self, step: f64, new_state: VectorN<f64, N>) {
        self.t += step;
        self.times.push(self.t);
        self.states.push(new_state);
    }
}

// Stepper Traits
pub trait StepSimple {
    fn step<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
    ) -> VectorN<f64, N>
    where
        DefaultAllocator: Allocator<f64, N>;
}
#[derive(Debug, Clone, PartialEq)]
pub struct StepResult<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Error norm of current step
    pub error: f64,
    // Solution estimate
    pub value: VectorN<f64, N>,
    // Derivative evaluation
    pub dyn_eval: VectorN<f64, N>,
}

pub trait StepWithError {
    fn step<N: DimName + Dim>(
        &self,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_0: f64,
        y_0: &VectorN<f64, N>,
        step: f64,
        atol: &VectorN<f64, N>,
        rtol: f64,
    ) -> StepResult<N>
    where
        DefaultAllocator: Allocator<f64, N>;
}

pub trait RkOrder {
    fn order(&self) -> usize;
}

// Tolerance object for RIDC integrator
pub struct Tolerances<N: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Absolute tolerance
    pub abs: Option<VectorN<f64, N>>,
    // Relative tolerance
    pub rel: Option<f64>,
}

// Integrator Traits
#[derive(Debug, Clone, PartialEq)]
pub struct IntegOptions<N: DimName + Dim>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Absolute tolerance
    pub atol: Option<VectorN<f64, N>>,
    // Relative Tolerance
    pub rtol: Option<f64>,
    // Minimum step. Errors if step goes below this threshold
    pub min_step: Option<f64>,
}
impl<N: DimName + Dim> IntegOptions<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    pub fn default() -> Self {
        Self {
            atol: None,
            rtol: None,
            min_step: None,
        }
    }
}
