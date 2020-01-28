extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

use crate::runge_kutta::adaptive::{AdaptiveStep, StepValid};
use crate::runge_kutta::common::StepWithError;
use crate::runge_kutta::common::{StepResult, Tolerances};
use crate::runge_kutta::rk_embed::RK32;

use super::traits::AdamsPrimer;

#[derive(Clone)]
pub struct MultiStepPrimer<N: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N>,
{
    pub times: Vec<f64>,
    pub states: Vec<VectorN<f64, N>>,
    pub dyn_evals: Vec<VectorN<f64, N>>,
    pub first_step: Option<f64>,
    pub fxn: Option<fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>>,
}

impl<N: DimName + Dim> MultiStepPrimer<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    fn new(t_0: f64, y_0: &VectorN<f64, N>, dy_0: &VectorN<f64, N>) -> Self {
        MultiStepPrimer {
            times: vec![t_0],
            states: vec![y_0.clone()],
            dyn_evals: vec![dy_0.clone()],
            first_step: None,
            fxn: None,
        }
    }
    fn push(&mut self, t_new: f64, y_new: &VectorN<f64, N>, dy_new: VectorN<f64, N>) {
        self.times.push(t_new);
        self.states.push(y_new.clone());
        self.dyn_evals.push(dy_new);
    }
}
impl<N: DimName + Dim> AdamsPrimer for MultiStepPrimer<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    type OutVec = VectorN<f64, N>;
    type TolGen = Tolerances<N>;
    type Primer = Self;

    fn prime(
        order: usize,
        fxn: fn(f64, &Self::OutVec) -> Self::OutVec,
        dt_max: f64,
        t_0: f64,
        y_0: &Self::OutVec,
        tol: &Self::TolGen,
    ) -> Result<Self, &'static str> {
        // unpack and initialize options
        let abs_tol = tol
            .abs
            .clone()
            .unwrap_or(VectorN::<f64, N>::repeat(1e-6_f64));
        let rel_tol = tol.rel.unwrap_or(1e-9_f64);

        // initialize an empty results object
        let dy_0 = fxn(t_0, y_0);
        let mut results = Self::new(t_0, &y_0, &dy_0);

        // initialize loop variables
        let mut t_last = t_0;
        let mut y_last = y_0.clone();
        let mut sub_step = dt_max / (4.0 * (order + 1) as f64);

        // Take enough Steps to initialize an Mth order adams method
        let mut step_res: StepResult<N>;
        let mut step_revision: StepValid;
        while results.len() < order + 1 {
            if t_last >= t_0 + dt_max {
                return Err("Initializer over-stepped maximum step");
            }
            step_res = RK32.step(fxn, t_last, &y_last, sub_step, &abs_tol, rel_tol);
            step_revision = RK32.revise_step(step_res.error, sub_step);

            match step_revision {
                StepValid::Accept(nxt_step) => {
                    t_last += sub_step;
                    y_last = step_res.value;
                    results.push(t_last, &y_last, fxn(t_last, &y_last));
                    sub_step = nxt_step;
                }
                StepValid::Refine(nxt_step) => {
                    sub_step = nxt_step;
                }
            }
        }

        // we return the step here because the RK32 will pick an ok
        // first step for us to take with the adams predictor. All other steps
        // are selected via comparison of 2 different orders of adams predictors
        results.first_step = Some(sub_step);
        results.fxn = Some(fxn);
        Ok(results)
    }

    fn len(&self) -> usize {
        self.times.len()
    }
}
