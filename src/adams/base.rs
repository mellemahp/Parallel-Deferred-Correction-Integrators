/// Base of adams multi-step integrator
///
/// Most of this process is derived from the work laid out in
/// "Computer Solutions to Differential Equations" in the chapter
/// on "Efficient implementation of the Adams method"
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

// local imports
use super::primer::MultiStepPrimer;
use super::traits::{
    AdamsCorrector, AdamsInit, AdamsPredictor, AdamsPrimer, AdamsStepper, AdamsUpdate,
};
use crate::lagrange::div_diff::divided_diff;
use std::collections::VecDeque;

// === End Imports ===

#[derive(Debug, PartialEq, Clone)]
pub enum AdamState {
    Uninitialized,
    Initialized,
}

#[derive(Clone)]
pub struct AdamsData<N: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // Order of polynomial fit to use for integration
    order: usize,
    // Time deltas (t_n - t_i)
    dtks: VecDeque<f64>,
    // Products of time deltas up to t_n
    psis_n: VecDeque<f64>,
    // Products of time deltas up to t_{n+1}
    psis_np1: VecDeque<f64>,
    // Beta coefficients for adams integrator
    betas: VecDeque<f64>,
    //
    alphas: VecDeque<f64>,
    // Divided differences
    phis: VecDeque<VectorN<f64, N>>,
    // Recursively defined quadrature coefficients
    gs: VecDeque<f64>,
    // Error divided difference
    phi_e: VecDeque<VectorN<f64, N>>,
    // Intermediate divided diffs
    phi_stars: VecDeque<VectorN<f64, N>>,
    // Dynamics function to integrate
    fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    // Times of polynomial fit
    times: VecDeque<f64>,
    // Evaluations of the dynamics function at the times above
    dyn_evals: VecDeque<VectorN<f64, N>>,
    // current state estimate
    state: VectorN<f64, N>,
    // Initial state
    init: AdamState,
}
impl<N: Dim + DimName> AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    pub fn from_primer(primer: MultiStepPrimer<N>) -> Self {
        let order = primer.len() - 1;
        let mut phi_vec = VecDeque::<VectorN<f64, N>>::with_capacity(order + 1);
        phi_vec.push_back(primer.states.first().unwrap().clone());

        AdamsData {
            order,
            dtks: VecDeque::<f64>::with_capacity(order + 1),
            psis_n: VecDeque::<f64>::with_capacity(order),
            psis_np1: VecDeque::<f64>::with_capacity(order),
            betas: VecDeque::<f64>::with_capacity(order + 1),
            alphas: VecDeque::<f64>::with_capacity(order),
            phis: phi_vec,
            gs: VecDeque::<f64>::with_capacity(order + 1),
            phi_e: VecDeque::<VectorN<f64, N>>::with_capacity(order + 1),
            phi_stars: VecDeque::<VectorN<f64, N>>::with_capacity(order + 1),
            fxn: primer.fxn.unwrap(),
            times: VecDeque::from(primer.times),
            dyn_evals: VecDeque::from(primer.dyn_evals),
            state: primer.states.last().unwrap().clone(),
            init: AdamState::Uninitialized,
        }
    }
}

impl<N: Dim + DimName> AdamsInit for AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    // computes all the delta t's for the data
    fn init_dtks(&mut self, t_kp1: f64) {
        for idx in 0..self.order {
            self.dtks.push_back(self.times[idx + 1] - self.times[idx]);
        }
        self.dtks.push_back(t_kp1 - self.times[self.order]);
    }

    fn init_psis_n(&mut self) {
        let mut psi_sum: f64 = 0.0;
        for i in 1..=self.order {
            psi_sum += self.dtks[self.dtks.len() - 1 - i];
            self.psis_n.push_back(psi_sum);
        }
    }

    fn init_psis_np1(&mut self) {
        let mut psi_sum: f64 = 0.0;
        for i in 1..=self.order {
            psi_sum += self.dtks[self.dtks.len() - i];
            self.psis_np1.push_back(psi_sum);
        }
    }

    fn init_betas(&mut self) {
        self.betas.push_back(1.0);
        let mut beta_prod: f64;
        for i in 1..=self.order {
            beta_prod = self.psis_np1[0] / self.psis_n[0];
            for j in 1..i {
                beta_prod *= self.psis_np1[j] / self.psis_n[j];
            }
            self.betas.push_back(beta_prod);
        }
    }

    fn init_alphas(&mut self, t_nxt: f64) {
        let step = t_nxt - self.times.back().unwrap();
        for val in self.psis_np1.iter() {
            self.alphas.push_back(step / val);
        }
    }

    fn init_phis(&mut self) {
        let divided_diff_vec = divided_diff(&self.dyn_evals, &self.times);
        let mut mult_res = VectorN::<f64, N>::repeat(1.0);
        for (idx, psi) in self.psis_n.iter().enumerate() {
            mult_res *= *psi;
            self.phis
                .push_back(mult_res.component_mul(&divided_diff_vec[idx]));
        }
    }

    fn init_adams(&mut self, t_nxt: f64) {
        self.init_dtks(t_nxt);
        self.init_psis_n();
        self.init_psis_np1();
        self.init_alphas(t_nxt);
        self.init_betas();
        self.init_phis();
        self.init = AdamState::Initialized;
    }
}

impl<N: Dim + DimName> AdamsPredictor for AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    type OutVec = VectorN<f64, N>;

    fn get_g1s(&mut self) {
        self.gs.clear();
        for i in 0..=self.order {
            self.gs.push_back(Self::gen_g(i, 1, &self.alphas));
        }
    }

    fn get_phi_star(&mut self) {
        self.phi_stars.clear();
        for idx in 0..=self.order {
            self.phi_stars.push_back(self.betas[idx] * &self.phis[idx]);
        }
    }

    fn get_phi_e(&mut self) {
        self.phi_e.clear();
        self.phi_e.push_back(VectorN::<f64, N>::zeros());
        for idx in 1..self.order {
            self.phi_e
                .push_back(&self.phi_e[idx - 1] + &self.phi_stars[self.order + 1 - idx]);
        }
    }

    fn predict(&self, m: usize, step: f64) -> VectorN<f64, N>
    where
        DefaultAllocator: Allocator<f64, N>,
    {
        let mut sum = VectorN::<f64, N>::zeros();
        for i in 0..=m {
            sum += self.gs[i] * &self.phi_stars[i];
        }
        &self.state + step * sum
    }
}

impl<N: Dim + DimName> AdamsCorrector for AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    type OutVec = VectorN<f64, N>;

    fn correct(
        &self,
        step: f64,
        t_nxt: f64,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        prediction: &VectorN<f64, N>,
    ) -> VectorN<f64, N> {
        prediction
            + step * self.gs.back().unwrap() * (fxn(t_nxt, prediction) - self.phi_e.back().unwrap())
    }
}
impl<N: Dim + DimName> AdamsUpdate for AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    fn update_phis(&mut self) {
        // clear out space for the new phi in the buffer. This prevents re-allocation of the ring buffer
        self.phis.pop_back();
        let phi_kp1 = self.dyn_evals.front().unwrap() - self.phi_e.back().unwrap();
        self.phis.push_front(phi_kp1.clone());
        // start with k+1
        for idx in 1..=self.order {
            self.phis[self.order - idx] = &self.phi_e[self.order + 1 - idx] + &phi_kp1;
        }
    }
}

impl<N: Dim + DimName> AdamsStepper for AdamsData<N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    type OutVec = VectorN<f64, N>;

    fn step(
        &mut self,
        order: usize,
        fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
        t_nxt: f64,
    ) -> Result<VectorN<f64, N>, &'static str> {
        if order > self.order {
            return Err("Order requested is larger than integrator order");
        }
        use AdamState::*;
        match self.init {
            Uninitialized => {
                self.init_adams(t_nxt);
            }
            Initialized => {
                // Update coeffcients for new step
            }
        }
        self.get_g1s();
        self.get_phi_star();
        self.get_phi_e();

        // Run predictions
        let step = t_nxt - self.times.back().unwrap();
        let prediction = self.predict(order, step);

        // evaluate derivative, correct, and re-evaluate derivative
        let corrected = &prediction
            + step
                * self.gs.back().unwrap()
                * (fxn(t_nxt, &prediction) - self.phi_e.back().unwrap());
        self.dyn_evals.pop_front();
        self.dyn_evals.push_back(fxn(t_nxt, &corrected));
        self.state = corrected.clone();
        self.times.pop_front();
        self.times.push_back(t_nxt);

        // Re-update divided differences
        //self.update_phis();

        Ok(corrected)
    }
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::common::Tolerances;
    use crate::test_fxns::{two_d_dynamics, two_d_solution, IT_2_D, IV_2_D};
    use na::Vector2;

    #[test]
    fn test_gen() {
        let m: usize = 4;
        let tol = Tolerances {
            abs: Some(Vector2::new(1.0e-6, 1.0e-6)),
            rel: Some(1.0e-9_f64),
        };
        let max_step_0 = 0.4;

        let primer =
            MultiStepPrimer::prime(m, two_d_dynamics, max_step_0, IT_2_D, &IV_2_D, &tol).unwrap();
        let step = primer.first_step.unwrap();
        println!("STEP | {:?}", step);
        let t_nxt = primer.times.last().unwrap() + step;
        let mut adams_stepper = AdamsData::from_primer(primer);
        let ans = adams_stepper.step(m, two_d_dynamics, t_nxt);
        println!("EST   | {:?}", ans);
        println!("TRUTH | {:?}", two_d_solution(t_nxt));
    }
}
