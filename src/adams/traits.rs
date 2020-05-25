/// Traits for the adams multi-step integrator
///
///
// === Begin Imports ===
// standard library imports
use std::collections::VecDeque;

// === End Imports ===

pub trait AdamsPrimer {
    type OutVec;
    type TolGen;
    type Primer;

    fn prime(
        order: usize,
        fxn: fn(f64, &<Self as AdamsPrimer>::OutVec) -> <Self as AdamsPrimer>::OutVec,
        dt_max: f64,
        t_0: f64,
        y_0: &<Self as AdamsPrimer>::OutVec,
        tol: &<Self as AdamsPrimer>::TolGen,
    ) -> Result<<Self as AdamsPrimer>::Primer, &'static str>;

    fn len(&self) -> usize;
}

pub trait AdamsInit {
    fn init_dtks(&mut self, t_kp1: f64);
    fn init_psis_n(&mut self);
    fn init_psis_np1(&mut self);
    fn init_betas(&mut self);
    fn init_alphas(&mut self, t_nxt: f64);
    fn init_phis(&mut self);
    fn init_adams(&mut self, t_nxt: f64);
}

pub trait AdamsPredictor: AdamsInit {
    type OutVec;
    fn gen_g(i: usize, j: usize, alphas: &VecDeque<f64>) -> f64 {
        match i {
            0 => 1.0 / (j as f64),
            1 => 1.0 / (j as f64 * (j as f64 + 1.0)),
            i => Self::gen_g(i - 1, j, alphas) - alphas[i - 1] * Self::gen_g(i - 1, j + 1, alphas),
        }
    }

    fn get_g1s(&mut self);
    fn get_phi_star(&mut self);
    fn get_phi_e(&mut self);
    fn predict(&self, m: usize, step: f64) -> <Self as AdamsPredictor>::OutVec;
}

pub trait AdamsCorrector {
    type OutVec;

    fn correct(
        &self,
        step: f64,
        t_nxt: f64,
        fxn: fn(f64, &<Self as AdamsCorrector>::OutVec) -> <Self as AdamsCorrector>::OutVec,
        prediction: &<Self as AdamsCorrector>::OutVec,
    ) -> <Self as AdamsCorrector>::OutVec;
}

pub trait AdamsUpdate: AdamsPredictor + AdamsInit {
    fn update_phis(&mut self);
}

pub trait AdamsStepper: AdamsUpdate + AdamsPredictor + AdamsCorrector + AdamsInit {
    type OutVec;

    fn step(
        &mut self,
        order: usize,
        fxn: fn(f64, &<Self as AdamsStepper>::OutVec) -> <Self as AdamsStepper>::OutVec,
        t_nxt: f64,
    ) -> Result<<Self as AdamsStepper>::OutVec, &'static str>;
}
