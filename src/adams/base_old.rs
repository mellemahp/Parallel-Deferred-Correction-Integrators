extern crate nalgebra as na;
use na::allocator::Allocator;
use na::{DefaultAllocator, Dim, DimName, VectorN};

use crate::runge_kutta::adaptive::{AdaptiveStep, StepValid};
use crate::runge_kutta::common::{StepResult, StepWithError, Tolerances};
use crate::runge_kutta::rk_embed::RK32;
use crate::runge_kutta::common::{IntegOptions}

// Divided diff from:
// https://www.math.usm.edu/lambers/mat460/fall09/lecture17.pdf
pub fn divided_diff<N: Dim + DimName>(
    points: &[VectorN<f64, N>],
    times: &[f64],
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let n = times.len();
    // initialize empty finite difference table
    let mut diffs = vec![vec![VectorN::<f64, N>::zeros(); n]; n];
    let mut times = times.clone().to_vec();
    times.reverse();
    for i in (0..n).rev() {
        diffs[i][0] = points[i].clone();
    }
    for j in 1..n {
        for i in 0..n - j {
            diffs[i][j] = (&diffs[i + 1][j - 1] - &diffs[i][j - 1]) / (times[i + j] - times[i])
        }
    }
    diffs[0].clone()
}

// Finds the divided difference Coefficients using the following
// Phi equation is phi_{k,i} = \mult_{j=0}^{i-1} \psi_{k-1}T[t_k,...,t_{k-i}]
// where T[t_k,...,t_{k-i}] is the newton divided difference
// NOTE: T[t_k,...,t_{k-i}] is reverse the intuitive order!
//

// N is the size of the state vector. K is the order of the Predictor.
// In other words, K is the order of the polynommial fit
pub struct AdamsPredictor<N: Dim + DimName, K: Dim + DimName>
where
    DefaultAllocator: Allocator<f64, N> + Allocator<f64, K>,
{
    pub xs: Vec<VectorN<f64, N>>,
    pub hs: VectorN<f64, K>,
    pub psis: Vec<f64>,
}

impl<N: Dim + DimName, K: Dim + DimName> AdamsPredictor<N, K>
where
    DefaultAllocator: Allocator<f64, K> + Allocator<f64, N>,
{
    // t_kp1 is the first time step
}

// i runs from i = 0, 1, ..., k
//
pub fn get_dtks(times: &[f64], t_kp1: f64) -> Vec<f64> {
    let mut dtk: Vec<f64> = Vec::new();
    for idx in 0..times.len() - 1 {
        dtk.push(times[idx + 1] - times[idx]);
    }
    dtk.push(t_kp1 - times[times.len() - 1]);
    dtk
}

pub fn get_betas(poly_order: usize, psis_n: &Vec<f64>, psis_np1: &Vec<f64>) -> Vec<f64> {
    let mut betas: Vec<f64> = Vec::with_capacity(poly_order + 1);
    betas.push(1.0);
    let mut beta_prod: f64;
    for i in 1..poly_order + 1 {
        beta_prod = psis_np1[0] / psis_n[0];
        for j in 1..i {
            beta_prod *= psis_np1[j] / psis_n[j];
        }
        betas.push(beta_prod);
    }
    betas
}

pub fn get_psis_n(poly_order: usize, dtks: &Vec<f64>) -> Vec<f64> {
    let mut psis: Vec<f64> = Vec::new();
    let mut psi_sum: f64 = 0.0;

    for i in 1..poly_order + 1 {
        psi_sum += dtks[dtks.len() - 1 - i];
        psis.push(psi_sum);
    }
    psis
}

pub fn get_psis_np1(poly_order: usize, dtks: &Vec<f64>) -> Vec<f64> {
    let mut psis: Vec<f64> = Vec::new();
    let mut psi_sum: f64 = 0.0;

    for i in 1..poly_order + 1 {
        psi_sum += dtks[dtks.len() - i];
        psis.push(psi_sum);
    }
    psis
}

pub fn get_g1s(m: usize, alphas: &Vec<f64>) -> Vec<f64> {
    let mut gs: Vec<f64> = Vec::new();
    for i in 0..m + 1 {
        gs.push(gen_g(i, 1, alphas))
    }
    gs
}

pub fn gen_g(i: usize, j: usize, alphas: &Vec<f64>) -> f64 {
    match i {
        0 => 1.0 / (j as f64),
        1 => 1.0 / (j as f64 * (j as f64 + 1.0)),
        i => gen_g(i - 1, j, alphas) - alphas[i - 1] * gen_g(i - 1, j + 1, alphas),
    }
}

pub fn get_alphas(psis_np1: &Vec<f64>, step: f64) -> Vec<f64> {
    let mut alphas: Vec<f64> = Vec::new();
    for val in psis_np1.iter() {
        alphas.push(step / val);
    }
    alphas
}

pub fn get_phis<N: Dim + DimName>(
    fn_eval_n: &VectorN<f64, N>,
    divided_diff_vec: &Vec<VectorN<f64, N>>,
    psis_last: &Vec<f64>,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut phis: Vec<VectorN<f64, N>> = Vec::new();
    let mut mult_res = VectorN::<f64, N>::repeat(1.0);
    phis.push(fn_eval_n.clone());
    for (idx, psi) in psis_last.iter().enumerate() {
        mult_res *= *psi;
        phis.push(mult_res.component_mul(&divided_diff_vec[idx]));
    }
    phis
}

pub fn get_phi_star<N: Dim + DimName>(
    m: usize,
    betas: &Vec<f64>,
    phis: &Vec<VectorN<f64, N>>,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut phi_stars: Vec<VectorN<f64, N>> = Vec::new();
    for idx in 0..m + 1 {
        phi_stars.push(betas[idx] * phis[idx].clone())
    }
    phi_stars
}

pub fn get_prediction<N: Dim + DimName>(
    m: usize,
    phi_star: &Vec<VectorN<f64, N>>,
    gs: &Vec<f64>,
    step: f64,
    y_n: &VectorN<f64, N>,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut sum = VectorN::<f64, N>::zeros();
    // TODO: Validate that it should be M+1, not M
    for i in 0..m {
        sum += gs[i] * phi_star[i].clone()
    }
    y_n + step * sum
}

pub fn get_phi_e<N: Dim + DimName>(
    m: usize,
    phi_star: &Vec<VectorN<f64, N>>,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut phi_e: Vec<VectorN<f64, N>> = Vec::new();
    phi_e.push(VectorN::<f64, N>::zeros());
    // TODO: Validate whether this should be M+1 or M
    for idx in 1..m {
        phi_e.push(phi_e[idx - 1].clone() + phi_star[phi_star.len() - idx].clone());
    }
    phi_e
}

fn update_phis<N: Dim + DimName>(
    m: usize,
    corr_fxn_eval: VectorN<f64, N>,
    phi_e: Vec<VectorN<f64, N>>,
) -> Vec<VectorN<f64, N>>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut phis_new: Vec<VectorN<f64, N>> = Vec::new();
    // start with k+1
    phis_new.push(corr_fxn_eval.clone() - phi_e.last().unwrap().clone());
    for idx in 1..m {
        phis_new.push(phi_e[m - idx].clone() + phis_new[0].clone());
    }
    phis_new
}

pub fn update_psis(psis: &mut Vec<f64>, step: f64) -> Vec<f64> {
    psis.iter_mut().map(|psis| *psis + step).collect()
}

#[cfg(test)]
mod tests {
    use super::*;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::RK32;
    use crate::test_fxns::{two_d_dynamics, two_d_solution, IT_2_D, IV_2_D};
    use na::{Vector1, Vector2, Vector3, Vector4};

    //#[test]
    fn test_divided_difference_1() {
        // Case testing against geeksforgeeks.com
        // NOTE: because of the way the divided diff works for
        // the adams predictor this vector is flipped from the
        // one in the geeks for geeks site
        let pts = vec![
            Vector1::new(16.0),
            Vector1::new(14.0),
            Vector1::new(13.0),
            Vector1::new(12.0),
        ];
        let times = vec![5.0, 6.0, 9.0, 11.0];

        let ans = divided_diff(&pts, &times);

        let diff_true = vec![
            Vector1::new(12.0),
            Vector1::new(1.0),
            Vector1::new(-1.0 / 6.0),
            Vector1::new(1.0 / 20.0),
        ];

        const TOL: f64 = 1.0e-10;
        for idx in 0..diff_true.len() {
            assert!((ans[idx] - diff_true[idx]) < Vector1::new(TOL));
        }
    }

    #[test]
    fn test_g_gen() {
        // Running from i = 0 -> k where k = 3
        // Answers here checked by hand, 2 seperate times with same result
        // also checked answer using a small python script
        let alphas = vec![1.0, 1.0, 1.0, 1.0];
        let gs = get_g1s(3, &alphas);
        let hand_calc = vec![
            1.0,
            1.0 / 2.0,
            1.0 / 3.0,
            1.0 / 3.0 - 1.0 / 6.0 + 1.0 / 12.0,
        ];
        const TOL: f64 = 1.0e-12_f64;
        assert_eq!(gs.len(), 4);
        for i in 0..4 {
            assert!((gs[i] - hand_calc[i]).abs() < TOL);
        }
    }

    #[test]
    fn test_dtks() {
        let times = vec![0.0, 1.0, 2.0, 4.0];
        let t_nxt = 7.0;
        let dtks = get_dtks(&times, t_nxt);
        let sol = vec![1.0, 1.0, 2.0, 3.0];
        assert_eq!(dtks.len(), 4);
        const TOL: f64 = 1.0e-12_f64;
        for i in 0..4 {
            assert!((dtks[i] - sol[i]).abs() < TOL);
        }
    }

    #[test]
    fn test_psis_n() {
        // from previous test for `get_dtks`
        let order: usize = 3;
        let dtks = vec![1.0, 1.0, 2.0, 3.0];

        let psis_n = get_psis_n(order, &dtks);
        assert_eq!(psis_n, vec![2.0, 3.0, 4.0]);
    }

    #[test]
    fn test_psis_np1() {
        let order: usize = 3;
        let dtks = vec![1.0, 1.0, 2.0, 3.0];

        let psis_np1 = get_psis_np1(order, &dtks);
        assert_eq!(psis_np1, vec![3.0, 5.0, 6.0]);
    }

    #[test]
    fn test_betas() {
        let order: usize = 3;
        let phis_n = vec![1.0, 1.0, 1.0];
        let phis_np1 = vec![2.0, 2.0, 2.0];
        let betas = get_betas(order, &phis_n, &phis_np1);
        assert_eq!(betas.len(), 4);
        let sol = vec![1.0, 2.0, 4.0, 8.0];
        assert_eq!(betas, sol);
    }

    #[test]
    fn test_alphas() {
        let step = 4.0;
        let phis_np1 = vec![1.0, 2.0, 3.0, 4.0];
        let alphas = get_alphas(&phis_np1, step);
        let sol = vec![4.0, 2.0, 4.0 / 3.0, 1.0];
        assert_eq!(alphas, sol);
    }

    #[test]
    fn test_phi_e() {
        let order: usize = 3;
        let phi_star = vec![
            Vector1::new(1.0),
            Vector1::new(1.0),
            Vector1::new(1.0),
            Vector1::new(1.0),
        ];
        let phi_e = get_phi_e(order, &phi_star);
        assert_eq!(
            phi_e,
            vec![Vector1::new(0.0), Vector1::new(1.0), Vector1::new(2.0)]
        );
    }

    #[test]
    fn test_phi_star() {
        let order: usize = 3;
        //let phis = vec![];
        //let betas = vec![];
        //let phi_star = get_phi_star(order, betas, phis);
        //println!("PHI_STAR | {:?}", phi_star);
    }

    #[test]
    fn test_prediction() {
        let order: usize = 3;
        let phi_star = vec![Vector1::new(1.0), Vector1::new(1.0), Vector1::new(1.0)];
        let gs = vec![1.0, 1.0, 1.0];
        let step = 2.0;
        let y_n = Vector1::new(1.0);

        let pred = get_prediction(order, &phi_star, &gs, step, &y_n);
        assert_eq!(pred, Vector1::new(7.0));
    }

    #[test]
    fn test_gen() {
        let m: usize = 7;
        let abs_tol = Vector2::new(1.0e-12, 1.0e-12);
        let rel_tol = 1.0e-9_f64;
        let step_0 = 0.4;

        let init_vals = initialize_multistep(
            m + 1,
            &abs_tol,
            rel_tol,
            two_d_dynamics,
            step_0,
            IT_2_D,
            &IV_2_D,
        )
        .expect("Could not Generate Initialization values");
        println!("MULTISTEP_INIT | {:?}", init_vals);

        let step = (abs_tol.norm() + IV_2_D.norm() * rel_tol.sqrt()) / 2.0;

        let (pred_fxn, _, _) = predictor(m, step, &init_vals).expect("Could not run Predictor");
        println!("PRED FXN | {:?}", pred_fxn);

        let (pred_fxn_2, _, _) =
            predictor(m + 1, step, &init_vals).expect("Could not run Predictor");
        println!("PRED FXN 2 | {:?}", pred_fxn_2);

        // ERROR BASED ON: https://arxiv.org/pdf/1104.3187.pdf
        let error_vec = Vector2::from_iterator(
            (&pred_fxn - &pred_fxn_2)
                .abs()
                .iter()
                .enumerate()
                .map(|(idx, delta)| delta / (abs_tol[idx] + rel_tol * pred_fxn[idx])),
        );
        let error = error_vec[error_vec.imax()];
        println!("ERROR EST | {:?}", error);

        let step_change = (1.0 / error).powf(1.0 / (1.0 + m as f64));
        println!("ERROR SHIFT | {:?}", step_change);

        let gamma = 1.0; // check_gamma(m, &abs_tol, rel_tol, &pred_fxn, &pred_fxn_2);
        println!("GAMMA | {:?}", gamma);

        let step_2 = step * gamma;
        let t_last = init_vals.times.last().unwrap();
        let (pred, phi_e, gs) = predictor(m, step_2, &init_vals).expect("Could not run Predictor");
        let corr_fxn = corrector(&pred, two_d_dynamics, t_last + step_2, step_2, &gs, &phi_e);
        println!("CORR FXN | {:?}", corr_fxn);

        let truth = two_d_solution(t_last + step_2);
        println!("TRUTH | {:?}", truth);

        // Update differences
        /*
        let updated_phis = update_phis(m, corr_fxn_eval, phi_e);
        println!("UPDATED_PHIS | {:?}", updated_phis);

        //=================================================
        // Try a second step! =============================
        //=================================================

        let t_nxt_2 = 0.5;
        let step_2 = t_nxt_2 - t_nxt;

        // Update psis
        let psis_n = update_psis(&mut psis_n, step_2);
        let psis_np1 = update_psis(&mut psis_np1, step_2);
        println!("Next PSIS | {:?} {:?}", psis_n, psis_np1);

        // Update Alphas
        let alphas = get_alphas(&psis_np1, step_2);

        // Update Betas
        let betas = get_betas(m, &psis_n, &psis_np1);

        // Get new gs
        let gs = get_g1s(m, &alphas);

        // Rerun it all!
        let phi_star = get_phi_star(m, &betas, &phis);
        pts.remove(0);
        pts.push(corrected);
        let pred = get_prediction(m, &phi_star, &gs, step_2, &pts[pts.len() - 1]);
        println!("PRED | {:?}", pred);

        let phi_e = get_phi_e(m, &phi_star);
        println!("PHI_E | {:?}", phi_e);

        // Evaluate fxn again now, with prediction
        let new_eval = two_d_dynamics(t_nxt, &pred);

        let corrected = pred + step_2 * gs.last().unwrap() * (new_eval - phi_e.last().unwrap());
        println!("CORRECTED VAL | {:?}", corrected);

        let truth = two_d_solution(t_nxt_2);
        println!("TRUTH | {:?}", truth);

        // re-evaluate dynamics with correction
        let corr_fxn_eval = two_d_dynamics(t_nxt_2, &corrected);
        println!("CORRECTED DT | {:?}", corr_fxn_eval);
        */
    }
}

pub fn predictor<N: Dim + DimName>(
    order: usize,
    step: f64,
    data: &MultiStepData<N>,
) -> Result<(VectorN<f64, N>, Vec<VectorN<f64, N>>, Vec<f64>), String>
where
    DefaultAllocator: Allocator<f64, N>,
{
    if data.times.len() < order + 1 {
        return Err(format!(
            "Polynomial of order {} cannot be fit to data of len {}",
            data.times.len(),
            order
        ));
    }
    // Extract points and times from the data
    if data.times.len() != data.states.len() || data.times.len() != data.dynamics.len() {
        return Err(format!(
            "Length of Times {}, States {}, And Dynamics {} must be equal",
            data.times.len(),
            data.states.len(),
            data.dynamics.len()
        ));
    }
    let k = data.times.len();
    let times = &data.times[k - (order + 1)..k];
    let pts = &data.dynamics[k - (order + 1)..k];
    let t_nxt = times.last().unwrap() + step;

    // Find all recursive relationship data
    let dtks = get_dtks(times, t_nxt);
    let psis_n = get_psis_n(order, &dtks);
    let psis_np1 = get_psis_np1(order, &dtks);
    let betas = get_betas(order, &psis_n, &psis_np1);
    let alphas = get_alphas(&psis_np1, step);
    let gs = get_g1s(order, &alphas);
    let div_diff = divided_diff(pts, times);
    let phis = get_phis(pts.last().unwrap(), &div_diff, &psis_n);
    let phi_star = get_phi_star(order, &betas, &phis);
    let phi_e = get_phi_e(order, &phi_star);

    Ok((
        get_prediction(order, &phi_star, &gs, step, &data.states.last().unwrap()),
        phi_e,
        gs,
    ))
}

pub fn corrector<N: Dim + DimName>(
    prediction: &VectorN<f64, N>,
    fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    t_nxt: f64,
    step: f64,
    gs: &Vec<f64>,
    phi_e: &Vec<VectorN<f64, N>>,
) -> VectorN<f64, N>
where
    DefaultAllocator: Allocator<f64, N>,
{
    prediction + step * gs.last().unwrap() * (fxn(t_nxt, &prediction) - phi_e.last().unwrap())
}

// Checks the gamma for a given prediction
pub fn check_gamma<N: Dim + DimName>(
    order: usize,
    abs_tol: &VectorN<f64, N>,
    rel_tol: f64,
    pred_lower: &VectorN<f64, N>,
    pred_higher: &VectorN<f64, N>,
) -> f64
where
    DefaultAllocator: Allocator<f64, N>,
{
    let numer = abs_tol + (rel_tol * pred_higher).abs();
    let denom = 2.0 * (pred_higher - pred_lower).abs();
    let gamma = numer
        .component_div(&denom)
        .map(|val| val.powf(1.0 / (order as f64 + 2.0)));
    gamma[gamma.imin()]
}

fn initialize_multistep<N: Dim + DimName>(
    order: usize,
    atol: &VectorN<f64, N>,
    rtol: f64,
    fxn: fn(f64, &VectorN<f64, N>) -> VectorN<f64, N>,
    step: f64,
    t_0: f64,
    y_0: &VectorN<f64, N>,
) -> Result<MultiStepData<N>, &'static str>
where
    DefaultAllocator: Allocator<f64, N>,
{
    let mut fxn_evals: Vec<VectorN<f64, N>> = Vec::new();
    let mut times: Vec<f64> = vec![t_0];
    let mut states: Vec<VectorN<f64, N>> = Vec::new();
    states.push(y_0.clone());

    let mut t_last = t_0;
    let mut y_last = y_0.clone();
    let mut sub_step = step / (2.0 * (order + 1) as f64);

    // Take enough Steps to initialize an Mth order adams method
    let mut step_res: StepResult<N>;
    let mut step_revision: StepValid;
    while times.len() < order + 1 {
        if t_last >= t_0 + step {
            return Err("Initializer over-stepped end of interval");
        }
        step_res = RK32.step(fxn, t_last, &y_last, sub_step, atol, rtol);
        step_revision = RK32.revise_step(step_res.error, sub_step);

        match step_revision {
            StepValid::Accept(nxt_step) => {
                times.push(t_last + sub_step);
                fxn_evals.push(fxn(t_last, &y_last));
                t_last += sub_step;
                y_last = step_res.value.clone();
                states.push(step_res.value);
                sub_step = nxt_step;
            }
            StepValid::Refine(nxt_step) => {
                sub_step = nxt_step;
            }
        }
    }
    fxn_evals.push(fxn(t_last, &y_last));

    Ok(MultiStepData {
        times,
        states,
        dynamics: fxn_evals,
    })
}