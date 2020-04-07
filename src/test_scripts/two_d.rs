/// Test script for two dimensional test problem
///

#[cfg(test)]
mod tests {
    // === Begin Imports ===
    // third party imports
    extern crate nalgebra as na;
    use na::Vector2;
    // std imports
    use crate::ridc::base::RIDCIntegratorAdaptive;
    use crate::ridc::base::RIDCIntegratorFixed;
    use crate::ridc::common::IntegOptionsParallel;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, DOPRI78, RK32};
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::two_d::*;
    use std::time::Instant;
    // === End Imports ===

    // Generates data for plotting the Acccuracy vs CPU time for adaptive steppers
    #[test]
    #[ignore]
    fn test_time_to_acc_2d_all() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];
        let time_end = 20.0;
        let dt = time_end - IT_2_D;

        let bogacki = RK32.clone();
        let cash_karp45 = CASH_KARP45.clone();
        let dopri78 = DOPRI78.clone();

        // test bogacki
        for integ in vec!["bogacki32", "cash_karp45"] {
            for acc in &accs {
                let reg_options = IntegOptions {
                    atol: Some(Vector2::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                };
                let start = Instant::now();
                let (dur, diff) = match integ {
                    "bogacki32" => {
                        let ans_reg = bogacki
                            .integrate(
                                two_d_dynamics,
                                IT_2_D,
                                IV_2_D.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (two_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "cash_karp45" => {
                        let ans_reg = cash_karp45
                            .integrate(
                                two_d_dynamics,
                                IT_2_D,
                                IV_2_D.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (two_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "dopri78" => {
                        // takes too long otherwise. Goes rapidly to machine precision
                        if acc >= &1e-9_f64 {
                            let reg_options = IntegOptions {
                                atol: Some(Vector2::repeat(*acc * 10e2_f64)),
                                rtol: Some(*acc * 10e5_f64),
                                min_step: Some(1e-10_f64),
                            };
                            let ans_reg = dopri78
                                .integrate(
                                    two_d_dynamics,
                                    IT_2_D,
                                    IV_2_D.clone(),
                                    dt,
                                    reg_options.clone(),
                                )
                                .unwrap();
                            let dur = start.elapsed().as_millis();
                            let diff = (two_d_solution(time_end) - ans_reg.last_y()).norm();
                            (dur, diff)
                        } else {
                            (0, -1000.0)
                        }
                    }
                    _ => (0, -1000.0),
                };
                println!("[2d adaptive], {:?}, {:?}, {:?}", integ, dur, diff);
            }
        }
        /*
        // Test RIDC Methods
        for n in 1..=4 {
            for acc in &accs {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector2::repeat(*acc * 10e4_f64)),
                    rtol: Some(*acc * 10e7_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-6_f64),
                };

                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(two_d_dynamics, IT_2_D, &IV_2_D, dt, par_options.clone())
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (two_d_solution(time_end) - ans_par.last_y()).norm();
                println!("[2d adaptive], RIDC(3)({})[100], {:?}, {:?}", n, dur, diff);
            }
        }
        */
        for c in 1..=4 {
            for step in vec![1.0, 0.75, 0.5, 0.25, 0.1] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector2::repeat(1e-10)),
                    rtol: Some(1e-7),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(c),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-10_f64),
                };
                let start = Instant::now();
                let ans_par = RK4
                    .parallel_integrator(
                        two_d_dynamics,
                        IT_2_D,
                        &IV_2_D,
                        dt,
                        step,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (two_d_solution(time_end) - ans_par.last_y()).norm();
                println!("[2d fixed], RIDC_F(3)({})[100], {:?}, {:?}", c, dur, diff);
            }
        }
    }
}
