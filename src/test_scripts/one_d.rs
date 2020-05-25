/// Test script for one dimensional test problem
///

#[cfg(test)]
mod tests {
    // === Begin Imports ===
    // third party imports
    extern crate nalgebra as na;
    use na::Vector1;
    // std imports
    use std::time::Instant;
    // local imports
    use crate::ridc::base::RIDCIntegratorAdaptive;
    use crate::ridc::base::RIDCIntegratorFixed;
    use crate::ridc::common::IntegOptionsParallel;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, DOPRI78, RK32};
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::one_d::*;
    // === End Imports ===

    // Generates data for plotting the Acccuracy vs CPU time for adaptive steppers
    #[test]
    #[ignore]
    fn test_time_to_acc_1d_all() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;

        let bogacki = RK32.clone();
        let cash_karp45 = CASH_KARP45.clone();
        let dopri78 = DOPRI78.clone();

        // test bogacki
        for integ in vec!["bogacki32", "cash_karp45"] {
            for acc in &accs {
                let reg_options = IntegOptions {
                    atol: Some(Vector1::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                };
                let start = Instant::now();
                let (dur, diff) = match integ {
                    "bogacki32" => {
                        let ans_reg = bogacki
                            .integrate(
                                one_d_dynamics,
                                ONE_D_INIT_TIME,
                                ONE_D_INIT_VAL.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "cash_karp45" => {
                        let ans_reg = cash_karp45
                            .integrate(
                                one_d_dynamics,
                                ONE_D_INIT_TIME,
                                ONE_D_INIT_VAL.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "dopri78" => {
                        // takes too long otherwise. Goes rapidly to machine precision
                        if acc >= &1e-9_f64 {
                            let reg_options = IntegOptions {
                                atol: Some(Vector1::repeat(*acc * 10e2_f64)),
                                rtol: Some(*acc * 10e5_f64),
                                min_step: Some(1e-10_f64),
                            };
                            let ans_reg = dopri78
                                .integrate(
                                    one_d_dynamics,
                                    ONE_D_INIT_TIME,
                                    ONE_D_INIT_VAL.clone(),
                                    dt,
                                    reg_options.clone(),
                                )
                                .unwrap();
                            let dur = start.elapsed().as_millis();
                            let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                            (dur, diff)
                        } else {
                            (0, -1000.0)
                        }
                    }
                    _ => (0, -1000.0),
                };
                println!("[1d adaptive], {:?}, {:?}, {:?}", integ, dur, diff);
            }
        }

        // Test RIDC Methods
        for n in vec![1, 2, 3, 4, 6, 8] {
            for acc in &accs {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector1::new(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(200),
                    convergence_tol: Some(1e-8_f64),
                };

                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(
                        one_d_dynamics,
                        ONE_D_INIT_TIME,
                        &ONE_D_INIT_VAL,
                        dt,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
                println!("[1d adaptive], RIDC(3)({})[100], {:?}, {:?}", n, dur, diff);
            }
        }

        for c in vec![1, 2, 3, 4, 6, 8] {
            for step in vec![0.1, 0.05, 0.025, 0.015, 0.005] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector1::new(1e-9)),
                    rtol: Some(1e-6),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(c),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-8_f64),
                };
                let start = Instant::now();
                let ans_par = RK4
                    .parallel_integrator(
                        one_d_dynamics,
                        ONE_D_INIT_TIME,
                        &ONE_D_INIT_VAL,
                        dt,
                        step,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
                println!("[1d fixed], RIDC_F(3)({})[100], {:?}, {:?}", c, dur, diff);
            }
        }
    }

    #[test]
    #[ignore]
    fn test_time_to_acc_1d_adaptive_ridc_restart_adaptive() {
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;

        // Test RIDC Methods
        for n in 1..=10 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector1::new(1e-10)),
                rtol: Some(1e-7),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(3),
                restart_length: Some(20 * n),
                convergence_tol: Some(1e-8_f64),
            };
            let start = Instant::now();
            let ans_par = RK32
                .parallel_integrator(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    &ONE_D_INIT_VAL,
                    dt,
                    par_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
            println!(
                "[1d adaptive], RIDC(3)(3)[{:?}], {:?}, {:?}",
                n * 20,
                dur,
                diff
            );
        }
    }

    #[test]
    #[ignore]
    fn test_time_to_acc_1d_adaptive_ridc_restart_fixed() {
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;
        let step = 0.005;
        // Test RIDC Methods
        for n in 1..=10 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector1::new(1e-10)),
                rtol: Some(1e-7),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(3),
                restart_length: Some(20 * n),
                convergence_tol: Some(1e-8_f64),
            };
            let start = Instant::now();
            let ans_par = RK4
                .parallel_integrator(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    &ONE_D_INIT_VAL,
                    dt,
                    step,
                    par_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
            println!(
                "[1d fixed], RIDC(3)(3)[{:?}], {:?}, {:?}",
                n * 20,
                dur,
                diff
            );
        }
    }

    #[test]
    fn test_time_to_acc_1d_adaptive_ridc_correction() {
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;

        // Test RIDC Methods
        for n in 1..=10 {
            let par_options = IntegOptionsParallel {
                atol: Some(Vector1::new(1e-10)),
                rtol: Some(1e-7),
                min_step: Some(1e-10_f64),
                poly_order: Some(3),
                corrector_order: Some(n),
                restart_length: Some(100),
                convergence_tol: Some(1e-8_f64),
            };
            let start = Instant::now();
            let ans_par = RK32
                .parallel_integrator(
                    one_d_dynamics,
                    ONE_D_INIT_TIME,
                    &ONE_D_INIT_VAL,
                    dt,
                    par_options.clone(),
                )
                .unwrap();
            let dur = start.elapsed().as_millis();
            let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
            println!("[1d adaptive], RIDC(3)({})[100], {:?}, {:?}", n, dur, diff);
        }
    }

    #[test]
    #[ignore]
    fn test_time_to_acc_1d_fixed() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];
        let time_end = 10.0;
        let dt = time_end - ONE_D_INIT_TIME;

        let bogacki = RK32.clone();
        let cash_karp45 = CASH_KARP45.clone();
        let dopri78 = DOPRI78.clone();

        // test bogacki
        for integ in vec!["bogacki32", "cash_karp45", "dopri78"] {
            for acc in &accs {
                let reg_options = IntegOptions {
                    atol: Some(Vector1::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                };
                let start = Instant::now();
                let (dur, diff) = match integ {
                    "bogacki32" => {
                        let ans_reg = bogacki
                            .integrate(
                                one_d_dynamics,
                                ONE_D_INIT_TIME,
                                ONE_D_INIT_VAL.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "cash_karp45" => {
                        let ans_reg = cash_karp45
                            .integrate(
                                one_d_dynamics,
                                ONE_D_INIT_TIME,
                                ONE_D_INIT_VAL.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "dopri78" => {
                        // takes too long otherwise. Goes rapidly to machine precision
                        if acc >= &1e-9_f64 {
                            let reg_options = IntegOptions {
                                atol: Some(Vector1::repeat(*acc * 10e2_f64)),
                                rtol: Some(*acc * 10e5_f64),
                                min_step: Some(1e-10_f64),
                            };
                            let ans_reg = dopri78
                                .integrate(
                                    one_d_dynamics,
                                    ONE_D_INIT_TIME,
                                    ONE_D_INIT_VAL.clone(),
                                    dt,
                                    reg_options.clone(),
                                )
                                .unwrap();
                            let dur = start.elapsed().as_millis();
                            let diff = (one_d_solution(time_end) - ans_reg.last_y()).norm();
                            (dur, diff)
                        } else {
                            (0, -1000.0)
                        }
                    }
                    _ => (0, -1000.0),
                };
                println!("[1d adaptive], {:?}, {:?}, {:?}", integ, dur, diff);
            }
        }

        // Test RIDC Methods
        for n in 0..4 {
            for acc in &accs {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector1::new(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-8_f64),
                };

                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(
                        one_d_dynamics,
                        ONE_D_INIT_TIME,
                        &ONE_D_INIT_VAL,
                        dt,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (one_d_solution(time_end) - ans_par.last_y()).norm();
                println!("[1d adaptive], RIDC(3)({})[100], {:?}, {:?}", n, dur, diff);
            }
        }
    }
}
