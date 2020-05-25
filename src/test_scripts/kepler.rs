/// Test Script for Unperturbed Keplers Problem
///

#[cfg(test)]
mod tests {
    // === Begin Imports ===
    // third party imports
    extern crate nalgebra as na;
    use na::Vector6;
    // std imports
    use std::time::Instant;
    // local imports
    use crate::ridc::base::RIDCIntegratorAdaptive;
    use crate::ridc::base::RIDCIntegratorFixed;
    use crate::ridc::common::IntegOptionsParallel;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, DOPRI78, RK32, RKF45};
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::kepler::*;
    use crate::utils::finite_diff::fdiff_jacobian_2;
    // === End Imports ===
    // Generates data for plotting the Acccuracy vs CPU time
    #[test]
    #[ignore]
    fn test_jacobian() {
        let kep_init = ISTATE_LEO.clone().propagate_to_time(100.0);
        let cart_init = kep_init.into_cartesian();
        let fxn = |x: &Vector6<f64>| two_body_dyn(0.0, x);
        let jac = fdiff_jacobian_2(&fxn, &Vector6::repeat(0.0), &cart_init);
        println!("CART_INIT: {:?}", cart_init);
        println!(
            "JAC: {:?} {:?} {} {} {} {} \n {:?} {:?} {} {} {} {} \n {:?} {:?} {} {} {} {} \n {:?} {:?} {} {} {} {}",
            jac[(0, 0)],
            jac[(0, 1)],
            jac[(0, 2)],
            jac[(0, 3)],
            jac[(0, 4)],
            jac[(0, 5)],
            jac[(1, 0)],
            jac[(1, 1)],
            jac[(1, 2)],
            jac[(1, 3)],
            jac[(1, 4)],
            jac[(1, 5)],
            jac[(2, 0)],
            jac[(2, 1)],
            jac[(2, 2)],
            jac[(2, 3)],
            jac[(2, 4)],
            jac[(2, 5)],
            jac[(3, 0)],
            jac[(3, 1)],
            jac[(3, 2)],
            jac[(3, 3)],
            jac[(3, 4)],
            jac[(3, 5)]
        );
    }

    #[test]
    fn test_time_to_acc_kep_0001() {
        // INitial state
        let kep_init = ISTATE_GTO.clone();
        let cart_init = kep_init.into_cartesian();
        println!("{:?}", cart_init);

        // Initialize integrators
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64, 1e-13_f64,
            1e-14_f64, 1e-15_f64, 1e-16_f64, 1e-17_f64, 1e-18_f64,
        ];

        let time_end = 2000.0;
        let dt = time_end;

        // test reg runge kutta integrators
        for integ in vec!["bogacki32", "cash_karp45", "dopri78"] {
            for acc in &accs {
                let reg_options = IntegOptions {
                    atol: Some(Vector6::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                };
                let start = Instant::now();
                let (dur, diff_pos, diff_vel) = match integ {
                    "bogacki32" => {
                        let ans_reg = RK32
                            .clone()
                            .integrate(
                                two_body_dyn,
                                0.0,
                                cart_init.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = kep_init.propagate_to_time(time_end).into_cartesian()
                            - ans_reg.last_y();
                        let diff_pos =
                            (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                        let diff_vel =
                            (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();
                        (dur, diff_pos, diff_vel)
                    }
                    "cash_karp45" => {
                        let ans_reg = CASH_KARP45
                            .clone()
                            .integrate(
                                two_body_dyn,
                                0.0,
                                cart_init.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = kep_init.propagate_to_time(time_end).into_cartesian()
                            - ans_reg.last_y();
                        let diff_pos =
                            (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                        let diff_vel =
                            (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();
                        (dur, diff_pos, diff_vel)
                    }
                    "dopri78" => {
                        // takes too long otherwise. Goes rapidly to machine precision
                        if *acc >= 1e-8_f64 {
                            let reg_options = IntegOptions {
                                atol: Some(Vector6::repeat(*acc * 10e2_f64)),
                                rtol: Some(*acc * 10e5_f64),
                                min_step: Some(1e-10_f64),
                            };
                            let ans_reg = DOPRI78
                                .integrate(
                                    two_body_dyn,
                                    0.0,
                                    cart_init.clone(),
                                    dt,
                                    reg_options.clone(),
                                )
                                .unwrap();
                            let dur = start.elapsed().as_millis();
                            let diff = kep_init.propagate_to_time(time_end).into_cartesian()
                                - ans_reg.last_y();
                            let diff_pos =
                                (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                            let diff_vel =
                                (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();
                            (dur, diff_pos, diff_vel)
                        } else {
                            (0, 0.0, 0.0)
                        }
                    }
                    _ => (0, 0.0, 0.0),
                };

                println!("{:?}, {:?}, {:?}, {:?}", integ, dur, diff_pos, diff_vel);
            }
        }
        for c in vec![3, 4, 5] {
            for step in vec![100.0, 50.0, 10.0, 5.0, 4.0, 1.0, 0.1] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(1e-9)),
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
                        two_body_dyn,
                        0.0,
                        &cart_init.clone(),
                        dt,
                        step,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = kep_init.propagate_to_time(time_end).into_cartesian() - ans_par.last_y();
                let diff_pos = (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                let diff_vel = (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();

                println!(
                    "RIDC FIXED {}, {:?}, {:?}, {:?}",
                    c, dur, diff_pos, diff_vel
                );
            }
        }
        // test adaptive RIDC
        for n in vec![3, 4, 5] {
            for acc in vec![1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(acc)),
                    rtol: Some(acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-8),
                };
                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(
                        two_body_dyn,
                        0.0,
                        &cart_init.clone(),
                        dt,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = kep_init.propagate_to_time(time_end).into_cartesian() - ans_par.last_y();
                let diff_pos = (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                let diff_vel = (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();

                println!(
                    "RIDC ADAPTIVE {}, {:?}, {:?}, {:?}",
                    n, dur, diff_pos, diff_vel
                );
            }
        }
    }
}
