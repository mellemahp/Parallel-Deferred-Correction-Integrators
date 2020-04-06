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
    use crate::runge_kutta::fixed::FixedStep;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, DOPRI78, RK32, RKF45};
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::kepler::*;

    // === End Imports ===
    // Generates data for plotting the Acccuracy vs CPU time

    #[test]
    fn test_time_to_acc_kep_pert_0001() {
        // INitial state
        let kep_init = ISTATE_GTO.clone();
        let cart_init = kep_init.into_cartesian();
        let time_end = 2000.0;
        let dt = time_end;

        // pre compute "truth"
        /*
        let reg_options = IntegOptions {
            atol: Some(Vector6::repeat(1e-8)),
            rtol: Some(1e-5),
            min_step: Some(1e-12_f64),
        };
        let cart_truth = DOPRI78
            .clone()
            .integrate(
                full_perturbed_2body_dyn,
                0.0,
                cart_init.clone(),
                dt,
                reg_options.clone(),
            )
            .unwrap();
        println!("{:?}", cart_truth.last_y());
        */

        let cart_truth = Vector6::new(
            8068.09618087574,
            10436.052401720519,
            2259.5590100814093,
            -1.416870031479932,
            6.383472235632609,
            0.5235383661448467,
        );

        // Initialize integrators
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64, 1e-13_f64,
            1e-14_f64, 1e-15_f64, 1e-16_f64,
        ];

        // test reg runge kutta integrators
        for integ in vec!["bogacki32", "cash_karp45"] {
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
                                full_perturbed_2body_dyn,
                                0.0,
                                cart_init.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = &cart_truth - ans_reg.last_y();
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
                                full_perturbed_2body_dyn,
                                0.0,
                                cart_init.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = &cart_truth - ans_reg.last_y();
                        let diff_pos =
                            (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                        let diff_vel =
                            (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();
                        (dur, diff_pos, diff_vel)
                    }
                    _ => (0, 0.0, 0.0),
                };

                println!("{:?}, {:?}, {:?}, {:?}", integ, dur, diff_pos, diff_vel);
            }
        }

        for c in vec![4] {
            for step in vec![100.0, 50.0, 10.0, 5.0, 4.0, 1.0] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(1e-9)),
                    rtol: Some(1e-6),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(c),
                    restart_length: Some(50),
                    convergence_tol: Some(1e-8_f64),
                };
                let start = Instant::now();
                let ans_par = RK4
                    .parallel_integrator(
                        full_perturbed_2body_dyn,
                        0.0,
                        &cart_init.clone(),
                        dt,
                        step,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = &cart_truth - ans_par.last_y();
                let diff_pos = (diff[0].powf(2.0) + diff[1].powf(2.0) + diff[2].powf(2.0)).sqrt();
                let diff_vel = (diff[1].powf(2.0) + diff[2].powf(2.0) + diff[3].powf(2.0)).sqrt();

                println!(
                    "RIDC FIXED {}, {:?}, {:?}, {:?}",
                    c, dur, diff_pos, diff_vel
                );
            }
        }

        println!("STARTING DAT SEXY PARALLEL INTEGRATION");
        // test adaptive RIDC
        for n in vec![4] {
            for acc in vec![1e-6_f64, 1e-7_f64, 1e-8_f64] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(acc)),
                    rtol: Some(acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(50),
                    convergence_tol: Some(1e-8),
                };
                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(
                        full_perturbed_2body_dyn,
                        0.0,
                        &cart_init.clone(),
                        dt,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = &cart_truth - ans_par.last_y();
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
