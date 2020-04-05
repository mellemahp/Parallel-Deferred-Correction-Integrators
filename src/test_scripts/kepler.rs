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
    use crate::ridc::common::IntegOptionsParallel;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_simp::RK4;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, RK32, RKF45};
    use crate::test_fxns::kepler::*;
    use crate::utils::finite_diff::fdiff_jacobian;
    use crate::ridc::base::RIDCIntegratorFixed;
    // === End Imports ===
    // Generates data for plotting the Acccuracy vs CPU time
    #[test]
    #[ignore]
    fn test_jacobian() {
        let kep_init = ISTATE_LEO.clone();
        let cart_init = kep_init.into_cartesian();
        let fxn = |x: &Vector6<f64>| two_body_dyn(0.0, x);
        let jac = fdiff_jacobian(&fxn, &Vector6::repeat(0.0), &cart_init);
        println!("{:?}", jac[3]);
    }

    #[test]
    fn test_time_to_acc_kep_0001() {
        // INitial state
        let kep_init = ISTATE_LEO.clone();
        let cart_init = kep_init.into_cartesian();
        println!("{:?}", cart_init);

        // Initialize integrators
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];

        let time_end = 3500.0;
        let dt = time_end;

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
                    _ => (0, 0.0, 0.0),
                };

                println!("{:?}, {:?}, {:?}, {:?}", integ, dur, diff_pos, diff_vel);
            }
        }
        /*
        for c in 1..=4 {
            for step in vec![100.0, 10.0, 5.0, 1.0] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(1e-9)),
                    rtol: Some(1e-6),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(c),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-5_f64 * step),
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
                    "[RIDC FIXED] {}, {:?}, {:?}, {:?}",
                    c, dur, diff_pos, diff_vel
                );
            }
        }
        */

        println!("STARTING DAT SEXY PARALLEL INTEGRATION");
        // test adaptive RIDC
        for n in 1..=4 {
            for acc in &accs {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector6::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(200),
                    convergence_tol: Some(1e-3),
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
                    "[PARALLEL 2D] RIDC ADAPTIVE {}, {:?}, {:?}, {:?}",
                    n, dur, diff_pos, diff_vel
                );
            }
        }
    }
}
