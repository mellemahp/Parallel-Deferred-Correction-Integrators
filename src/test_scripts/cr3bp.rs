/// Test script for Circularly restricted Three body problem
///

#[cfg(test)]
mod tests {
    // === Begin Imports ===
    // third party imports
    extern crate nalgebra as na;
    use na::Vector4;

    // std imports
    use std::f64::consts::PI;

    // Local imports
    use crate::ridc::base::RIDCIntegratorAdaptive;
    use crate::ridc::base::RIDCIntegratorFixed;
    use crate::ridc::common::IntegOptionsParallel;
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::fixed::FixedStep;
    use crate::runge_kutta::rk_embed::{CASH_KARP45, DOPRI78, RK32};
    use crate::runge_kutta::rk_simp::RK4;
    use crate::test_fxns::cr3bp::*;
    use std::time::Instant;
    // === End Imports ===

    // Generates data for plotting the Acccuracy vs CPU time for adaptive steppers
    #[test]
    #[ignore]
    fn test_time_to_acc_cr3bp_all() {
        let accs: Vec<f64> = vec![
            1e-6_f64, 1e-7_f64, 1e-8_f64, 1e-9_f64, 1e-10_f64, 1e-11_f64, 1e-12_f64,
        ];
        let time_end = 1.0 * PI;
        let dt = time_end - IT_CR3BP;

        let bogacki = RK32.clone();
        let cash_karp45 = CASH_KARP45.clone();
        let dopri78 = DOPRI78.clone();

        /*
        // computes "truth" trajectory
        let reg_options = IntegOptions {
            atol: Some(Vector4::repeat(1e-14_f64)),
            rtol: Some(1e-11_f64),
            min_step: Some(1e-10_f64),
        };
        //let step_true = 1e-_f64;
        let truth_res = CASH_KARP45
            .clone()
            .integrate(
                cr3bp_dyn,
                IT_CR3BP,
                IV_CR3BP.clone(),
                dt,
                reg_options.clone(),
            )
            .unwrap();
        let truth = truth_res.last_y();
        println!("STEPZ: {:?}", truth_res.times.len());
        println!("TRUUUUTH {:?}", truth);
        */
        // pre-computed using above code snippet
        let truth = Vector4::new(
            1.0805353652657592,
            -0.06287779092299657,
            -0.07154723508377878,
            0.10262047715819808,
        );

        // test all
        for integ in vec!["bogacki32", "cash_karp45"] {
            for acc in &accs {
                let reg_options = IntegOptions {
                    atol: Some(Vector4::repeat(*acc)),
                    rtol: Some(*acc * 10e3_f64),
                    min_step: Some(1e-12_f64),
                };
                let start = Instant::now();
                let (dur, diff) = match integ {
                    "bogacki32" => {
                        let ans_reg = bogacki
                            .integrate(
                                cr3bp_dyn,
                                IT_CR3BP,
                                IV_CR3BP.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (truth - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "cash_karp45" => {
                        let ans_reg = cash_karp45
                            .integrate(
                                cr3bp_dyn,
                                IT_CR3BP,
                                IV_CR3BP.clone(),
                                dt,
                                reg_options.clone(),
                            )
                            .unwrap();
                        let dur = start.elapsed().as_millis();
                        let diff = (truth - ans_reg.last_y()).norm();
                        (dur, diff)
                    }
                    "dopri78" => {
                        // takes too long otherwise. Goes rapidly to machine precision
                        if acc >= &1e-9_f64 {
                            let reg_options = IntegOptions {
                                atol: Some(Vector4::repeat(*acc * 10e2_f64)),
                                rtol: Some(*acc * 10e5_f64),
                                min_step: Some(1e-10_f64),
                            };
                            let ans_reg = dopri78
                                .integrate(
                                    cr3bp_dyn,
                                    IT_CR3BP,
                                    IV_CR3BP.clone(),
                                    dt,
                                    reg_options.clone(),
                                )
                                .unwrap();
                            let dur = start.elapsed().as_millis();
                            let diff = (truth - ans_reg.last_y()).norm();
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

        println!("STARTING PARALLEL");
        // Test RIDC Methods
        for n in 1..=4 {
            for acc in vec![1e-9_f64] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector4::repeat(acc)),
                    rtol: Some(acc * 10e3_f64),
                    min_step: Some(1e-12_f64),
                    poly_order: Some(3),
                    corrector_order: Some(n),
                    restart_length: Some(50),
                    convergence_tol: Some(1e-10_f64),
                };

                let start = Instant::now();
                let ans_par = RK32
                    .parallel_integrator(cr3bp_dyn, IT_CR3BP, &IV_CR3BP, dt, par_options.clone())
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (truth - ans_par.last_y()).norm();
                println!("[2d adaptive], RIDC(3)({})[100], {:?}, {:?}", n, dur, diff);
            }
        }
        for c in vec![3] {
            for step in vec![0.001, 0.0008, 0.0005] {
                let par_options = IntegOptionsParallel {
                    atol: Some(Vector4::repeat(1e-10)),
                    rtol: Some(1e-7),
                    min_step: Some(1e-10_f64),
                    poly_order: Some(3),
                    corrector_order: Some(c),
                    restart_length: Some(100),
                    convergence_tol: Some(1e-8_f64),
                };
                let start = Instant::now();
                let ans_par = RK4
                    .parallel_integrator(
                        cr3bp_dyn,
                        IT_CR3BP,
                        &IV_CR3BP,
                        dt,
                        step,
                        par_options.clone(),
                    )
                    .unwrap();
                let dur = start.elapsed().as_millis();
                let diff = (truth - ans_par.last_y()).norm();
                println!(
                    "[CR3BP fixed], RIDC_F(3)({})[100], {:?}, {:?}",
                    c, dur, diff
                );
            }
        }
    }
}
