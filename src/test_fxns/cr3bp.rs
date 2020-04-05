///=== Circularly Restricted 3 body Problem ===
/// Circularly Restricted 3 Body problem
///
/// Dynamics function for Earth-Moon 3-Body system
/// i.e. Given current state X, fn(X) -> \dot{X}
/// Note: MU is hard coded here
/// Note: All state values are in non-dimensionalized units
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::Vector4;

// === End Imports ===

// Constants
pub const MU_CR3BP: f64 = 0.0121505; // km^3s

// Initial States
pub const IT_CR3BP: f64 = 0.0;
lazy_static! {
    pub static ref IV_CR3BP: Vector4<f64> =
        Vector4::new(0.8368516530226652, 0.0, 0.10189706350017445, 0.0);
}

// Dynamics Function
pub fn cr3bp_dyn(_time: f64, state: &Vector4<f64>) -> Vector4<f64> {
    let denom_1_v_partials = ((MU_CR3BP + state[0] - 1.0).powf(2.0) + state[1].powf(2.0))
        .sqrt()
        .powf(3.0);
    let denom_2_v_partials = ((MU_CR3BP + state[0]).powf(2.0) + state[1].powf(2.0))
        .sqrt()
        .powf(3.0);
    let v_partial_x = -(MU_CR3BP * (MU_CR3BP + state[0] - 1.0)) / denom_1_v_partials
        - (1.0 - MU_CR3BP) * (MU_CR3BP + state[0]) / denom_2_v_partials
        + state[0];
    let v_partial_y = -(MU_CR3BP * state[1]) / denom_1_v_partials
        - (1.0 - MU_CR3BP) * state[1] / denom_2_v_partials
        + state[1];

    Vector4::new(
        state[2],
        state[3],
        2.0 * state[3] + v_partial_x,
        -2.0 * state[2] + v_partial_y,
    )
}

#[cfg(test)]
mod tests {
    use super::*;

    // std imports
    use std::f64::consts::PI;

    // Local imports
    use crate::runge_kutta::adaptive::AdaptiveStep;
    use crate::runge_kutta::common::IntegOptions;
    use crate::runge_kutta::rk_embed::RK32;
    use crate::test_fxns::cr3bp::*;

    #[test]
    #[ignore]
    fn is_this_thing_on() {
        let time_end = 6.0 * PI;
        let dt = time_end - IT_CR3BP;

        // compute "truth" trajectory
        let reg_options = IntegOptions {
            atol: Some(Vector4::repeat(1e-9_f64)),
            rtol: Some(1e-6_f64),
            min_step: Some(1e-10_f64),
        };
        let truth_res = RK32
            .clone()
            .integrate(
                cr3bp_dyn,
                IT_CR3BP,
                IV_CR3BP.clone(),
                dt,
                reg_options.clone(),
            )
            .unwrap();
    }
}
