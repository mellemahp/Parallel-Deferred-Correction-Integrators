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
use na::Vector6;

// === End Imports ===

// Constants
pub const MU_CR3BP: f64 = 0.0121505; // km^3s

// Dynamics Function
pub fn cr3bp_dyn(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    // pre-computes inverse cubed of r1 and r2 to save computation time
    let r13_inv = 1.0
        / ((state[0] + MU_CR3BP).powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0))
            .sqrt()
            .powf(3.0);
    let r23_inv = 1.0
        / ((state[0] - (1.0 - MU_CR3BP).powf(2.0)) + state[1].powf(2.0) + state[2].powf(2.0))
            .sqrt()
            .powf(3.0);

    Vector6::new(
        state[3],
        state[4],
        state[5],
        state[0] + 2.0 * state[4]
            - (1.0 - MU_CR3BP) * (state[0] + MU_CR3BP) * &r13_inv
            - MU_CR3BP * (state[0] - (1.0 - MU_CR3BP)) * &r23_inv,
        state[1]
            - 2.0 * state[3]
            - (1.0 - MU_CR3BP) * state[1] * &r13_inv
            - MU_CR3BP * state[1] * &r23_inv,
        -(1.0 - MU_CR3BP) * state[2] * &r13_inv - MU_CR3BP * state[2] * &r23_inv,
    )
}
