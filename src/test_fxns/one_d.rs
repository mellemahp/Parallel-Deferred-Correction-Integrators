///=== 1-D Test Problem ===
/// The one dimensional test problem
///
/// Taken from Example 6: Pauls Online notes differential eq.
/// http://tutorial.math.lamar.edu/Classes/DE/Definitions.aspx
/// Validated the example using python's solve_ivp()
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::Vector1;

// standard library
use std::thread::sleep;
use std::time::Duration;

// === End Imports ===

// Initial conditions
pub const ONE_D_INIT_TIME: f64 = 1.0;

lazy_static! {
    pub static ref ONE_D_INIT_VAL: Vector1<f64> = Vector1::new(-4.0);
}

// Dynamics
pub fn one_d_dynamics(t: f64, y: &Vector1<f64>) -> Vector1<f64> {
    // Added For "WEIGHT"
    sleep(Duration::from_micros(10));
    // END WEIGHTING
    (Vector1::new(3.0) - 4.0 * y) / (2.0 * t)
}

// Analytic Solution
pub fn one_d_solution(t: f64) -> Vector1<f64> {
    Vector1::new(3.0 / 4.0 - 19.0 / (4.0 * t.powf(2.0)))
}
