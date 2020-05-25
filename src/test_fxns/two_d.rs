///=== 2-D Test Problem ===
/// We pull this example from example 4 of:
/// https://resources.saylor.org/wwwresources/archived/site/wp-content/uploads/2012/09/MA102-5.5.4-Equations-and-Initial-Value-Problems.pdf
///
/// Once again the answer is also validated with python solve IVP
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::Vector2;

// standard library
use std::thread::sleep;
use std::time::Duration;

// === End Imports ===

// Initial Conditions
pub const IT_2_D: f64 = 0.0;
lazy_static! {
    pub static ref IV_2_D: Vector2<f64> = Vector2::new(1.0, 4.0);
}

// Dynamics
pub fn two_d_dynamics(t: f64, y: &Vector2<f64>) -> Vector2<f64> {
    // Added For "WEIGHT"
    sleep(Duration::from_micros(10));
    // END WEIGHTING
    Vector2::new(y[1], 2.0 - 6.0 * t)
}

// Analytic Solution
pub fn two_d_solution(t: f64) -> Vector2<f64> {
    let y = t.powf(2.0) - t.powf(3.0) + 4.0 * t + 1.0;
    let dy = 2.0 * t - 3.0 * t.powf(2.0) + 4.0;
    Vector2::new(y, dy)
}
