/// Computational Test Functions (comp_fxns)
///
/// Defines the test cases to use for evaluating integrator performance 
///
/// The following LEO, GTO, and HEO test cases are pulled from: 
/// Amato D. et Al "Non-averaged regularized formulations as an alternative to ..."
/// 
/// The Resonant 3-Body test case is taken from : 
/// ???
///
/// The total set of cases provided are:
///     - LEO 
///     - HEO
///     - GTO
///     - Resonant 3-Body orbit 
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::{Vector1, Vector2, Vector3, Vector6};

// standard library
use std::f64::consts::PI;

// local imports 
use super::test_fxns::KeplerianState;

// === End Imports ===

// Basic 2 body dynamics

pub const EARTH_MU: f64 = 398600.4418; // km^3s

/// Dynamics function for Earth 2-Body system
/// i.e. Given current state X, fn(X) -> \dot{X}
/// Note: All state values are in km or km/sec
pub fn two_body_dyn(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let r3_inv = 1.0
        / (state[0].powf(2.0) + state[0].powf(2.0) + state[0].powf(2.0))
            .sqrt()
            .powf(3.0);
    Vector6::new(
        state[3],
        state[4],
        state[5],
        -EARTH_MU * state[0] * &r3_inv,
        -EARTH_MU * state[1] * &r3_inv,
        -EARTH_MU * state[2] * &r3_inv,
    )
}

// LEO Test case
// C_D = 2.2, Mass of sc is 400kg, cross sectional area 0.7m^2
// 5x5 geopotential, lunisolar perturbations, and atmospheric drag
// Solar flux is held constant as is geomagntic planetary index and amplitude (K_p = 3.0, A_p, F_10.7 = 140 SFU)
// === Initial state ===
// MJD  58171.738177
// a    6862.14 km
// e    0.0
// i    97.46 deg
// raan 281.0 deg
// u    0.0 deg
pub const DEG_TO_RAD: f64 = PI / 180.0;
pub const IT_LEO: f64 = 58171.738177; // MJD
lazy_static! {
    // Note: this is in equinotial elements
    pub static ref ISTATE_LEO: Vector6<f64> = Vector6::new(
        6852.14,
        0.0,
        0.0,
        (97.46 * DEG_TO_RAD / 2.0).sin() * (281.0 * DEG_TO_RAD).cos(),
        (97.46 * DEG_TO_RAD / 2.0).sin() * (281.0 * DEG_TO_RAD).sin(),
        0.0 + 281.0 * DEG_TO_RAD
    );
}

pub fn leo_dynamics(t: f64, state: &Vector6<f64>) -> Vector6<f64> {

}

// GTO TEST CASE
//
// C_D = 2.2, Mass of sc is 1000kg, cross sectional area 10.0m^2
// 5x5 geopotential, lunisolar perturbations, and atmospheric drag
// Solar flux is held constant as is geomagntic planetary index and amplitude (K_p = 3.0, A_p, F_10.7 = 140 SFU)
// === Initial state ===
// MJD      57249.958333
// a        24326.18 km
// e        0.73
// i        10.0 deg
// raan     310.0 deg
// omega    0.0 deg
// M        180 deg
pub const IT_GTO: f64 = 57249.958333; // MJD
lazy_static! {
    // Note: this is in equinotial elements
    pub static ref ISTATE_GTO: Vector6<f64> = Vector6::new(
        24326.18,
        0.73 * (0.0 + 310.0 * DEG_TO_RAD).sin(),
        0.73 * (0.0 + 310.0 * DEG_TO_RAD).cos(),
        (10.0 * DEG_TO_RAD / 2.0).sin() * (310.0 * DEG_TO_RAD).cos(),
        (10.0 * DEG_TO_RAD / 2.0).sin() * (310.0 * DEG_TO_RAD).sin(),
        180.0 * DEG_TO_RAD + 310 * DEG_TO_RAD
    );
}

pub fn gto_dynamics(t: f64, state: &Vector6<f64>) -> Vector6<f64> {
    const SAT_MASS: f64 = 1000; // kg
    const C_D: f64 = 2.2;
    const CROSS_SEC_AREA: f64 = 10.0; // m^2
}

// HEO TEST CASE
//
// C_D = 2.2, Mass of sc is 1470kg, cross sectional area 15m^2
// 5x5 geopotential, lunisolar perturbations, and atmospheric drag
// Solar flux is held constant as is geomagntic planetary index and amplitude (K_p = 3.0, A_p, F_10.7 = 140 SFU)
// Drag computed through US76 Model only
pub const IT_HEO: f64 = 56664.86336805; // MJD
lazy_static! {
    pub static ref ISTATE_HEO: Vector6<f64> = Vector6::new(
        106247.136454,
        0.75173,
        5.2789 * DEG_TO_RAD,
        49.351 * DEG_TO_RAD,
        180.0 * DEG_TO_RAD,
        0.0
    );
}

pub fn heo_dynamics(t: f64, state: &Vector6<f64>) -> Vector6<f64> {
    const SAT_MASS: f64 = 1470; // kg
    const C_D: f64 = 2.2;
    const CROSS_SEC_AREA: f64 = 15.0; // m^2
}

// Perturbation function
// pulls from Urrutxua H., "High-fidelity models for near-Earth object dynamics," 2015:
// TODO Moon PERT
// TODO Sun PERT

// FIND MOON POSITION
// retrieved from earth
MOON_ORBIT_CB_EARTH: KeplerianState = KeplerianState{
    radius: f64,
    a: f64,
    h: f64,
    incl: f64,
    raan: Option<f64>,
    ecc: f64,
    arg_peri: Option<f64>,
    true_anom: f64,
    time: f64,
}; 
EARTH_ORBIT_CB_SUN: KeplerianState = KeplerianState{ 
    radius: f64,
    a: f64,
    h: f64,
    incl: f64,
    praan: Option<f64>,
    ecc: f64,
    arg_peri: Option<f64>,
    true_anom: f64,
    time: f64,
};
const MU_SUN: f64 = 1.32712442076e20_f64; // m^3 * s^-2
const MU_MOON: f64 = 
enum ThirdBody { 
    SUN,
    MOON, 
    EARTH
}
accel_3rd_body(central_body: ThirdBody, t: f64, state: &Vector6<f64>) -> Vector6<f64> { 
    use ThirdBody::*; 
    let mu: f64
    match central_body { 
        SUN => {
            mu = MU_SUN;           
        }, 
        MOON => {
            mu = MU_MOON;

        }
    }

    // sum from k=1 -> n-1 mu_k * [(r_k - r) / || r_k - r||^3 - r_k / ||r_k||^3]
}

accel_spherical_harmonics(t: f64, state: &Vector6<f64>) -> Vector6<f64> { 
    // Need 5x5 
}

accel_srp(t: f64, state: &Vector6<f64>) -> Vector6<f64> { 
    // -rho(r_sun) * C_R * A / m * vecr / r
    // srp incoming radiation pressure
    Vector6::new(0.0, 0.0, 0.0, a_x, a_y, a_z)
}


/* equinoctial elements

    const SAT_MASS: f64 = 400; // kg
    const C_D: f64 = 2.2;
    const CROSS_SEC_AREA: f64 = 0.7; // m^2

    // got this from http://faculty.nps.edu/dad/orbital/th0.pdf
    let gamma = (1.0 - &state[1].powf(2.0) - &state[2].powf(2.0)).sqrt();
    let na2 = (MU * &state[0]).sqrt();
    let gamma_over_na2 = gamma / na2;
    let inv_2_na2_gamma = 1.0 / (2.0 * na2 * gamma);

    // Differential equations
    let da_dt = 2 * &state[0] / na2 * pert_dl();
    let dh_dt = -gamma / *(na2 * (1 + gamma)) * &state[1] * pert_dl()
        + gamma_over_na2 * pert_dk()
        + inv_2_na2_gamma * &state[2] * (&state[3] * pert_dP() + &state[4] * pert_dQ());
    let dk_dt = -gamma / *(na2 * (1 + gamma)) * &state[2] * pert_dl()
        - gamma_over_na2 * pert_dh()
        - inv_2_na2_gamma * &state[1] * (&state[3] * pert_dP() + &state[4] * pert_dQ());
    let dP_dt = -inv_2_na2_gamma * &state[3] * pert_dl() - 0.5 * inv_2_na2_gamma * pert_dQ()
        + inv_2_na2_gamma * &state[3] * (&state[1] * pert_dk() - &state[2] * pert_dh());
    let dQ_dt = -inv_2_na2_gamma * &state[4] * pert_dl()
        + 0.5 * inv_2_na2_gamma * pert_dP()
        + inv_2_na2_gamma * &state[4] * (&state[1] * pert_dk() - &state[2] * pert_dh());

    let dl_dt = &state[0].powf(2.0) / na2 - 2 * &state[0] / na2 * pert_da()
        + 0.5 * gamma_over_na2 * (&state[2] * pert_dk() + &state[1] * pert_dh())
        + inv_2_na2_gamma() * (&state[3] * pert_dP() + &state[4] * pert_dQ());

    Vector6::new(da_dt, dh_dt, dk_dt, dP_dt, dQ_dt, dl_dt)
*/ 