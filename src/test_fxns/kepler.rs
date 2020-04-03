/// === Two body Problem ===
/// Provides functions and methods for computing a simple keplerian, 2-body orbit
/// Also provides a set of higher order perturbations that can be added to the basic
/// two body problem  
///
/// The following perturbations are provided for testing:
/// - Drag (exponential model)
/// - J2
/// - J3
/// - Moon point mass (mean elements)
/// - Sun point mass (circular earth orbit)
/// NOTE: Sun and moon point mass are only for testing and are
/// not physically accurate models. Ephemerides should be used for
/// practical applications
///
// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::{Vector3, Vector6};

// standard library
use std::f64::consts::PI;

// local imports
use super::utils::newton_raphson_fdiff;

// === End Imports ===

// === Constants ===
// Gravitational constants
pub const MU_EARTH: f64 = 398600.4418; // km^3s
pub const MU_MOON: f64 = 0.004903e6; // km^3/s^2
pub const MU_SUN: f64 = 132712e6; // km^3/s^2

// gravity perturbations
pub const J2: f64 = 1.082626925638815e-03; // j2 gravity field coefficient, normalized
pub const J3: f64 = -0.0000025323; // j3 gravity field coefficient, normalized

// drag model
pub const R_E: f64 = 6378.1363; // radius of the earth (km)
pub const H_0: f64 = 88667.0; // reference height (m)
pub const R_0: f64 = 700000.0 + R_E * 1000.0; // reference radius (m)
pub const RHO_0: f64 = 0.0003614; // kg / km^3
pub const C_D: f64 = 2.0; // unitless
pub const A_SAT: f64 = 3.0e-6; // cross sectional area of satellite (km^2)
pub const MASS: f64 = 970.0; // kg

/// Keplerian state object
#[derive(Debug, Clone, PartialEq)]
pub struct KeplerianState {
    // Orbit radius at time t in km
    pub radius: f64,
    // Semi major axis of orbit in km
    pub a: f64,
    // Orbit angular momentum
    pub h: f64,
    // Inclination of orbit in Radians
    pub incl: f64,
    // Right ascension of the ascending node in Radians
    pub raan: Option<f64>,
    // Eccentricity of the orbit [0, 1]
    pub ecc: f64,
    // Argument of Perigee in Radians
    pub arg_peri: Option<f64>,
    // True anomaly in Radians
    pub true_anom: f64,
    // Current Time in Radians
    pub time: f64,
    // Gravitational Parameter of central body
    pub mu: f64,
}
impl KeplerianState {
    pub fn from_peri_rad(
        // Peripsis radius
        peri_rad: f64,
        // Eccentricity
        ecc: f64,
        // Inclination, rad
        incl: f64,
        // Right ascension of the ascending node, rad
        raan: f64,
        // Argument of perigee, rad
        arg_peri: f64,
        // Time since ref epoch, sec
        time: f64,
        // Gravitational parameter of central body, default = earth
        mu: Option<f64>,
    ) -> Self {
        let mu = mu.unwrap_or(MU_EARTH);
        let a = peri_rad / (1.0 - ecc);
        let h = (a * mu * (1.0 - ecc.powf(2.0))).sqrt();

        KeplerianState {
            radius: peri_rad,
            a,
            h,
            incl,
            raan: Some(raan),
            ecc,
            arg_peri: Some(arg_peri),
            true_anom: 0.0,
            time,
            mu,
        }
    }

    // This cartesian state conversion follows the proceedure
    // laid out in Curtis "Orbital Mechanics for engineering students"
    pub fn from_cartesian_state(
        // Time since ref epoch, sec
        time: f64,
        // Cartesian state vector ECI
        cart_state: Vector6<f64>,
        // Gravitational parameter of central body, default = earth
        mu: Option<f64>,
    ) -> Self {
        let mu = mu.unwrap_or(MU_EARTH);
        let radius = Vector3::<f64>::new(cart_state[0], cart_state[1], cart_state[2]);
        let velocity = Vector3::<f64>::new(cart_state[3], cart_state[4], cart_state[5]);
        let v_radial = radius.dot(&velocity) / radius.norm();

        // Find angular momentum
        let h_vec = radius.cross(&velocity);
        let h = h_vec.norm();

        // find inclination
        let incl = (h_vec[2] / h).acos();

        // find node
        let node_vec = Vector3::<f64>::new(0.0, 0.0, 1.0).cross(&h_vec);
        let node_mag = node_vec.norm();

        // calculate raan
        let raan: Option<f64>;
        if node_mag == 0.0 {
            // circular orbit case
            raan = None;
        } else if node_vec[1] >= 0.0 {
            raan = Some((node_vec[0] / node_mag).acos());
        } else {
            raan = Some(2.0 * PI - (node_vec[0] / node_mag).acos());
        }

        // calculate eccentricity
        let ecc_vec = 1.0 / mu
            * ((velocity.norm().powf(2.0) - mu / radius.norm()) * radius
                - (v_radial * radius.norm()) * velocity);
        let ecc = ecc_vec.norm();

        // calculate argument of perigee
        let arg_peri: Option<f64>;
        if ecc == 0.0 || node_mag == 0.0 {
            // circular orbit case
            arg_peri = None;
        } else if ecc_vec[2] >= 0.0 {
            arg_peri = Some((node_vec / node_mag).dot(&(ecc_vec / ecc)).acos());
        } else {
            arg_peri = Some(2.0 * PI - (node_vec / node_mag).dot(&(ecc_vec / ecc)).acos());
        }

        // calculate true anomaly
        let true_anom: f64;
        if v_radial >= 0.0 {
            true_anom = (ecc_vec / ecc).dot(&(radius / radius.norm())).acos();
        } else {
            true_anom = 2.0 * PI - (ecc_vec / ecc).dot(&(radius / radius.norm())).acos();
        }

        // calculate SMA
        let a = h.powf(2.0) / (mu * (1.0 - ecc.powf(2.0)));

        KeplerianState {
            radius: radius.norm(),
            a,
            h,
            incl,
            raan,
            ecc,
            arg_peri,
            true_anom,
            time,
            mu,
        }
    }

    pub fn get_eccentric_anom(&self) -> f64 {
        2.0 * ((1.0 - self.ecc).sqrt() * (self.true_anom / 2.0).tan())
            .atan2((1.0 + self.ecc).sqrt())
    }

    pub fn propagate_to_time(
        &self,
        // time since ref epoch to propagate to in sec
        new_time: f64,
    ) -> Self {
        match self.ecc {
            ecc if ecc == 0.0 => {
                let true_anom = new_time * self.mu.sqrt() / self.radius.powf(3.0 / 2.0);
                KeplerianState {
                    true_anom,
                    time: new_time,
                    ..*self
                }
            }
            _ => {
                let new_mean_anom = self.mu.powf(2.0) / self.h.powf(3.0)
                    * (1.0 - self.ecc.powf(2.0)).powf(3.0 / 2.0)
                    * new_time;
                let root_problem = |e: f64| e - self.ecc * e.sin() - new_mean_anom;
                let e_new = newton_raphson_fdiff(root_problem, new_mean_anom, 1e-12_f64).unwrap();
                let ta_new = ((e_new.cos() - self.ecc) / (1.0 - self.ecc * e_new.cos())).acos();
                let r_new = self.h.powf(2.0) / self.mu * 1.0 / (1.0 + self.ecc * ta_new.cos());

                KeplerianState {
                    radius: r_new,
                    true_anom: ta_new,
                    time: new_time,
                    ..*self
                }
            }
        }
    }
    pub fn into_cartesian(&self) -> Vector6<f64> {
        let raan = self.raan.unwrap();
        let t_lat = self.arg_peri.unwrap() + self.true_anom;
        let i = self.incl;
        let r = self.radius;

        let x = r * (raan.cos() * (t_lat).cos() - raan.sin() * (t_lat).sin() * i.cos());
        let y = r * (raan.sin() * (t_lat).cos() + raan.cos() * (t_lat).sin() * i.cos());
        let z = r * (i.sin() * (t_lat).sin());
        let rp = r * self.a * (1.0 - self.ecc.powf(2.0));
        let dx = x * self.h * self.ecc / rp * self.true_anom.sin()
            - self.h / r * (raan.cos() * (t_lat).sin() + raan.sin() * (t_lat).cos() * i.cos());
        let dy = y * self.h * self.ecc / rp * self.true_anom.sin()
            - self.h / r * (raan.sin() * (t_lat).sin() - raan.cos() * (t_lat).cos() * i.cos());
        let dz = z * self.h * self.ecc / rp * self.true_anom.sin()
            + self.h / r * (i.sin() * (t_lat).cos());

        Vector6::new(x, y, z, dx, dy, dz)
    }
}

/// == Istates ==
/// The following LEO, GTO, and HEO test cases are pulled from:
/// Amato D. et Al "Non-averaged regularized formulations as an alternative to ..."

// LEO Test case
// C_D = 2.2, Mass of sc is 400kg, cross sectional area 0.7m^2
// 5x5 geopotential, lunisolar perturbations, and atmospheric drag
// Solar flux is held constant as is geomagntic planetary index and amplitude (K_p = 3.0, A_p, F_10.7 = 140 SFU)
// === Initial state ===
// a    6862.14 km
// e    0.0
// i    97.46 deg
// raan 281.0 deg
// u    0.0 deg
pub const DEG_TO_RAD: f64 = PI / 180.0;

pub const IT_LEO: f64 = 0.0; // MJD
lazy_static! {
    pub static ref ISTATE_LEO: KeplerianState = KeplerianState::from_peri_rad(
        6862.14,
        0.0,
        97.46 * DEG_TO_RAD,
        2810.0 * DEG_TO_RAD,
        0.0,
        0.0,
        None
    );
}

// GTO TEST CASE
// === Initial state ===
// MJD      57249.958333
// a        24326.18 km
// e        0.73
// i        10.0 deg
// raan     310.0 deg
// omega    0.0 deg
// M        180 deg
pub const IT_GTO: f64 = 0.0; // MJD
lazy_static! {
    // Note: this is in equinotial elements
    pub static ref ISTATE_GTO: KeplerianState = KeplerianState::from_peri_rad(
        24326.18 * (1.0 - 0.73),
        0.73,
        10.0 * DEG_TO_RAD,
        310.0 * DEG_TO_RAD,
        0.0,
        0.0,
        None
    );
}

// HEO TEST CASE
// === Initial state ===
// MJD      57249.958333
// a        106247.135454 km
// e        0.75173
// i        5.2789
// raan     49.351 deg
// omega    180.0 deg
// M        0 deg
pub const IT_HEO: f64 = 0.0; // MJD
lazy_static! {
    pub static ref ISTATE_HEO: KeplerianState = KeplerianState::from_peri_rad(
        106247.135454 * (1.0 - 0.75173),
        0.75173,
        5.2789 * DEG_TO_RAD,
        49.351 * DEG_TO_RAD,
        180.0 * DEG_TO_RAD,
        0.0,
        None,
    );
}

/// Dynamics function for Earth 2-Body system
/// i.e. Given current state X, fn(X) -> \dot{X}
/// Note: MU is hard coded here
/// Note: All state values are in km or km/sec
pub fn two_body_dyn(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let r3_inv = 1.0
        / (state[0].powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0))
            .sqrt()
            .powf(3.0);
    Vector6::new(
        state[3],
        state[4],
        state[5],
        -MU_EARTH * state[0] * &r3_inv,
        -MU_EARTH * state[1] * &r3_inv,
        -MU_EARTH * state[2] * &r3_inv,
    )
}

///=== Perturbations ===

// == Drag perturbations ==
pub fn drag_pert(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let r = (state[0].powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0)).sqrt() * 1000.0; // converts to m
    let rho = RHO_0 * (-(r - R_0) / H_0).exp(); // non-dimensional so we do not need to convert back to km
    let f_drag_mag = -1.0 / 2.0
        * rho
        * (C_D * A_SAT / MASS)
        * (state[0].powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0)).sqrt();
    Vector6::new(
        0.0,
        0.0,
        0.0,
        f_drag_mag * state[3],
        f_drag_mag * state[4],
        f_drag_mag * state[5],
    )
}

// == Non-spherical Gravity perturbations ==
pub fn j2_pert(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let r = (state[0].powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0)).sqrt();
    let r_2_inv = 1.0 / r.powf(2.0);
    let r_5_inv_scaled = 1.0 / (2.0 * r.powf(5.0));
    let z2 = state[2].powf(2.0);
    Vector6::new(
        0.0,
        0.0,
        0.0,
        -3.0 * J2 * state[0] * r_5_inv_scaled * (1.0 - 5.0 * z2 * r_2_inv),
        -3.0 * J2 * state[1] * r_5_inv_scaled * (1.0 - 5.0 * z2 * r_2_inv),
        -3.0 * J2 * state[2] * r_5_inv_scaled * (3.0 - 5.0 * z2 * r_2_inv),
    )
}

pub fn j3_pert(_time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let r = (state[0].powf(2.0) + state[1].powf(2.0) + state[2].powf(2.0)).sqrt();
    let re_3 = R_E.powf(3.0);
    let r_7_inv_scaled = 1.0 / (2.0 * r.powf(7.0));
    let r_2_inv = 1.0 / r.powf(2.0);
    let z_2 = state[2].powf(2.0);
    let z_3 = state[2] * z_2;
    let z_4 = state[2] * z_3;

    Vector6::new(
        0.0,
        0.0,
        0.0,
        -5.0 * J3
            * MU_EARTH
            * re_3
            * state[0]
            * r_7_inv_scaled
            * (3.0 * state[2] - 7.0 * z_3 * r_2_inv),
        -5.0 * J3
            * MU_EARTH
            * re_3
            * state[1]
            * r_7_inv_scaled
            * (3.0 * state[2] - 7.0 * z_3 * r_2_inv),
        -5.0 * J3
            * MU_EARTH
            * re_3
            * state[2]
            * r_7_inv_scaled
            * (6.0 * z_2 - 7.0 * z_4 * r_2_inv - 3.0 / 5.0 * r.powf(2.0)),
    )
}

// == Third Body perturbations ==
lazy_static! {
    pub static ref MOON_ORBIT: KeplerianState =
        KeplerianState::from_peri_rad(383397.7725, 0.0, 0.4984066932, 0.0, 0.0, 0.0, None,);
}

pub fn moon_pert(time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let new_moon_state_kep = MOON_ORBIT.propagate_to_time(time);
    let new_moon_state_cart = new_moon_state_kep.into_cartesian();
    let r_sat_moon = new_moon_state_cart - state;
    let r3_inv = 1.0
        / (r_sat_moon[0].powf(2.0) + r_sat_moon[1].powf(2.0) + r_sat_moon[2].powf(2.0))
            .sqrt()
            .powf(3.0);
    Vector6::new(
        0.0,
        0.0,
        0.0,
        MU_MOON * r_sat_moon[0] * &r3_inv,
        MU_MOON * r_sat_moon[1] * &r3_inv,
        MU_MOON * r_sat_moon[2] * &r3_inv,
    )
}

lazy_static! {
    pub static ref EARTH_ORBIT: KeplerianState =
        KeplerianState::from_peri_rad(149.60e6, 0.0, 0.401425728, 0.0, 0.0, 0.0, Some(MU_SUN));
}

pub fn sun_pert(time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    let new_earth_state_kep = EARTH_ORBIT.propagate_to_time(time);
    let new_sun_state_cart = -1.0 * new_earth_state_kep.into_cartesian();
    let r_sat_sun = new_sun_state_cart - state;
    let r3_inv = 1.0
        / (r_sat_sun[0].powf(2.0) + r_sat_sun[1].powf(2.0) + r_sat_sun[2].powf(2.0))
            .sqrt()
            .powf(3.0);
    Vector6::new(
        0.0,
        0.0,
        0.0,
        MU_SUN * r_sat_sun[0] * &r3_inv,
        MU_SUN * r_sat_sun[1] * &r3_inv,
        MU_SUN * r_sat_sun[2] * &r3_inv,
    )
}

// Dynamics with all available perturbations added
pub fn full_perturbed_2body_dyn(time: f64, state: &Vector6<f64>) -> Vector6<f64> {
    two_body_dyn(time, state)
        + drag_pert(time, state)
        + j2_pert(time, state)
        + j3_pert(time, state)
        + moon_pert(time, state)
        + sun_pert(time, state)
}

#[cfg(test)]
mod tests {
    use super::*;
    const TOL: f64 = 1.0e-12;

    #[test]
    fn test_2body_dynamics() {
        let state_1 = Vector6::new(1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0);
        let deriv = two_body_dyn(1.0, &state_1);
        let true_deriv = Vector6::new(
            1.0,
            1.0,
            1.0,
            -0.07671069079,
            -0.07671069079,
            -0.07671069079,
        );
        let diffs = true_deriv - deriv;
        for diff in diffs.iter() {
            assert!(diff.abs() < TOL);
        }
    }
    // Taken from example 3.3 in curtis
    #[test]
    fn test_2body_propagation() {
        let start_state = KeplerianState {
            radius: 6878.0,
            h: 58458.0,
            incl: 0.0,
            raan: Some(0.0),
            ecc: 0.24649,
            arg_peri: Some(0.0),
            true_anom: 0.0,
            time: 0.0,
            a: 9128.0,
            mu: MU_EARTH,
        };
        let new_state_a = start_state.propagate_to_time(866.77);
        let curtis_true_anom_a = 1.00222042;
        const TOL: f64 = 1.0e-3;
        println!(
            "TRUE {:?} | EST {:?}",
            curtis_true_anom_a, new_state_a.true_anom
        );
        assert!((new_state_a.true_anom - curtis_true_anom_a).abs() < TOL);
    }

    /// This test case is taken from example 4.3 in Curtis
    /// pg 200
    #[test]
    fn test_cart_to_kep() {
        let state = Vector6::new(-6045.0, -3490.0, 2500.0, -3.457, 6.618, 2.533);
        let kep_state = KeplerianState::from_cartesian_state(0.0, state, None);
        let big_tols = 2.0; // the numbers in curtis are very aggressively rounded
        let small_tols = 0.001;
        let curtis_state = KeplerianState {
            radius: 7414.0,
            h: 58310.0,
            incl: 2.6738, // curtis lists this in degrees, but radians are used here
            raan: Some(4.4558),
            ecc: 0.1712,
            arg_peri: Some(0.3503),
            true_anom: 0.49654617,
            time: 0.0,
            a: 0.0,
            mu: MU_EARTH,
        };
        assert!((kep_state.radius - curtis_state.radius).abs() < big_tols);
        assert!((kep_state.h - curtis_state.h).abs() < big_tols);
        assert!((kep_state.incl - curtis_state.incl).abs() < small_tols);
        assert!((kep_state.raan.unwrap() - curtis_state.raan.unwrap()).abs() < small_tols);
        assert!((kep_state.ecc - curtis_state.ecc).abs() < small_tols);
        assert!((kep_state.arg_peri.unwrap() - curtis_state.arg_peri.unwrap()).abs() < small_tols);
        assert!((kep_state.true_anom - curtis_state.true_anom).abs() < small_tols);
    }

    #[test]
    fn test_cartesian_conversion() {
        // test 1
        let state = Vector6::new(-6045.0, -3490.0, 2500.0, -3.457, 6.618, 2.533);
        let kep_state = KeplerianState::from_cartesian_state(0.0, state, None);
        let re_convert = kep_state.into_cartesian();
        let diffs = state - re_convert;
        const TOL: f64 = 1.0e-2;
        for val in diffs.iter() {
            assert!(val.abs() < TOL);
        }

        // test 2
        let state2 = Vector6::new(-7000.0, -7500.0, 6000.0, -3.457, 5.0, 2.533);
        let kep_state2 = KeplerianState::from_cartesian_state(0.0, state2, None);
        let re_convert2 = kep_state2.into_cartesian();
        let diffs2 = state2 - re_convert2;
        for val in diffs2.iter() {
            assert!(val.abs() < TOL);
        }
    }
}
