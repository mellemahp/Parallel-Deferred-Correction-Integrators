/// Test Functions (test_fxns)
///
/// Defines the test cases for use throughout the library
/// The basic test cases are as follows:
/// 1 - D:
/// 2 - D:
/// 3 - D: Keplerian 2 Body orbit and propagator
///

// === Begin Imports ===
// third party imports
extern crate nalgebra as na;
use na::{Vector1, Vector2, Vector3, Vector6};

// standard library
use std::f64::consts::PI;

// === End Imports ===

// Helper functions
pub fn newton_raphson_fdiff<F>(funcd: F, y1: f64, yacc: f64) -> Result<f64, &'static str>
where
    F: Fn(f64) -> f64,
{
    const MAX_ITER: i32 = 100;

    // pre-initialize variables
    let mut fk = funcd(y1);
    let mut fdk = fwd_diff(&funcd, fk, y1);
    let mut y_new: f64;
    let mut y_last = y1;

    // Iterate to victory!
    for _j in 0..MAX_ITER {
        y_new = y_last - fk / fdk;
        if (y_new - y_last).abs() < yacc {
            return Ok(y_new);
        }
        y_last = y_new;
        fk = funcd(y_new);
        fdk = fwd_diff(&funcd, fk, y_new);
    }
    return Err("Maximum Number of Iterations Reached");
}

// Selection of h opt is taken from https://people.sc.fsu.edu/~pbeerli/classes/isc5315-notes/Harvey_Stein-4pages.pdf
// Here they suggest using ~ 7e-6
// Adding auto-diff will replace the need for a step size and should give even better
// accuracy
pub fn fwd_diff<F>(fxn: &F, f_y: f64, y: f64) -> f64
where
    F: Fn(f64) -> f64,
{
    const H_STEP: f64 = 7e-6;

    (fxn(y + H_STEP) - f_y) / H_STEP
}

///=== 1-D Test Problem ===
/// The one dimensional test problem
///
/// Taken from Example 6: Pauls Online notes differential eq.
/// http://tutorial.math.lamar.edu/Classes/DE/Definitions.aspx
/// Validated the example using python's solve_ivp()
///
pub const ONE_D_INIT_TIME: f64 = 1.0;

lazy_static! {
    pub static ref ONE_D_INIT_VAL: Vector1<f64> = Vector1::new(-4.0);
}
pub fn one_d_dynamics(t: f64, y: &Vector1<f64>) -> Vector1<f64> {
    (Vector1::new(3.0) - 4.0 * y) / (2.0 * t)
}

pub fn one_d_solution(t: f64) -> Vector1<f64> {
    Vector1::new(3.0 / 4.0 - 19.0 / (4.0 * t.powf(2.0)))
}

///=== 2-D Test Problem ===
/// We pull this example from example 4 of:
/// https://resources.saylor.org/wwwresources/archived/site/wp-content/uploads/2012/09/MA102-5.5.4-Equations-and-Initial-Value-Problems.pdf
///
/// Once again the answer is also validated with python solve IVP
pub const IT_2_D: f64 = 0.0;
lazy_static! {
    pub static ref IV_2_D: Vector2<f64> = Vector2::new(1.0, 4.0);
}

pub fn two_d_dynamics(t: f64, y: &Vector2<f64>) -> Vector2<f64> {
    Vector2::new(y[1], 2.0 - 6.0 * t)
}

pub fn two_d_solution(t: f64) -> Vector2<f64> {
    let y = t.powf(2.0) - t.powf(3.0) + 4.0 * t + 1.0;
    let dy = 2.0 * t - 3.0 * t.powf(2.0) + 4.0;
    Vector2::new(y, dy)
}

///=== Two body Problem ===
/// Uses a simple keplerian, 2-body orbit

pub const MU: f64 = 398600.4418; // km^3s

/// Dynamics function for Earth 2-Body system
/// i.e. Given current state X, fn(X) -> \dot{X}
/// Note: MU is hard coded here
/// Note: All state values are in km or km/sec
pub fn two_body_dyn(_time: f64, state: Vector6<f64>) -> Vector6<f64> {
    let r3_inv = 1.0
        / (state[0].powf(2.0) + state[0].powf(2.0) + state[0].powf(2.0))
            .sqrt()
            .powf(3.0);
    Vector6::new(
        state[3],
        state[4],
        state[5],
        -MU * state[0] * &r3_inv,
        -MU * state[1] * &r3_inv,
        -MU * state[2] * &r3_inv,
    )
}

/// Spits out the state a given a future time and
/// initial conditions of the keplerian system
#[derive(Debug, Clone, PartialEq)]
pub struct KeplerianState {
    pub radius: f64,
    pub a: f64,
    pub h: f64,
    pub incl: f64,
    pub raan: Option<f64>,
    pub ecc: f64,
    pub arg_peri: Option<f64>,
    pub true_anom: f64,
    pub time: f64,
}
impl KeplerianState {
    // This cartesian state conversion follows the proceedure
    // laid out in Curtis "Orbital Mechanics for engineering students"
    //
    pub fn from_cartesian_state(time: f64, cart_state: Vector6<f64>) -> Self {
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
        let ecc_vec = 1.0 / MU
            * ((velocity.norm().powf(2.0) - MU / radius.norm()) * radius
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
        let a = h.powf(2.0) / (MU * (1.0 - ecc.powf(2.0)));

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
        }
    }

    pub fn get_eccentric_anom(&self) -> f64 {
        2.0 * ((1.0 - self.ecc).sqrt() * (self.true_anom / 2.0).tan())
            .atan2((1.0 + self.ecc).sqrt())
    }

    pub fn propagate_to_time(&self, new_time: f64) -> Self {
        match self.ecc {
            ecc if ecc == 0.0 => {
                let true_anom = new_time * MU.sqrt() / self.radius.powf(3.0 / 2.0);
                KeplerianState {
                    true_anom,
                    time: new_time,
                    ..*self
                }
            }
            _ => {
                let new_mean_anom = MU.powf(2.0) / self.h.powf(3.0)
                    * (1.0 - self.ecc.powf(2.0)).powf(3.0 / 2.0)
                    * new_time;
                let root_problem = |e: f64| e - self.ecc * e.sin() - new_mean_anom;
                let e_new = newton_raphson_fdiff(root_problem, new_mean_anom, 1e-13).unwrap();
                let ta_new = ((e_new.cos() - self.ecc) / (1.0 - self.ecc * e_new.cos())).acos();
                let r_new = self.h.powf(2.0) / MU * 1.0 / (1.0 + self.ecc * ta_new.cos());

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

#[cfg(test)]
mod tests {
    use super::*;

    const TOL: f64 = 1.0e-12;

    #[test]
    fn test_2body_dynamics() {
        let state_1 = Vector6::new(1000.0, 1000.0, 1000.0, 1.0, 1.0, 1.0);
        let deriv = two_body_dyn(1.0, state_1);
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
        };
        let new_state_a = start_state.propagate_to_time(866.77);
        let curtis_true_anom_a = 1.00222042;
        const TOL: f64 = 1.0e-3;
        assert!((new_state_a.true_anom - curtis_true_anom_a).abs() < TOL);
    }

    /// This test case is taken from example 4.3 in Curtis
    /// pg 200
    #[test]
    fn test_cart_to_kep() {
        let state = Vector6::new(-6045.0, -3490.0, 2500.0, -3.457, 6.618, 2.533);
        let kep_state = KeplerianState::from_cartesian_state(0.0, state);
        let big_tols = 2.0; // the numbers in curtis are very aggressively rounded
        let small_tols = 0.001;
        let curtis_state = KeplerianState {
            radius: 7414.0,
            h: 58310.0,
            incl: 2.6738, // curtis lists this in degres, but radians are used here
            raan: Some(4.4558),
            ecc: 0.1712,
            arg_peri: Some(0.3503),
            true_anom: 0.49654617,
            time: 0.0,
            a: 0.0,
        };
        assert!((kep_state.radius - curtis_state.radius).abs() < big_tols);
        assert!((kep_state.h - curtis_state.h).abs() < big_tols);
        assert!((kep_state.incl - curtis_state.incl).abs() < small_tols);
        assert!((kep_state.raan.unwrap() - curtis_state.raan.unwrap()).abs() < small_tols);
        assert!((kep_state.ecc - curtis_state.ecc).abs() < small_tols);
        assert!((kep_state.arg_peri.unwrap() - curtis_state.arg_peri.unwrap()).abs() < small_tols);
        assert!((kep_state.true_anom - curtis_state.true_anom).abs() < small_tols);

        println!("{:?}", kep_state);
    }

    #[test]
    fn test_cartesian_conversion() {
        // test 1
        let state = Vector6::new(-6045.0, -3490.0, 2500.0, -3.457, 6.618, 2.533);
        let kep_state = KeplerianState::from_cartesian_state(0.0, state);
        let re_convert = kep_state.into_cartesian();
        let diffs = state - re_convert;
        const TOL: f64 = 1.0e-2;
        for val in diffs.iter() {
            assert!(val.abs() < TOL);
        }

        // test 2
        let state2 = Vector6::new(-7000.0, -7500.0, 6000.0, -3.457, 5.0, 2.533);
        let kep_state2 = KeplerianState::from_cartesian_state(0.0, state2);
        let re_convert2 = kep_state2.into_cartesian();
        let diffs2 = state2 - re_convert2;
        for val in diffs2.iter() {
            assert!(val.abs() < TOL);
        }
    }
}
