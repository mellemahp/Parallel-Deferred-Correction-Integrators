/// Main Entrypoint for Runge Kutta Integrators
/// Contains a set of both pre-built integrators as
/// well as base integrators useful for creating new
/// RK and RKF integrators from a set of Butcher Tableaus
extern crate nalgebra;
pub mod adaptive;
pub mod base;
pub mod common;
pub mod embedded;
pub mod fixed;
pub mod tableaus;
// === PRE-BUILT: Simple ===
// Note: only explicit integrators are provided
//  at the current time
/// All prebuilts use butcher tables from:
/// https://en.wikipedia.org/wiki/List_of_Runge%E2%80%93Kutta_methods
pub mod rk_simp {
    use super::base::RKStepper;
    use super::nalgebra::{Matrix2, Matrix4, Vector2, Vector4, U2, U4};
    use super::tableaus::Tableau;

    // RK2 also called explicit midpoint method
    lazy_static! {
        pub static ref RK2: RKStepper<U2> = RKStepper::new(
            "RK2",
            Tableau {
                a_vals: Matrix2::new(0.0, 0.0, 0.5, 0.0),
                b_vals: Vector2::new(0.0, 1.0),
                c_vals: Vector2::new(0.0, 0.5),
            }
        )
        .unwrap();
    }
    // Heuns Method
    lazy_static! {
        pub static ref HEUN: RKStepper<U2> = RKStepper::new(
            "Heuns Method",
            Tableau {
                a_vals: Matrix2::new(0.0, 0.0, 1.0, 0.0),
                b_vals: Vector2::new(0.5, 0.5),
                c_vals: Vector2::new(0.0, 1.0),
            }
        )
        .unwrap();
    }

    // RK4 ("Original Runge-Kutta")
    lazy_static! {
        pub static ref RK4: RKStepper<U4> = RKStepper::new(
            "Classic RK4",
            Tableau {
                a_vals: Matrix4::new(
                    0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 0.5, 0.0, 0.0, 0.0, 0.0, 1.0, 0.0,
                ),
                b_vals: Vector4::new(1.0 / 6.0, 1.0 / 3.0, 1.0 / 3.0, 1.0 / 6.0),
                c_vals: Vector4::new(0.0, 0.5, 0.5, 1.0),
            }
        )
        .unwrap();
    }
}

/// === PRE-BUILT: Embedded ===
// Note: only explicit integrators are provided
//  at the current time
pub mod rk_embed {
    use super::embedded::EmbeddedRKStepper;
    use super::tableaus::EmbeddedTableau;
    use nalgebra::{Matrix4, Matrix6, Vector4, Vector6, U4, U6};

    // RK 3(2) (Bogacki Shampine Method)
    lazy_static! {
        pub static ref RK32: EmbeddedRKStepper<U4> = EmbeddedRKStepper::new(
            "Bogacki-Shampine 3(2)",
            EmbeddedTableau {
                a_vals: Matrix4::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.5,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.75,
                    0.0,
                    0.0,
                    2.0 / 9.0,
                    1.0 / 3.0,
                    4.0 / 9.0,
                    0.0,
                ),
                c_vals: Vector4::new(0.0, 0.5, 0.75, 1.0),
                b_vals: Vector4::new(2.0 / 9.0, 1.0 / 3.0, 4.0 / 9.0, 0.0),
                b_hat_vals: Vector4::new(7.0 / 24.0, 0.25, 1.0 / 3.0, 1.0 / 8.0),
            }
        )
        .unwrap();
    }

    // RK 4(5) ("Classic" Runge-Kutta-Fehlberg)
    lazy_static! {
        pub static ref RKF45: EmbeddedRKStepper<U6> = EmbeddedRKStepper::new(
            "Runge-Kutta-Fehlberg 4(5)",
            EmbeddedTableau {
                a_vals: Matrix6::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.25,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    3.0 / 32.0,
                    9.0 / 32.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1932.0 / 2197.0,
                    -7200.0 / 2197.0,
                    7296.0 / 2197.0,
                    0.0,
                    0.0,
                    0.0,
                    439.0 / 216.0,
                    -8.0,
                    3680.0 / 513.0,
                    -845.0 / 4104.0,
                    0.0,
                    0.0,
                    -8.0 / 27.0,
                    2.0,
                    -3544.0 / 2565.0,
                    1859.0 / 4104.0,
                    -11.0 / 40.0,
                    0.0,
                ),
                c_vals: Vector6::new(0.0, 0.25, 3.0 / 8.0, 12.0 / 13.0, 1.0, 0.5),
                b_vals: Vector6::new(
                    16.0 / 135.0,
                    0.0,
                    6656.0 / 12825.0,
                    28561.0 / 56430.0,
                    -9.0 / 50.0,
                    2.0 / 55.0,
                ),
                b_hat_vals: Vector6::new(
                    25.0 / 216.0,
                    0.0,
                    1408.0 / 2565.0,
                    2197.0 / 4104.0,
                    -1.0 / 5.0,
                    0.0,
                ),
            }
        )
        .unwrap();
    }

    // Cash-Karp 4th order
    lazy_static! {
        pub static ref CASH_KARP45: EmbeddedRKStepper<U6> = EmbeddedRKStepper::new(
            "Cash-Karp 4(5)",
            EmbeddedTableau {
                a_vals: Matrix6::new(
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / 5.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    3.0 / 40.0,
                    9.0 / 40.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    3.0 / 10.0,
                    -9.0 / 10.0,
                    6.0 / 5.0,
                    0.0,
                    0.0,
                    0.0,
                    -11.0 / 54.0,
                    5.0 / 2.0,
                    -70.0 / 27.0,
                    35.0 / 27.0,
                    0.0,
                    0.0,
                    1631.0 / 55296.0,
                    175.0 / 512.0,
                    575.0 / 13824.0,
                    44275.0 / 110592.0,
                    253.0 / 4096.0,
                    0.0,
                ),
                b_vals: Vector6::new(
                    37.0 / 378.0,
                    0.0,
                    250.0 / 621.0,
                    125.0 / 594.0,
                    0.0,
                    512.0 / 1771.0,
                ),
                b_hat_vals: Vector6::new(
                    2825.0 / 27648.0,
                    0.0,
                    18575.0 / 48384.0,
                    13525.0 / 55296.0,
                    277.0 / 14336.0,
                    1.0 / 4.0,
                ),
                c_vals: Vector6::new(0.0, 1.0 / 5.0, 3.0 / 10.0, 3.0 / 5.0, 1.0, 7.0 / 8.0),
            }
        )
        .unwrap();
    }
}
