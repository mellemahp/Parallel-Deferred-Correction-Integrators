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
    use nalgebra::{Matrix4, Matrix6, MatrixN, Vector4, Vector6, VectorN, U13, U4, U6};

    // Types for 8th order integrators
    type Vector13<N> = VectorN<N, U13>;
    type Matrix13<N> = MatrixN<N, U13>;

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

    // Dormand Prince 7(8) from GMAT math spec
    lazy_static! {
        pub static ref DOPRI78: EmbeddedRKStepper<U13> = EmbeddedRKStepper::new(
            "Cash-Karp 4(5)",
            EmbeddedTableau {
                a_vals: Matrix13::from_row_slice(&[
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / 18.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / 48.0,
                    1.0 / 16.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    1.0 / 32.0,
                    0.0,
                    3.0 / 32.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    5.0 / 16.0,
                    0.0,
                    -75.0 / 64.0,
                    75.0 / 64.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    3.0 / 80.0,
                    0.0,
                    0.0,
                    3.0 / 16.0,
                    3.0 / 20.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    29443841.0 / 614563906.0,
                    0.0,
                    0.0,
                    77736538.0 / 692538347.0,
                    -28693883.0 / 1125000000.0,
                    23124283.0 / 1800000000.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    16016141.0 / 946692911.0,
                    0.0,
                    0.0,
                    61564180.0 / 158732637.0,
                    22789713.0 / 633445777.0,
                    545815736.0 / 2771057229.0,
                    -180193667.0 / 1043307555.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    39632708.0 / 573591083.0,
                    0.0,
                    0.0,
                    -433636366.0 / 683701615.0,
                    -421739975.0 / 2616292301.0,
                    100302831.0 / 723423059.0,
                    790204164.0 / 839813087.0,
                    800635310.0 / 3783071287.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    246121993.0 / 1340847787.0,
                    0.0,
                    0.0,
                    -37695042795.0 / 15268766246.0,
                    -309121744.0 / 1061227803.0,
                    -12992083.0 / 490766935.0,
                    6005943493.0 / 2108947869.0,
                    393006217.0 / 1396673457.0,
                    123872331.0 / 1001029789.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -1028468189.0 / 846180014.0,
                    0.0,
                    0.0,
                    8478235783.0 / 508512852.0,
                    1311729495.0 / 1432422823.0,
                    -10304129995.0 / 1701304382.0,
                    -48777925059.0 / 3047939560.0,
                    15336726248.0 / 1032824649.0,
                    -45442868181.0 / 3398467696.0,
                    3065993473.0 / 597172653.0,
                    0.0,
                    0.0,
                    0.0,
                    185892177.0 / 718116043.0,
                    0.0,
                    0.0,
                    -3185094517.0 / 667107341.0,
                    -477755414.0 / 1098053517.0,
                    -703635378.0 / 230739211.0,
                    5731566787.0 / 1027545527.0,
                    5232866602.0 / 850066563.0,
                    -4093664535.0 / 808688257.0,
                    3962137247.0 / 1805957418.0,
                    65686358.0 / 487910083.0,
                    0.0,
                    0.0,
                    403863854.0 / 491063109.0,
                    0.0,
                    0.0,
                    -5068492393.0 / 434740067.0,
                    -411421997.0 / 543043805.0,
                    652783627.0 / 914296604.0,
                    11173962825.0 / 925320556.0,
                    -13158990841.0 / 6184727034.0,
                    3936647629.0 / 1978049680.0,
                    -160528059.0 / 685178525.0,
                    248638103.0 / 1413531060.0,
                    0.0,
                    0.0
                ]),

                b_vals: Vector13::from_row_slice(&[
                    14005451.0 / 335480064.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -59238493.0 / 1068277825.0,
                    181606767.0 / 758867731.0,
                    561292985.0 / 797845732.0,
                    -1041891430.0 / 13713433529.0,
                    760417239.0 / 1151165299.0,
                    118820643.0 / 751138087.0,
                    -528747749.0 / 2220607170.0,
                    0.0
                ]),

                b_hat_vals: Vector13::from_row_slice(&[
                    13451932.0 / 455176623.0,
                    0.0,
                    0.0,
                    0.0,
                    0.0,
                    -808719846.0 / 976000145.0,
                    1757004468.0 / 5645159321.0,
                    656045339.0 / 265891186.0,
                    -3867574721.0 / 1518517206.0,
                    465885868.0 / 322736535.0,
                    53011238.0 / 667516719.0,
                    2.0 / 45.0,
                    0.0
                ]),
                c_vals: Vector13::from_row_slice(&[
                    0.0,
                    1.0 / 18.0,
                    1.0 / 12.0,
                    1.0 / 8.0,
                    5.0 / 16.0,
                    3.0 / 8.0,
                    59.0 / 400.0,
                    93.0 / 200.0,
                    5490023248.0 / 9719169821.0,
                    13.0 / 20.0,
                    1201146811.0 / 1299019798.0,
                    1.0,
                    1.0
                ]),
            }
        )
        .unwrap();
    }
}
