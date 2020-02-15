//! Fluke is a Rust library for solving ODE Initial Value Prolbmes
//!
//! ### Full Examples
//!
//! Full examples, detailing and explaining usage of the basic functionality of the
//! library, can be found in the [`examples`] directory.
//!
//! # Installation
//!
//!
//!
#[macro_use]
extern crate lazy_static;

pub mod adams;
//pub mod euler;
pub mod lagrange;
pub mod newton_raphson;
pub mod ridc;
pub mod runge_kutta;
pub mod test_fxns;
//pub mod comp_fxns;
