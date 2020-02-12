/// Legendre Quadrature (quadrature)
///
///
// === Begin Imports ===
// third party imports

// standard library
use std::collections::VecDeque;

// === Begin Imports ===

fn get_x_pow(x_0: f64, x: f64, order: usize) -> Vec<f64> {
    let mut x_pow: Vec<f64> = Vec::with_capacity(order);
    for i in 1..order + 1 {
        x_pow.push(x.powi(i as i32) - x_0.powi(i as i32))
    }
    x_pow
}

fn w_0(x_pow: &Vec<f64>) -> f64 {
    x_pow[0]
}

fn w_1(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    0.5 * x_pow[1] - times[0] * x_pow[0]
}

fn w_2(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 6.0
        * (6.0 * times[0] * times[1] * x_pow[0] - 3.0 * (times[0] + times[1]) * x_pow[1]
            + 2.0 * x_pow[2])
}

fn w_3(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 12.0
        * (-12.0 * times[0] * times[1] * times[2] * x_pow[0]
            + 6.0 * (times[0] * (times[1] + times[2]) + times[1] * times[2]) * x_pow[1]
            - 4.0 * (times[0] + times[1] + times[2]) * x_pow[2]
            + 3.0 * x_pow[3])
}

fn w_4(x_pow: &Vec<f64>, times: &VecDeque<f64>) -> f64 {
    1.0 / 60.0
        * (60.0 * times[0] * times[1] * times[2] * times[3] * x_pow[0]
            - 30.0
                * (times[0] * times[1] * (times[2] * times[3])
                    + times[0] * times[2] * times[3]
                    + times[1] * times[2] * times[3])
                * x_pow[1]
            + 20.0
                * (times[0] * (times[1] + times[2] + times[3])
                    + times[1] * (times[2] + times[3])
                    + times[2] * times[3])
                * x_pow[2]
            - 15.0 * (times[0] + times[1] + times[2] + times[3]) * x_pow[3]
            + 12.0 * x_pow[4])
}

// Tests
#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_x_pow() {
        // test trivial cases
        let x_0: f64 = 0.0;
        let x: f64 = 1.0;
        let ans = get_x_pow(x_0, x, 4);

        for i in 0..ans.len() {
            assert_eq!(ans[i], 1.0);
        }
        let x_0: f64 = 4.0;
        let x: f64 = 4.0;
        let ans = get_x_pow(x_0, x, 4);

        for i in 0..ans.len() {
            assert_eq!(ans[i], 0.0);
        }

        // test one hand calculated one
        let x_0: f64 = 2.0;
        let x: f64 = 3.0;
        let ans = get_x_pow(x_0, x, 4);
        let truth = vec![1.0, 5.0, 19.0, 65.0];
        for i in 0..ans.len() {
            assert_eq!(ans[i], truth[i]);
        }
    }
}
