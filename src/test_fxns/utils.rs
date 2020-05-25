/// Utility Functions for Test Functions
///
/// Defines peripheral functions used by other test functions
///
// === Begin Imports ===
// === End Imports ===

// === Root Finding ===
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

// === Finite Differencing ===
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
