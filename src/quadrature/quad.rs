trait TrapzSum {
    fn next(&mut self) -> f64;
}

trait TrapzInt<T: TrapzSum> {
    fn integrate(&mut self, tol: Option<f64>) -> Result<f64, &'static str>;
}

trait QSimple<T: TrapzSum> {
    fn qsimp(&mut self, tol: Option<f64>) -> Result<f64, &'static str>;
}

pub struct IntegProblem {
    dynamics: fn(f64) -> f64,
    lvl_of_refinement: u32,
    sum: f64,
    lim_low: f64,
    lim_high: f64,
}

impl IntegProblem {
    fn new(func: fn(f64) -> f64, lower: f64, upper: f64) -> Result<IntegProblem, &'static str> {
        if lower >= upper {
            return Err("Invalid Range. Lower and upper bounds must differ and lower must be less than upper");
        }
        Ok(IntegProblem {
            dynamics: func,
            lvl_of_refinement: 0,
            sum: 0.0,
            lim_low: lower,
            lim_high: upper,
        })
    }
}

impl TrapzSum for IntegProblem {
    fn next(&mut self) -> f64 {
        self.lvl_of_refinement += 1;

        if self.lvl_of_refinement == 1 {
            self.sum = 0.5
                * (self.lim_high - self.lim_low)
                * ((self.dynamics)(self.lim_low) + (self.dynamics)(self.lim_high));
        } else {
            let added_points = u32::pow(2, self.lvl_of_refinement - 2);
            let del: f64 = (self.lim_high - self.lim_low) / added_points as f64; // spacing of points to add
            let mut x = self.lim_low + 0.5 * del;
            let mut s: f64 = 0.0;
            for _ in 0..added_points {
                s += (self.dynamics)(x);
                x += del;
            }
            self.sum = 0.5 * (self.sum + (self.lim_high - self.lim_low) * s / added_points as f64);
        }
        self.sum
    }
}

impl TrapzInt<IntegProblem> for IntegProblem {
    fn integrate(&mut self, tol: Option<f64>) -> Result<f64, &'static str> {
        let tol = tol.unwrap_or(1.0e-10);
        const JMAX: i32 = 20; // Max allowed steps is 2^(JMAX - 1)
        let mut s: f64;
        let mut olds = 0.0;
        for iter in 0..JMAX {
            s = self.next();
            if iter > 5 {
                let x = s - olds;
                println!("{} | {}", x, tol * olds);
                if (x.abs() < tol * olds.abs()) || (s == 0.0 && olds == 0.0) {
                    return Ok(s);
                }
            }
            olds = s;
        }
        Err("Maximum number of iterations exceeded")
    }
}

impl QSimple<IntegProblem> for IntegProblem {
    // Q Trap algorithm
    // ONLY works when the function being evaluated has a finite 4th derivative
    //
    fn qsimp(&mut self, tol: Option<f64>) -> Result<f64, &'static str> {
        let tol = tol.unwrap_or(1.0e-10);
        const JMAX: i32 = 20; // Max allowed steps is 2^(JMAX - 1)
        let (mut s, mut st): (f64, f64);
        let (mut ost, mut os) = (0.0, 0.0);

        for iter in 0..JMAX {
            st = self.next();
            s = (4.0 * st - ost) / 3.0;
            if iter > 5 {
                if ((s - os).abs() < tol * os.abs()) || (s == 0.0 && os == 0.0) {
                    return Ok(s);
                }
            }
            os = s;
            ost = st;
        }
        Err("Maximum number of iterations exceeded")
    }
}

pub fn dynamicallyness(x: f64) -> f64 {
    x * 2.0
}

pub fn cubic(x: f64) -> f64 {
    f64::powf(x, 3.0)
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn linear() {
        let mut integ = IntegProblem::new(dynamicallyness, 0.0, 10.0).unwrap();
        let a = integ.next();
        //println!("{:?}", a);
        let mut b = integ.next();
        //println!("{:?}", b);
        for _iter in 1..15 {
            b = integ.next();
            //println!("{:?}", b);
        }
        assert_eq!(1, 1)
    }
    #[test]
    fn cubed() {
        let mut integ = IntegProblem::new(cubic, 0.0, 10.0).unwrap();
        let a = integ.next();
        //println!("{:?}", a);
        let mut b = integ.next();
        //println!("{:?}", b);
        for _iter in 1..15 {
            b = integ.next();
            //println!("{:?}", b);
        }
        assert_eq!(1, 1)
    }
    #[test]
    fn int_cubed() {
        let mut integ = IntegProblem::new(cubic, 0.0, 10.0).unwrap();
        let a = integ.integrate(Option::None);
        println!("{:?}", a);
    }
    #[test]
    fn int_cubed_qsimp() {
        let mut integ = IntegProblem::new(cubic, 0.0, 10.0).unwrap();
        let a = integ.qsimp(Option::None);
        println!("{:?}", a);
    }
}
