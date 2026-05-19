mod cauer;
mod foster;
mod rational;

pub use cauer::cauer_from_foster_lanczos;
pub use foster::foster_from_time_spectrum;
pub use rational::{
    cauer_from_foster_poly_long_f64, foster_impedance_rational_f64,
    normalize_rational_polynomials_f64, FosterRationalF64,
};
