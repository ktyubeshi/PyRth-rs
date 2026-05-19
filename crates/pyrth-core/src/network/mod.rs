mod cauer;
mod foster;
mod rational;

pub use cauer::cauer_from_foster_lanczos;
pub use foster::foster_from_time_spectrum;
#[cfg(feature = "mpfr")]
pub use rational::{
    boor_golub_cauer_mpfr_raw, cauer_from_foster_boor_golub_mpfr, cauer_from_foster_khatwani_mpfr,
    cauer_from_foster_poly_long_mpfr, cauer_from_foster_sobhy_mpfr,
};
pub use rational::{
    cauer_from_foster_poly_long_f64, foster_impedance_rational_f64,
    normalize_rational_polynomials_f64, FosterRationalF64,
};
