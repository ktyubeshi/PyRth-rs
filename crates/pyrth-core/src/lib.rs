//! Numerical core for the PyRth Rust port.
//!
//! The initial surface mirrors the Python pipeline boundaries so each Rust
//! stage can be compared against Python golden fixtures before more algorithms
//! are ported.

pub mod config;
pub mod data;
pub mod deconvolution;
pub mod derivative;
pub mod error;
pub mod evaluation;
pub mod network;
pub mod preprocess;

pub use config::{DeconvMode, EvaluationParams, InputMode, StructureMethod};
pub use data::{ImpedanceData, TransientInput};
pub use deconvolution::{bayesian_deconvolution, response_matrix, time_spectrum_bayesian};
pub use derivative::z_fit_deriv;
pub use error::{PyrthError, Result};
pub use evaluation::{evaluate, CauerNetwork, DerivativeResult, EvaluationResult, FosterNetwork};
pub use network::{cauer_from_foster_lanczos, foster_from_time_spectrum};
pub use preprocess::make_impedance_data;
