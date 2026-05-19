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
pub mod export;
pub mod network;
pub mod prediction;
pub mod preprocess;
pub mod t3ster;
pub mod theoretical;

pub use config::{DeconvMode, EvaluationParams, FourierFilter, InputMode, StructureMethod};
pub use data::{ImpedanceData, TransientInput};
pub use deconvolution::{
    bayesian_deconvolution, response_matrix, time_spectrum_bayesian, time_spectrum_fourier,
};
pub use derivative::z_fit_deriv;
pub use error::{PyrthError, Result};
pub use evaluation::{evaluate, CauerNetwork, DerivativeResult, EvaluationResult, FosterNetwork};
pub use export::{export_csv, ExportedCsvFiles};
pub use network::{cauer_from_foster_lanczos, foster_from_time_spectrum};
pub use prediction::{predict_temperature, TemperaturePredictionResult};
pub use preprocess::make_impedance_data;
pub use t3ster::{
    parse_t3ster_calibration_text, parse_t3ster_power_step, parse_t3ster_raw_text,
    t3ster_raw_to_temperature_input, T3sterRaw,
};
pub use theoretical::{theoretical_impedance_input, TheoreticalModel};
