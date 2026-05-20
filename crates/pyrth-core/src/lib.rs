//! Numerical core for the PyRth Rust port.
//!
//! The initial surface mirrors the Python pipeline boundaries so each Rust
//! stage can be compared against Python golden fixtures before more algorithms
//! are ported.

pub mod bootstrap;
pub mod comparison;
pub mod config;
pub mod data;
pub mod deconvolution;
pub mod derivative;
pub mod error;
pub mod evaluation;
pub mod export;
pub mod foster_step;
pub mod network;
pub mod optimization;
pub mod prediction;
pub mod preprocess;
pub mod t3ster;
pub mod theoretical;

pub use bootstrap::{
    bootstrap_from_foster_step_response, bootstrap_from_impedance_data, BootstrapResult,
};
pub use comparison::{compare_evaluations, ComparisonResult};
pub use config::{DeconvMode, EvaluationParams, FourierFilter, InputMode, StructureMethod};
pub use data::{ImpedanceData, TransientInput};
pub use deconvolution::{
    bayesian_deconvolution, response_matrix, time_spectrum_bayesian, time_spectrum_fourier,
};
pub use derivative::z_fit_deriv;
pub use error::{PyrthError, Result};
pub use evaluation::{evaluate, CauerNetwork, DerivativeResult, EvaluationResult, FosterNetwork};
pub use export::{export_csv, export_svg_figures, ExportedCsvFiles, ExportedFigureFiles};
pub use foster_step::{foster_step_response_input, FosterStepResponseModel};
pub use network::{cauer_from_foster_lanczos, foster_from_time_spectrum};
pub use optimization::{
    impedance_residual_norm, optimize_rc_parameters, relative_l2_norm, OptimizationConfig,
    OptimizationResult, RcParameterBounds, RcParameters,
};
pub use prediction::{
    foster_impulse_response_on, predict_temperature, predict_temperature_from_impulse_response,
    predict_temperature_from_optimization_result, predict_temperature_from_rc_parameters,
    TemperaturePredictionCoreResult, TemperaturePredictionResult,
};
pub use preprocess::make_impedance_data;
pub use t3ster::{
    parse_t3ster_calibration_text, parse_t3ster_power_step, parse_t3ster_raw_text,
    t3ster_raw_to_temperature_input, T3sterRaw,
};
pub use theoretical::{
    structure_params_to_func, structure_to_time_const, theoretical_module, time_const_to_impedance,
    TheoreticalStructureResult,
};

#[deprecated(
    note = "This is a Foster lumped RC step response, not Python theoretical_module parity. Use foster_step_response_input instead."
)]
pub use foster_step::foster_step_response_input as theoretical_impedance_input;

#[deprecated(
    note = "Use FosterStepResponseModel for this lumped Foster model, or theoretical_module for Python parity."
)]
pub use foster_step::FosterStepResponseModel as TheoreticalModel;
