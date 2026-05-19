//! Numerical core for the PyRth Rust port.
//!
//! The initial surface mirrors the Python pipeline boundaries so each Rust
//! stage can be compared against Python golden fixtures before more algorithms
//! are ported.

pub mod config;
pub mod data;
pub mod error;
pub mod evaluation;
pub mod preprocess;

pub use config::{DeconvMode, EvaluationParams, InputMode, StructureMethod};
pub use data::{ImpedanceData, TransientInput};
pub use error::{PyrthError, Result};
pub use evaluation::{evaluate, CauerNetwork, DerivativeResult, EvaluationResult, FosterNetwork};
pub use preprocess::make_impedance_data;
