use thiserror::Error;

pub type Result<T> = std::result::Result<T, PyrthError>;

#[derive(Debug, Error)]
pub enum PyrthError {
    #[error("input data must contain at least one sample")]
    EmptyInput,

    #[error("time and value arrays must have the same length: {time_len} != {value_len}")]
    LengthMismatch { time_len: usize, value_len: usize },

    #[error("time values must be finite, positive, and strictly increasing")]
    InvalidTimeAxis,

    #[error("input values must be finite")]
    InvalidValues,

    #[error("time constant spectrum is empty after filtering")]
    EmptySpectrum,

    #[error("unsupported input mode for this stage: {0}")]
    UnsupportedInputMode(String),

    #[error("unknown {kind} label: {value}")]
    UnknownMode { kind: &'static str, value: String },
}
