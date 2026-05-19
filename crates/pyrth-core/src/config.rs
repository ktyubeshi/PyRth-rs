use std::{fmt, str::FromStr};

use serde::{Deserialize, Serialize};

use crate::error::{PyrthError, Result};

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum InputMode {
    T3ster,
    Temperature,
    Voltage,
    Impedance,
}

impl InputMode {
    pub fn from_label(value: &str) -> Result<Self> {
        match normalize_label(value).as_str() {
            "t3ster" => Ok(Self::T3ster),
            "temp" | "temperature" => Ok(Self::Temperature),
            "volt" | "voltage" => Ok(Self::Voltage),
            "impedance" => Ok(Self::Impedance),
            _ => Err(PyrthError::UnknownMode {
                kind: "input mode",
                value: value.to_string(),
            }),
        }
    }
}

impl FromStr for InputMode {
    type Err = PyrthError;

    fn from_str(value: &str) -> Result<Self> {
        Self::from_label(value)
    }
}

impl fmt::Display for InputMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::T3ster => "t3ster",
            Self::Temperature => "temp",
            Self::Voltage => "volt",
            Self::Impedance => "impedance",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum DeconvMode {
    Bayesian,
    Fourier,
    Lasso,
    Adaptive,
}

impl DeconvMode {
    pub fn from_label(value: &str) -> Result<Self> {
        match normalize_label(value).as_str() {
            "bayesian" => Ok(Self::Bayesian),
            "fft" | "fourier" => Ok(Self::Fourier),
            "lasso" => Ok(Self::Lasso),
            "adaptive" => Ok(Self::Adaptive),
            _ => Err(PyrthError::UnknownMode {
                kind: "deconvolution mode",
                value: value.to_string(),
            }),
        }
    }
}

impl FromStr for DeconvMode {
    type Err = PyrthError;

    fn from_str(value: &str) -> Result<Self> {
        Self::from_label(value)
    }
}

impl fmt::Display for DeconvMode {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Bayesian => "bayesian",
            Self::Fourier => "fourier",
            Self::Lasso => "lasso",
            Self::Adaptive => "adaptive",
        })
    }
}

#[derive(Clone, Copy, Debug, Eq, PartialEq, Serialize, Deserialize)]
#[serde(rename_all = "snake_case")]
pub enum StructureMethod {
    Lanczos,
    Sobhy,
    BoorGolub,
    Khatwani,
    PolyLong,
}

impl StructureMethod {
    pub fn from_label(value: &str) -> Result<Self> {
        match normalize_label(value).as_str() {
            "lanczos" => Ok(Self::Lanczos),
            "sobhy" => Ok(Self::Sobhy),
            "boor_golub" | "boorgolub" => Ok(Self::BoorGolub),
            "khatwani" => Ok(Self::Khatwani),
            "polylong" | "poly_long" => Ok(Self::PolyLong),
            _ => Err(PyrthError::UnknownMode {
                kind: "structure method",
                value: value.to_string(),
            }),
        }
    }
}

impl FromStr for StructureMethod {
    type Err = PyrthError;

    fn from_str(value: &str) -> Result<Self> {
        Self::from_label(value)
    }
}

impl fmt::Display for StructureMethod {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Lanczos => "lanczos",
            Self::Sobhy => "sobhy",
            Self::BoorGolub => "boor_golub",
            Self::Khatwani => "khatwani",
            Self::PolyLong => "polylong",
        })
    }
}

#[derive(Clone, Debug, PartialEq, Serialize, Deserialize)]
pub struct EvaluationParams {
    pub input_mode: InputMode,
    pub deconv_mode: DeconvMode,
    pub structure_method: StructureMethod,
    pub precision: usize,
    pub log_time_size: usize,
    pub bay_steps: usize,
    pub pad_factor_pre: f64,
    pub pad_factor_after: f64,
    pub minimum_window_length: f64,
    pub maximum_window_length: f64,
    pub minimum_window_size: usize,
    pub window_increment: f64,
    pub expected_var: f64,
    pub min_index: usize,
    pub timespec_interpolate_factor: f64,
    pub blockwise_sum_width: usize,
    pub calc_struc: bool,
    pub only_make_z: bool,
}

impl Default for EvaluationParams {
    fn default() -> Self {
        Self {
            input_mode: InputMode::Impedance,
            deconv_mode: DeconvMode::Bayesian,
            structure_method: StructureMethod::Sobhy,
            precision: 250,
            log_time_size: 250,
            bay_steps: 1000,
            pad_factor_pre: 0.01,
            pad_factor_after: 0.01,
            minimum_window_length: 0.35,
            maximum_window_length: 3.0,
            minimum_window_size: 70,
            window_increment: 0.1,
            expected_var: 0.09,
            min_index: 3,
            timespec_interpolate_factor: 1.0,
            blockwise_sum_width: 20,
            calc_struc: true,
            only_make_z: false,
        }
    }
}

fn normalize_label(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}
