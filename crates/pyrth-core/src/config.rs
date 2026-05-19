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
pub enum FourierFilter {
    Hann,
    Rectangular,
    Gauss,
    Fermi,
    Nuttall,
    BlackmanNuttall,
    BlackmanHarris,
}

impl FourierFilter {
    pub fn from_label(value: &str) -> Result<Self> {
        match normalize_label(value).as_str() {
            "hann" => Ok(Self::Hann),
            "rectangular" => Ok(Self::Rectangular),
            "gauss" | "gaussian" => Ok(Self::Gauss),
            "fermi" => Ok(Self::Fermi),
            "nuttall" => Ok(Self::Nuttall),
            "blackman_nuttall" | "blackmannuttall" => Ok(Self::BlackmanNuttall),
            "blackman_harris" | "blackmanharris" => Ok(Self::BlackmanHarris),
            _ => Err(PyrthError::UnknownMode {
                kind: "Fourier filter",
                value: value.to_string(),
            }),
        }
    }
}

impl Default for FourierFilter {
    fn default() -> Self {
        Self::Hann
    }
}

impl FromStr for FourierFilter {
    type Err = PyrthError;

    fn from_str(value: &str) -> Result<Self> {
        Self::from_label(value)
    }
}

impl fmt::Display for FourierFilter {
    fn fmt(&self, f: &mut fmt::Formatter<'_>) -> fmt::Result {
        f.write_str(match self {
            Self::Hann => "hann",
            Self::Rectangular => "rectangular",
            Self::Gauss => "gauss",
            Self::Fermi => "fermi",
            Self::Nuttall => "nuttall",
            Self::BlackmanNuttall => "blackman_nuttall",
            Self::BlackmanHarris => "blackman_harris",
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
    #[serde(default)]
    pub filter_name: FourierFilter,
    #[serde(default = "default_filter_range")]
    pub filter_range: f64,
    #[serde(default)]
    pub filter_parameter: f64,
    pub bay_steps: usize,
    pub pad_factor_pre: f64,
    pub pad_factor_after: f64,
    pub minimum_window_length: f64,
    pub maximum_window_length: f64,
    pub minimum_window_size: usize,
    pub window_increment: f64,
    pub expected_var: f64,
    #[serde(default = "default_lasso_alpha")]
    pub lasso_alpha: f64,
    #[serde(default = "default_lasso_max_iter")]
    pub lasso_max_iter: usize,
    #[serde(default = "default_lasso_tol")]
    pub lasso_tol: f64,
    pub min_index: usize,
    pub timespec_interpolate_factor: f64,
    pub blockwise_sum_width: usize,
    pub calc_struc: bool,
    pub only_make_z: bool,
    pub power_step: f64,
    pub power_scale_factor: f64,
    pub optical_power: f64,
    pub is_heating: bool,
    #[serde(default)]
    pub extrapolate: bool,
    #[serde(default)]
    pub lower_fit_limit: Option<f64>,
    #[serde(default)]
    pub upper_fit_limit: Option<f64>,
    pub data_cut_lower: usize,
    pub data_cut_upper: Option<usize>,
    pub temp_0_avg_range: (usize, usize),
    pub kfac_fit_deg: usize,
    pub calibration: Option<Vec<[f64; 2]>>,
}

impl Default for EvaluationParams {
    fn default() -> Self {
        Self {
            input_mode: InputMode::Impedance,
            deconv_mode: DeconvMode::Bayesian,
            structure_method: StructureMethod::Sobhy,
            precision: 250,
            log_time_size: 250,
            filter_name: FourierFilter::Hann,
            filter_range: default_filter_range(),
            filter_parameter: 0.0,
            bay_steps: 1000,
            pad_factor_pre: 0.01,
            pad_factor_after: 0.01,
            minimum_window_length: 0.35,
            maximum_window_length: 3.0,
            minimum_window_size: 70,
            window_increment: 0.1,
            expected_var: 0.09,
            lasso_alpha: default_lasso_alpha(),
            lasso_max_iter: default_lasso_max_iter(),
            lasso_tol: default_lasso_tol(),
            min_index: 3,
            timespec_interpolate_factor: 1.0,
            blockwise_sum_width: 20,
            calc_struc: true,
            only_make_z: false,
            power_step: 1.0,
            power_scale_factor: 1.0,
            optical_power: 0.0,
            is_heating: false,
            extrapolate: false,
            lower_fit_limit: None,
            upper_fit_limit: None,
            data_cut_lower: 0,
            data_cut_upper: None,
            temp_0_avg_range: (0, 1),
            kfac_fit_deg: 2,
            calibration: None,
        }
    }
}

fn normalize_label(value: &str) -> String {
    value.trim().to_ascii_lowercase().replace('-', "_")
}

fn default_filter_range() -> f64 {
    0.60
}

fn default_lasso_alpha() -> f64 {
    1e-4
}

fn default_lasso_max_iter() -> usize {
    10000
}

fn default_lasso_tol() -> f64 {
    1e-4
}
