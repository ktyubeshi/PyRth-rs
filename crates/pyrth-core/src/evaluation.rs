use ndarray::Array1;

use crate::{
    config::{EvaluationParams, InputMode},
    data::{ImpedanceData, TransientInput},
    deconvolution::time_spectrum_bayesian,
    derivative::z_fit_deriv,
    error::{PyrthError, Result},
    network::foster_from_time_spectrum,
    preprocess::make_impedance_data,
};

#[derive(Clone, Debug, PartialEq)]
pub struct DerivativeResult {
    pub imp_smooth: Array1<f64>,
    pub imp_deriv_interp: Array1<f64>,
    pub log_time_interp: Array1<f64>,
    pub imp_smooth_full: Array1<f64>,
    pub log_time_pad: Array1<f64>,
    pub log_time_delta: f64,
}

#[derive(Clone, Debug, PartialEq)]
pub struct FosterNetwork {
    pub resistance: Array1<f64>,
    pub capacitance: Array1<f64>,
    pub tau: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct CauerNetwork {
    pub resistance: Array1<f64>,
    pub capacitance: Array1<f64>,
    pub cumulative_resistance: Array1<f64>,
    pub cumulative_capacitance: Array1<f64>,
    pub differential_structure: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct EvaluationResult {
    pub impedance: ImpedanceData,
    pub derivative: Option<DerivativeResult>,
    pub time_spectrum: Option<Array1<f64>>,
    pub foster: Option<FosterNetwork>,
    pub cauer: Option<CauerNetwork>,
}

pub fn evaluate(input: TransientInput, params: &EvaluationParams) -> Result<EvaluationResult> {
    if params.input_mode != InputMode::Impedance {
        return Err(PyrthError::UnsupportedInputMode(
            params.input_mode.to_string(),
        ));
    }

    let impedance = make_impedance_data(input)?;
    let derivative = z_fit_deriv(&impedance.impedance, &impedance.log_time, params)?;
    let time_spectrum = time_spectrum_bayesian(&derivative, params);
    let foster = foster_from_time_spectrum(&derivative.log_time_pad, &time_spectrum, 1e-10)?;

    Ok(EvaluationResult {
        impedance,
        derivative: Some(derivative),
        time_spectrum: Some(time_spectrum),
        foster: Some(foster),
        cauer: None,
    })
}
