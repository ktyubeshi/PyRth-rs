use ndarray::Array1;

use crate::{
    config::{EvaluationParams, InputMode},
    data::{ImpedanceData, TransientInput},
    deconvolution::time_spectrum_bayesian,
    derivative::z_fit_deriv,
    error::{PyrthError, Result},
    network::{cauer_from_foster_lanczos, foster_from_time_spectrum},
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
    validate_params(params)?;
    if params.input_mode != InputMode::Impedance {
        return Err(PyrthError::UnsupportedInputMode(
            params.input_mode.to_string(),
        ));
    }
    if !params.only_make_z && input.time.len() < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "data",
            expected: "at least two samples when only_make_z is false",
            actual: input.time.len().to_string(),
        });
    }

    let impedance = make_impedance_data(input)?;
    if params.only_make_z {
        return Ok(EvaluationResult {
            impedance,
            derivative: None,
            time_spectrum: None,
            foster: None,
            cauer: None,
        });
    }

    let derivative = z_fit_deriv(&impedance.impedance, &impedance.log_time, params)?;
    let time_spectrum = time_spectrum_bayesian(&derivative, params);
    let foster = foster_from_time_spectrum(&derivative.log_time_pad, &time_spectrum, 1e-10)?;
    let cauer = if params.calc_struc {
        Some(cauer_from_foster_lanczos(
            &foster.capacitance,
            &foster.resistance,
            params,
        ))
    } else {
        None
    };

    Ok(EvaluationResult {
        impedance,
        derivative: Some(derivative),
        time_spectrum: Some(time_spectrum),
        foster: Some(foster),
        cauer,
    })
}

fn validate_params(params: &EvaluationParams) -> Result<()> {
    if params.precision == 0 {
        return invalid_param("precision", "greater than zero", params.precision);
    }
    if params.log_time_size == 0 {
        return invalid_param("log_time_size", "greater than zero", params.log_time_size);
    }
    if params.minimum_window_size == 0 {
        return invalid_param(
            "minimum_window_size",
            "greater than zero",
            params.minimum_window_size,
        );
    }
    if !params.pad_factor_pre.is_finite() || params.pad_factor_pre < 0.0 {
        return invalid_param(
            "pad_factor_pre",
            "finite and non-negative",
            params.pad_factor_pre,
        );
    }
    if !params.pad_factor_after.is_finite() || params.pad_factor_after < 0.0 {
        return invalid_param(
            "pad_factor_after",
            "finite and non-negative",
            params.pad_factor_after,
        );
    }
    if !params.minimum_window_length.is_finite() || params.minimum_window_length <= 0.0 {
        return invalid_param(
            "minimum_window_length",
            "finite and greater than zero",
            params.minimum_window_length,
        );
    }
    if !params.maximum_window_length.is_finite()
        || params.maximum_window_length < params.minimum_window_length
    {
        return invalid_param(
            "maximum_window_length",
            "finite and at least minimum_window_length",
            params.maximum_window_length,
        );
    }
    if !params.window_increment.is_finite() || params.window_increment <= 0.0 {
        return invalid_param(
            "window_increment",
            "finite and greater than zero",
            params.window_increment,
        );
    }
    if !params.expected_var.is_finite() || params.expected_var < 0.0 {
        return invalid_param(
            "expected_var",
            "finite and non-negative",
            params.expected_var,
        );
    }
    if !params.timespec_interpolate_factor.is_finite() || params.timespec_interpolate_factor < 1.0 {
        return invalid_param(
            "timespec_interpolate_factor",
            "finite and at least 1",
            params.timespec_interpolate_factor,
        );
    }
    if params.blockwise_sum_width == 0 {
        return invalid_param(
            "blockwise_sum_width",
            "greater than zero",
            params.blockwise_sum_width,
        );
    }
    Ok(())
}

fn invalid_param<T: ToString>(
    parameter: &'static str,
    expected: &'static str,
    actual: T,
) -> Result<()> {
    Err(PyrthError::InvalidParameter {
        parameter,
        expected,
        actual: actual.to_string(),
    })
}
