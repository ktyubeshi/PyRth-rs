use ndarray::Array1;

use crate::{
    data::TransientInput,
    error::{PyrthError, Result},
    evaluation::{evaluate, EvaluationResult},
    optimization::{OptimizationResult, RcParameters},
    EvaluationParams,
};

#[derive(Clone, Debug, PartialEq)]
pub struct TemperaturePredictionResult {
    pub base: EvaluationResult,
    pub lin_time: Array1<f64>,
    pub predicted_temperature: Array1<f64>,
    pub power_function_int: Array1<f64>,
    pub impulse_response_int: Array1<f64>,
}

#[derive(Clone, Debug, PartialEq)]
pub struct TemperaturePredictionCoreResult {
    pub lin_time: Array1<f64>,
    pub predicted_temperature: Array1<f64>,
    pub power_function_int: Array1<f64>,
    pub impulse_response_int: Array1<f64>,
}

pub fn predict_temperature(
    input: TransientInput,
    power_data: TransientInput,
    params: &EvaluationParams,
    lin_sampling_period: f64,
) -> Result<TemperaturePredictionResult> {
    if !lin_sampling_period.is_finite() || lin_sampling_period <= 0.0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "lin_sampling_period",
            expected: "finite and greater than zero",
            actual: lin_sampling_period.to_string(),
        });
    }
    validate_power_data(&power_data)?;

    let base = evaluate(input, params)?;
    let derivative = base
        .derivative
        .as_ref()
        .ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "only_make_z",
            expected: "false for temperature prediction",
            actual: params.only_make_z.to_string(),
        })?;
    let reference_time = derivative.log_time_pad.mapv(f64::exp);
    let reference_impulse = derivative.imp_deriv_interp.clone();
    if reference_time.is_empty() || reference_impulse.is_empty() {
        return Err(PyrthError::EmptySpectrum);
    }

    let predicted = predict_temperature_from_impulse_response(
        &power_data,
        &reference_time,
        &reference_impulse,
        lin_sampling_period,
    )?;

    Ok(TemperaturePredictionResult {
        base,
        lin_time: predicted.lin_time,
        predicted_temperature: predicted.predicted_temperature,
        power_function_int: predicted.power_function_int,
        impulse_response_int: predicted.impulse_response_int,
    })
}

pub fn predict_temperature_from_rc_parameters(
    power_data: &TransientInput,
    parameters: &RcParameters,
    reference_time: &Array1<f64>,
    lin_sampling_period: f64,
) -> Result<TemperaturePredictionCoreResult> {
    let reference_impulse = foster_impulse_response_on(parameters, reference_time)?;
    predict_temperature_from_impulse_response(
        power_data,
        reference_time,
        &reference_impulse,
        lin_sampling_period,
    )
}

pub fn predict_temperature_from_optimization_result(
    power_data: &TransientInput,
    result: &OptimizationResult,
    reference_time: &Array1<f64>,
    lin_sampling_period: f64,
) -> Result<TemperaturePredictionCoreResult> {
    predict_temperature_from_rc_parameters(
        power_data,
        &result.parameters,
        reference_time,
        lin_sampling_period,
    )
}

pub fn predict_temperature_from_impulse_response(
    power_data: &TransientInput,
    reference_time: &Array1<f64>,
    reference_impulse: &Array1<f64>,
    lin_sampling_period: f64,
) -> Result<TemperaturePredictionCoreResult> {
    if !lin_sampling_period.is_finite() || lin_sampling_period <= 0.0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "lin_sampling_period",
            expected: "finite and greater than zero",
            actual: lin_sampling_period.to_string(),
        });
    }
    validate_power_data(power_data)?;
    validate_reference_impulse(reference_time, reference_impulse)?;

    let power_t_end = power_data.time[power_data.time.len() - 1];
    let reference_t_end = reference_time[reference_time.len() - 1];
    let t_min = -(power_t_end + reference_t_end);
    let t_max = power_t_end + reference_t_end;
    let lin_count = ((t_max - t_min) / lin_sampling_period).floor() as usize;
    if lin_count < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "lin_sampling_period",
            expected: "small enough to create at least two samples",
            actual: lin_sampling_period.to_string(),
        });
    }

    let dt = (t_max - t_min) / (lin_count - 1) as f64;
    let lin_time_full = Array1::from_iter((0..lin_count).map(|index| t_min + dt * index as f64));
    let power_full = interpolate_left_zero(&lin_time_full, &power_data.time, &power_data.value);
    let impulse_full = interpolate_zero_edges(&lin_time_full, &reference_time, &reference_impulse);
    let predicted_full = convolve_same(&power_full, &impulse_full).mapv(|value| value * dt);

    let start = lin_count / 2 - 1;
    Ok(TemperaturePredictionCoreResult {
        lin_time: slice_from(&lin_time_full, start),
        predicted_temperature: slice_from(&predicted_full, start),
        power_function_int: slice_from(&power_full, start),
        impulse_response_int: impulse_full,
    })
}

pub fn foster_impulse_response_on(
    parameters: &RcParameters,
    time: &Array1<f64>,
) -> Result<Array1<f64>> {
    parameters.validate()?;
    validate_reference_time(time)?;

    Ok(Array1::from_iter(time.iter().copied().map(|sample_time| {
        parameters
            .resistance
            .iter()
            .zip(parameters.capacitance.iter())
            .map(|(resistance, capacitance)| {
                let tau = resistance * capacitance;
                resistance / tau * (-sample_time / tau).exp()
            })
            .sum()
    })))
}

fn interpolate_left_zero(x: &Array1<f64>, xp: &Array1<f64>, fp: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(x.iter().map(|value| {
        if *value < xp[0] {
            0.0
        } else if *value >= xp[xp.len() - 1] {
            fp[fp.len() - 1]
        } else {
            interpolate_value(*value, xp, fp)
        }
    }))
}

fn validate_power_data(power_data: &TransientInput) -> Result<()> {
    if power_data.time.is_empty() {
        return Err(PyrthError::EmptyInput);
    }
    if power_data.time.len() != power_data.value.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: power_data.time.len(),
            value_len: power_data.value.len(),
        });
    }
    if !power_data.value.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidValues);
    }
    if !power_data
        .time
        .iter()
        .all(|time| time.is_finite() && *time >= 0.0)
    {
        return Err(PyrthError::InvalidTimeAxis);
    }
    if power_data
        .time
        .windows(2)
        .into_iter()
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(PyrthError::InvalidTimeAxis);
    }
    Ok(())
}

fn validate_reference_impulse(
    reference_time: &Array1<f64>,
    reference_impulse: &Array1<f64>,
) -> Result<()> {
    validate_reference_time(reference_time)?;
    if reference_time.len() != reference_impulse.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: reference_time.len(),
            value_len: reference_impulse.len(),
        });
    }
    if !reference_impulse.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidValues);
    }
    Ok(())
}

fn validate_reference_time(reference_time: &Array1<f64>) -> Result<()> {
    if reference_time.is_empty() {
        return Err(PyrthError::EmptySpectrum);
    }
    if !reference_time
        .iter()
        .all(|time| time.is_finite() && *time >= 0.0)
    {
        return Err(PyrthError::InvalidTimeAxis);
    }
    if reference_time
        .windows(2)
        .into_iter()
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(PyrthError::InvalidTimeAxis);
    }
    Ok(())
}

fn interpolate_zero_edges(x: &Array1<f64>, xp: &Array1<f64>, fp: &Array1<f64>) -> Array1<f64> {
    Array1::from_iter(x.iter().map(|value| {
        if *value < xp[0] || *value > xp[xp.len() - 1] {
            0.0
        } else {
            interpolate_value(*value, xp, fp)
        }
    }))
}

fn interpolate_value(value: f64, xp: &Array1<f64>, fp: &Array1<f64>) -> f64 {
    let upper = xp
        .iter()
        .position(|sample| *sample >= value)
        .unwrap_or(xp.len() - 1);
    if upper == 0 {
        return fp[0];
    }
    let lower = upper - 1;
    let span = xp[upper] - xp[lower];
    if span == 0.0 {
        fp[lower]
    } else {
        let fraction = (value - xp[lower]) / span;
        fp[lower] + fraction * (fp[upper] - fp[lower])
    }
}

fn convolve_same(left: &Array1<f64>, right: &Array1<f64>) -> Array1<f64> {
    let full_len = left.len() + right.len() - 1;
    let mut full = vec![0.0; full_len];
    for (left_index, left_value) in left.iter().copied().enumerate() {
        for (right_index, right_value) in right.iter().copied().enumerate() {
            full[left_index + right_index] += left_value * right_value;
        }
    }

    let start = (right.len() - 1) / 2;
    Array1::from_iter((0..left.len()).map(|index| full[start + index]))
}

fn slice_from(values: &Array1<f64>, start: usize) -> Array1<f64> {
    Array1::from_iter(values.iter().skip(start).copied())
}
