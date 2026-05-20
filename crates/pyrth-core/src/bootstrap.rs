use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::{
    config::{EvaluationParams, InputMode},
    data::TransientInput,
    error::{PyrthError, Result},
    evaluation::evaluate,
    foster_step::FosterStepResponseModel,
};

#[derive(Clone, Debug, PartialEq)]
pub struct BootstrapResult {
    pub impedance_mean: Array1<f64>,
    pub impedance_p10: Array1<f64>,
    pub impedance_median: Array1<f64>,
    pub impedance_p90: Array1<f64>,
    pub time_spectrum_mean: Array1<f64>,
    pub time_spectrum_p10: Array1<f64>,
    pub time_spectrum_median: Array1<f64>,
    pub time_spectrum_p90: Array1<f64>,
    pub successful_repetitions: usize,
}

pub fn bootstrap_from_foster_step_response(
    model: &FosterStepResponseModel,
    time_start: f64,
    time_end: f64,
    time_size: usize,
    repetitions: usize,
    noise_std: f64,
    params: &EvaluationParams,
    seed: u64,
) -> Result<BootstrapResult> {
    let base = model.to_transient_input(time_start, time_end, time_size)?;
    bootstrap_from_input(&base, repetitions, noise_std, params, seed)
}

#[deprecated(note = "Use bootstrap_from_foster_step_response for Foster step response data.")]
pub fn bootstrap_from_theoretical(
    model: &FosterStepResponseModel,
    time_start: f64,
    time_end: f64,
    time_size: usize,
    repetitions: usize,
    noise_std: f64,
    params: &EvaluationParams,
    seed: u64,
) -> Result<BootstrapResult> {
    bootstrap_from_foster_step_response(
        model,
        time_start,
        time_end,
        time_size,
        repetitions,
        noise_std,
        params,
        seed,
    )
}

pub fn bootstrap_from_impedance_data(
    input: &TransientInput,
    repetitions: usize,
    noise_std: f64,
    params: &EvaluationParams,
    seed: u64,
) -> Result<BootstrapResult> {
    input.validate()?;
    if params.input_mode != InputMode::Impedance {
        return Err(PyrthError::InvalidParameter {
            parameter: "input_mode",
            expected: "impedance for bootstrap_from_impedance_data",
            actual: params.input_mode.to_string(),
        });
    }

    bootstrap_from_input(input, repetitions, noise_std, params, seed)
}

fn bootstrap_from_input(
    base: &TransientInput,
    repetitions: usize,
    noise_std: f64,
    params: &EvaluationParams,
    seed: u64,
) -> Result<BootstrapResult> {
    if repetitions == 0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "repetitions",
            expected: "greater than zero",
            actual: repetitions.to_string(),
        });
    }
    if !noise_std.is_finite() || noise_std < 0.0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "noise_std",
            expected: "finite and non-negative",
            actual: noise_std.to_string(),
        });
    }

    let normal = if noise_std > 0.0 {
        Some(
            Normal::new(0.0, noise_std).map_err(|err| PyrthError::InvalidParameter {
                parameter: "noise_std",
                expected: "valid normal distribution standard deviation",
                actual: err.to_string(),
            })?,
        )
    } else {
        None
    };
    let mut rng = StdRng::seed_from_u64(seed);
    let mut impedance_samples: Vec<Array1<f64>> = Vec::new();
    let mut time_spectrum_samples: Vec<Array1<f64>> = Vec::new();

    for _ in 0..repetitions {
        let value = match normal.as_ref() {
            Some(normal) => base
                .value
                .iter()
                .map(|value| value + normal.sample(&mut rng))
                .collect::<Array1<_>>(),
            None => base.value.clone(),
        };
        let input = TransientInput::new(base.time.clone(), value)?;
        let Ok(result) = evaluate(input, params) else {
            continue;
        };
        let Some(time_spectrum) = result.time_spectrum else {
            continue;
        };

        if let (Some(first_impedance), Some(first_spectrum)) =
            (impedance_samples.first(), time_spectrum_samples.first())
        {
            if first_impedance.len() != result.impedance.impedance.len()
                || first_spectrum.len() != time_spectrum.len()
            {
                continue;
            }
        }

        impedance_samples.push(result.impedance.impedance);
        time_spectrum_samples.push(time_spectrum);
    }

    if impedance_samples.is_empty() {
        return Err(PyrthError::InvalidParameter {
            parameter: "repetitions",
            expected: "at least one successful evaluation",
            actual: "0".to_string(),
        });
    }

    let impedance_mean = mean_arrays(&impedance_samples);
    let time_spectrum_mean = mean_arrays(&time_spectrum_samples);
    Ok(BootstrapResult {
        impedance_p10: percentile_arrays(&impedance_samples, 0.10),
        impedance_median: percentile_arrays(&impedance_samples, 0.50),
        impedance_p90: percentile_arrays(&impedance_samples, 0.90),
        impedance_mean,
        time_spectrum_p10: percentile_arrays(&time_spectrum_samples, 0.10),
        time_spectrum_median: percentile_arrays(&time_spectrum_samples, 0.50),
        time_spectrum_p90: percentile_arrays(&time_spectrum_samples, 0.90),
        time_spectrum_mean,
        successful_repetitions: impedance_samples.len(),
    })
}

fn mean_arrays(samples: &[Array1<f64>]) -> Array1<f64> {
    let mut sum = Array1::zeros(samples[0].len());
    for sample in samples {
        sum += sample;
    }
    sum / samples.len() as f64
}

fn percentile_arrays(samples: &[Array1<f64>], quantile: f64) -> Array1<f64> {
    Array1::from_iter((0..samples[0].len()).map(|index| {
        let mut values = samples
            .iter()
            .map(|sample| sample[index])
            .collect::<Vec<_>>();
        values.sort_by(|left, right| left.total_cmp(right));
        interpolate_quantile(&values, quantile)
    }))
}

fn interpolate_quantile(values: &[f64], quantile: f64) -> f64 {
    if values.len() == 1 {
        return values[0];
    }
    let position = quantile.clamp(0.0, 1.0) * (values.len() - 1) as f64;
    let lower = position.floor() as usize;
    let upper = position.ceil() as usize;
    if lower == upper {
        values[lower]
    } else {
        let fraction = position - lower as f64;
        values[lower] + fraction * (values[upper] - values[lower])
    }
}
