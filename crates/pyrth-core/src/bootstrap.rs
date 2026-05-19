use ndarray::Array1;
use rand::{rngs::StdRng, SeedableRng};
use rand_distr::{Distribution, Normal};

use crate::{
    config::EvaluationParams,
    data::TransientInput,
    error::{PyrthError, Result},
    evaluation::evaluate,
    theoretical::TheoreticalModel,
};

#[derive(Clone, Debug, PartialEq)]
pub struct BootstrapResult {
    pub impedance_mean: Array1<f64>,
    pub time_spectrum_mean: Array1<f64>,
    pub successful_repetitions: usize,
}

pub fn bootstrap_from_theoretical(
    model: &TheoreticalModel,
    time_start: f64,
    time_end: f64,
    time_size: usize,
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

    let base = model.to_transient_input(time_start, time_end, time_size)?;
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
    let mut impedance_sum: Option<Array1<f64>> = None;
    let mut time_spectrum_sum: Option<Array1<f64>> = None;
    let mut successful_repetitions = 0usize;

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

        match (&mut impedance_sum, &mut time_spectrum_sum) {
            (Some(impedance_sum), Some(time_spectrum_sum))
                if impedance_sum.len() == result.impedance.impedance.len()
                    && time_spectrum_sum.len() == time_spectrum.len() =>
            {
                *impedance_sum += &result.impedance.impedance;
                *time_spectrum_sum += &time_spectrum;
            }
            (None, None) => {
                impedance_sum = Some(result.impedance.impedance);
                time_spectrum_sum = Some(time_spectrum);
            }
            _ => continue,
        }
        successful_repetitions += 1;
    }

    if successful_repetitions == 0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "repetitions",
            expected: "at least one successful evaluation",
            actual: "0".to_string(),
        });
    }

    let scale = successful_repetitions as f64;
    Ok(BootstrapResult {
        impedance_mean: impedance_sum.expect("successful repetition initializes impedance") / scale,
        time_spectrum_mean: time_spectrum_sum
            .expect("successful repetition initializes time spectrum")
            / scale,
        successful_repetitions,
    })
}
