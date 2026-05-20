use ndarray::Array1;
use num_complex::Complex64;

use crate::error::{PyrthError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct TheoreticalStructureResult {
    pub log_time: Array1<f64>,
    pub cumulative_resistance: Array1<f64>,
    pub cumulative_capacitance: Array1<f64>,
    pub differential_structure: Array1<f64>,
    pub time_const_spectrum: Array1<f64>,
    pub impedance_derivative: Array1<f64>,
    pub impedance: Array1<f64>,
}

pub fn theoretical_module(
    resistances: &[f64],
    capacitances: &[f64],
    time_start: f64,
    time_end: f64,
    time_size: usize,
    delta: f64,
) -> Result<TheoreticalStructureResult> {
    validate_theoretical_inputs(
        resistances,
        capacitances,
        time_start,
        time_end,
        time_size,
        delta,
    )?;

    let log_start = time_start.ln();
    let log_end = time_end.ln();
    let log_step = (log_end - log_start) / (time_size - 1) as f64;
    let log_time = Array1::from_iter((0..time_size).map(|index| {
        if index == time_size - 1 {
            log_end
        } else {
            log_start + log_step * index as f64
        }
    }));

    let resistances = Array1::from(resistances.to_vec());
    let capacitances = Array1::from(capacitances.to_vec());
    let (cumulative_resistance, cumulative_capacitance) =
        structure_params_to_func(1000, &resistances, &capacitances)?;
    let cumulative_resistance = cumulative_resistance.slice(ndarray::s![1..]).to_owned();
    let cumulative_capacitance = cumulative_capacitance.slice(ndarray::s![1..]).to_owned();
    let differential_structure =
        differential_structure(&cumulative_resistance, &cumulative_capacitance);
    let time_const_spectrum =
        structure_to_time_const(&log_time, delta, &resistances, &capacitances)?;
    let (impedance_derivative, impedance) =
        time_const_to_impedance(&log_time, &time_const_spectrum)?;

    Ok(TheoreticalStructureResult {
        log_time,
        cumulative_resistance,
        cumulative_capacitance,
        differential_structure,
        time_const_spectrum,
        impedance_derivative,
        impedance,
    })
}

pub fn structure_params_to_func(
    number: usize,
    resistances: &Array1<f64>,
    capacitances: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    validate_rc_arrays(resistances, capacitances)?;
    if number < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "number",
            expected: "at least 2",
            actual: number.to_string(),
        });
    }

    let mut sum_res = Vec::with_capacity(resistances.len() + 1);
    let mut sum_cap = Vec::with_capacity(capacitances.len() + 1);
    sum_res.push(0.0);
    sum_cap.push(0.0);
    for (resistance, capacitance) in resistances.iter().zip(capacitances.iter()) {
        sum_res.push(sum_res.last().copied().unwrap() + resistance);
        sum_cap.push(sum_cap.last().copied().unwrap() + capacitance);
    }

    let total_resistance = *sum_res.last().unwrap();
    let step = total_resistance / (number - 1) as f64;
    let mut sum_res_int = (0..number)
        .map(|index| {
            if index == number - 1 {
                total_resistance
            } else {
                step * index as f64
            }
        })
        .collect::<Vec<_>>();

    for mid_v in &sum_res[1..sum_res.len() - 1] {
        let insert_index = sum_res_int.partition_point(|value| value < mid_v);
        sum_res_int.insert(insert_index, *mid_v);
    }

    let sum_cap_int = sum_res_int
        .iter()
        .map(|resistance| interp(*resistance, &sum_res, &sum_cap))
        .collect::<Vec<_>>();

    Ok((Array1::from(sum_res_int), Array1::from(sum_cap_int)))
}

pub fn structure_to_time_const(
    log_time: &Array1<f64>,
    delta: f64,
    resistances: &Array1<f64>,
    capacitances: &Array1<f64>,
) -> Result<Array1<f64>> {
    validate_log_time(log_time)?;
    validate_rc_arrays(resistances, capacitances)?;
    if !delta.is_finite() {
        return Err(PyrthError::InvalidParameter {
            parameter: "delta",
            expected: "finite",
            actual: delta.to_string(),
        });
    }

    let rotation = -Complex64::new(delta.cos(), delta.sin());
    let log_step = log_time[1] - log_time[0];
    let mut spectrum = Vec::with_capacity(log_time.len());

    for sample_log_time in log_time {
        let complex_time = rotation * (-sample_log_time).exp();
        let mut last_z = Complex64::new(0.0, 0.0);
        let mut z_result = last_z;
        for (resistance, capacitance) in resistances.iter().zip(capacitances.iter()).rev() {
            let gamma_l = (*resistance * *capacitance * complex_time).sqrt();
            let z_null = (*resistance / (*capacitance * complex_time)).sqrt();
            let tanh_gamma_l = gamma_l.tanh();
            let denominator = last_z * tanh_gamma_l + z_null;
            if denominator.norm() == 0.0 || !denominator.norm().is_finite() {
                return Err(PyrthError::InvalidParameter {
                    parameter: "structure_to_time_const",
                    expected: "non-zero finite complex denominator",
                    actual: "zero or non-finite".to_string(),
                });
            }
            z_result = z_null * (last_z + tanh_gamma_l * z_null) / denominator;
            last_z = z_result;
        }
        spectrum.push(z_result.im / std::f64::consts::PI * log_step);
    }

    Ok(Array1::from(spectrum))
}

pub fn time_const_to_impedance(
    log_time: &Array1<f64>,
    time_const: &Array1<f64>,
) -> Result<(Array1<f64>, Array1<f64>)> {
    validate_log_time(log_time)?;
    if time_const.len() != log_time.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: log_time.len(),
            value_len: time_const.len(),
        });
    }
    if !time_const.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidParameter {
            parameter: "time_const",
            expected: "finite values",
            actual: format!("{time_const:?}"),
        });
    }

    let delta_t = log_time[1] - log_time[0];
    let mut log_time_weight = Vec::new();
    let mut z = -7.0;
    while z < 7.0 + delta_t {
        log_time_weight.push(z);
        z += delta_t;
    }
    if log_time_weight.is_empty() {
        return Err(PyrthError::EmptyInput);
    }
    let weight = log_time_weight
        .iter()
        .map(|value| (value - value.exp()).exp())
        .collect::<Vec<_>>();
    let start = weight
        .iter()
        .enumerate()
        .max_by(|(_, left), (_, right)| left.total_cmp(right))
        .map(|(index, _)| index)
        .ok_or(PyrthError::EmptyInput)?;

    let long_len = time_const.len() + weight.len() - 1;
    let mut imp_deriv_long = vec![0.0; long_len];
    for (i, time_const_value) in time_const.iter().enumerate() {
        for (j, weight_value) in weight.iter().enumerate() {
            imp_deriv_long[i + j] += time_const_value * weight_value;
        }
    }

    let fin = start + log_time.len();
    if fin > imp_deriv_long.len() {
        return Err(PyrthError::InvalidParameter {
            parameter: "time_const_to_impedance",
            expected: "convolution slice covering log_time",
            actual: format!("slice {start}..{fin}, len {}", imp_deriv_long.len()),
        });
    }
    let imp_deriv = Array1::from(imp_deriv_long[start..fin].to_vec());
    let mut impedance = Array1::zeros(log_time.len());
    for index in 1..log_time.len() {
        let dx = log_time[index] - log_time[index - 1];
        impedance[index] =
            impedance[index - 1] + 0.5 * dx * (imp_deriv[index] + imp_deriv[index - 1]);
    }

    Ok((imp_deriv, impedance))
}

fn validate_theoretical_inputs(
    resistances: &[f64],
    capacitances: &[f64],
    time_start: f64,
    time_end: f64,
    time_size: usize,
    delta: f64,
) -> Result<()> {
    validate_rc_arrays(
        &Array1::from(resistances.to_vec()),
        &Array1::from(capacitances.to_vec()),
    )?;
    if !(time_start.is_finite()
        && time_end.is_finite()
        && time_start > 0.0
        && time_end > time_start)
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "theo_time",
            expected: "finite, positive, and increasing",
            actual: format!("[{time_start}, {time_end}]"),
        });
    }
    if time_size < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "theo_time_size",
            expected: "at least 2",
            actual: time_size.to_string(),
        });
    }
    if !delta.is_finite() {
        return Err(PyrthError::InvalidParameter {
            parameter: "theo_delta",
            expected: "finite",
            actual: delta.to_string(),
        });
    }
    Ok(())
}

fn validate_rc_arrays(resistances: &Array1<f64>, capacitances: &Array1<f64>) -> Result<()> {
    if resistances.is_empty() {
        return Err(PyrthError::EmptyInput);
    }
    if resistances.len() != capacitances.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: resistances.len(),
            value_len: capacitances.len(),
        });
    }
    if !resistances
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "resistances",
            expected: "finite and positive",
            actual: format!("{resistances:?}"),
        });
    }
    if !capacitances
        .iter()
        .all(|value| value.is_finite() && *value > 0.0)
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "capacitances",
            expected: "finite and positive",
            actual: format!("{capacitances:?}"),
        });
    }
    Ok(())
}

fn validate_log_time(log_time: &Array1<f64>) -> Result<()> {
    if log_time.len() < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "log_time",
            expected: "at least 2 samples",
            actual: log_time.len().to_string(),
        });
    }
    if !log_time.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidParameter {
            parameter: "log_time",
            expected: "finite values",
            actual: format!("{log_time:?}"),
        });
    }
    if log_time
        .windows(2)
        .into_iter()
        .any(|pair| pair[0] >= pair[1])
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "log_time",
            expected: "strictly increasing values",
            actual: format!("{log_time:?}"),
        });
    }
    Ok(())
}

fn differential_structure(
    cumulative_resistance: &Array1<f64>,
    cumulative_capacitance: &Array1<f64>,
) -> Array1<f64> {
    Array1::from_iter(
        (0..cumulative_resistance.len().saturating_sub(1)).map(|index| {
            let denom = cumulative_resistance[index] - cumulative_resistance[index + 1];
            if denom == 0.0 {
                0.0
            } else {
                (cumulative_capacitance[index] - cumulative_capacitance[index + 1]) / denom
            }
        }),
    )
}

fn interp(x: f64, xp: &[f64], fp: &[f64]) -> f64 {
    if x <= xp[0] {
        return fp[0];
    }
    let last = xp.len() - 1;
    if x >= xp[last] {
        return fp[last];
    }

    let upper = xp.partition_point(|value| *value <= x);
    let lower = upper - 1;
    let fraction = (x - xp[lower]) / (xp[upper] - xp[lower]);
    fp[lower] + fraction * (fp[upper] - fp[lower])
}
