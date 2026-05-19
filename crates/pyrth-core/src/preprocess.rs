use ndarray::Array1;

use crate::{
    config::{EvaluationParams, InputMode},
    data::{ImpedanceData, TransientInput},
    error::{PyrthError, Result},
};

pub fn make_impedance_data(
    input: TransientInput,
    params: &EvaluationParams,
) -> Result<ImpedanceData> {
    input.validate()?;
    match params.input_mode {
        InputMode::Impedance => from_impedance(input),
        InputMode::Temperature => from_temperature(input, params),
        InputMode::Voltage => from_voltage(input, params),
        InputMode::T3ster => Err(PyrthError::UnsupportedInputMode(
            params.input_mode.to_string(),
        )),
    }
}

fn from_impedance(input: TransientInput) -> Result<ImpedanceData> {
    build_impedance_data(input.time, input.value)
}

fn from_temperature(input: TransientInput, params: &EvaluationParams) -> Result<ImpedanceData> {
    let (time, temperature, t_zero) = if params.extrapolate {
        extrapolate_temperature(input.time, input.value, params)?
    } else {
        let (time, temperature) = cut_and_shift(input.time, input.value, params)?;
        let t_zero = average_range(&temperature, params.temp_0_avg_range, "temp_0_avg_range")?;
        (time, temperature, t_zero)
    };
    let impedance = tmp_to_z(&temperature, t_zero, params)?;
    build_impedance_data(time, impedance)
}

fn from_voltage(input: TransientInput, params: &EvaluationParams) -> Result<ImpedanceData> {
    let calibration = params
        .calibration
        .as_ref()
        .ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "calibration",
            expected: "calibration points for voltage input",
            actual: "None".to_string(),
        })?;
    let temperature = volt_to_temp(&input.value, calibration, params.kfac_fit_deg)?;
    from_temperature(
        TransientInput {
            time: input.time,
            value: temperature,
        },
        params,
    )
}

fn build_impedance_data(time: Array1<f64>, impedance: Array1<f64>) -> Result<ImpedanceData> {
    if !time.iter().all(|time| time.is_finite() && *time > 0.0) {
        return Err(PyrthError::InvalidTimeAxis);
    }
    if time.windows(2).into_iter().any(|pair| pair[0] >= pair[1]) {
        return Err(PyrthError::InvalidTimeAxis);
    }
    let log_time: Array1<f64> = time.mapv(f64::ln);
    Ok(ImpedanceData {
        time,
        impedance,
        log_time,
    })
}

fn cut_and_shift(
    time_raw: Array1<f64>,
    value_raw: Array1<f64>,
    params: &EvaluationParams,
) -> Result<(Array1<f64>, Array1<f64>)> {
    let start = params.data_cut_lower;
    let end = params
        .data_cut_upper
        .unwrap_or(time_raw.len())
        .min(time_raw.len());
    if start >= end {
        return Err(PyrthError::InvalidParameter {
            parameter: "data_cut_lower",
            expected: "less than data_cut_upper and input length",
            actual: start.to_string(),
        });
    }

    let time_zero = if start > 0 { time_raw[start - 1] } else { 0.0 };
    let time = Array1::from_iter((start..end).map(|index| time_raw[index] - time_zero));
    let value = Array1::from_iter((start..end).map(|index| value_raw[index]));

    Ok((time, value))
}

fn average_range(
    values: &Array1<f64>,
    range: (usize, usize),
    parameter: &'static str,
) -> Result<f64> {
    let start = range.0.min(values.len());
    let end = range.1.min(values.len());
    if start >= end {
        return Err(PyrthError::InvalidParameter {
            parameter,
            expected: "non-empty range inside input length",
            actual: format!("{range:?}"),
        });
    }
    Ok(values.iter().skip(start).take(end - start).sum::<f64>() / (end - start) as f64)
}

fn extrapolate_temperature(
    time_raw: Array1<f64>,
    temp_raw: Array1<f64>,
    params: &EvaluationParams,
) -> Result<(Array1<f64>, Array1<f64>, f64)> {
    let lower_fit_limit = params
        .lower_fit_limit
        .ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "lower_fit_limit",
            expected: "Some finite positive time when extrapolate is true",
            actual: "None".to_string(),
        })?;
    let upper_fit_limit = params
        .upper_fit_limit
        .ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "upper_fit_limit",
            expected: "Some finite positive time when extrapolate is true",
            actual: "None".to_string(),
        })?;
    if !lower_fit_limit.is_finite()
        || !upper_fit_limit.is_finite()
        || lower_fit_limit <= 0.0
        || upper_fit_limit <= lower_fit_limit
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "lower_fit_limit",
            expected: "finite positive lower limit below upper_fit_limit",
            actual: format!("{lower_fit_limit}"),
        });
    }

    let lower_fit_index = search_sorted(&time_raw, lower_fit_limit);
    let upper_fit_index = search_sorted(&time_raw, upper_fit_limit);
    if lower_fit_index >= upper_fit_index || upper_fit_index > time_raw.len() {
        return Err(PyrthError::InvalidParameter {
            parameter: "upper_fit_limit",
            expected: "fit range containing at least one input sample",
            actual: format!("{upper_fit_limit}"),
        });
    }

    let total_decades = time_raw[time_raw.len() - 1].log10() - time_raw[0].log10();
    if !total_decades.is_finite() || total_decades <= 0.0 {
        return Err(PyrthError::InvalidTimeAxis);
    }

    let additional_decades = 4.0;
    let extrapolation_start = time_raw[0] / 10.0_f64.powf(additional_decades);
    let extrapolation_decades = time_raw[lower_fit_index].log10() - extrapolation_start.log10();
    let fit_add_extrapolation =
        (time_raw.len() as f64 * (extrapolation_decades / total_decades)) as usize;
    if fit_add_extrapolation == 0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "lower_fit_limit",
            expected: "fit lower limit that creates extrapolated samples",
            actual: format!("{lower_fit_limit}"),
        });
    }

    let (intercept, slope) =
        fit_sqrt_time_line(&time_raw, &temp_raw, lower_fit_index, upper_fit_index)?;
    let time_start_log = extrapolation_start.log10();
    let time_stop_log = time_raw[lower_fit_index].log10();
    let log_step = (time_stop_log - time_start_log) / fit_add_extrapolation as f64;

    let mut time = Vec::with_capacity(fit_add_extrapolation + time_raw.len() - lower_fit_index);
    let mut temperature = Vec::with_capacity(time.capacity());
    for index in 0..fit_add_extrapolation {
        let sample_time = 10.0_f64.powf(time_start_log + log_step * index as f64);
        time.push(sample_time);
        temperature.push(intercept + slope * sample_time.sqrt());
    }
    for index in lower_fit_index..time_raw.len() {
        time.push(time_raw[index]);
        temperature.push(temp_raw[index]);
    }

    Ok((Array1::from(time), Array1::from(temperature), intercept))
}

fn search_sorted(values: &Array1<f64>, needle: f64) -> usize {
    values
        .iter()
        .position(|value| *value >= needle)
        .unwrap_or(values.len())
}

fn fit_sqrt_time_line(
    time: &Array1<f64>,
    temperature: &Array1<f64>,
    start: usize,
    end: usize,
) -> Result<(f64, f64)> {
    let count = end - start;
    let mut sum_x = 0.0;
    let mut sum_y = 0.0;
    let mut sum_xx = 0.0;
    let mut sum_xy = 0.0;

    for index in start..end {
        let x = time[index].sqrt();
        let y = temperature[index];
        sum_x += x;
        sum_y += y;
        sum_xx += x * x;
        sum_xy += x * y;
    }

    let count = count as f64;
    let denom = count * sum_xx - sum_x * sum_x;
    if denom == 0.0 || !denom.is_finite() {
        return Err(PyrthError::InvalidParameter {
            parameter: "lower_fit_limit",
            expected: "fit range with varying time samples",
            actual: "singular".to_string(),
        });
    }

    let slope = (count * sum_xy - sum_x * sum_y) / denom;
    let intercept = (sum_y - slope * sum_x) / count;
    Ok((intercept, slope))
}

fn tmp_to_z(
    temperature: &Array1<f64>,
    t_zero: f64,
    params: &EvaluationParams,
) -> Result<Array1<f64>> {
    let denom = (params.power_step.abs() - params.optical_power) * params.power_scale_factor;
    if !denom.is_finite() || denom <= 0.0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "effective power",
            expected: "positive finite value",
            actual: denom.to_string(),
        });
    }
    Ok(Array1::from_iter(temperature.iter().map(|temp| {
        let z = (t_zero - temp) / denom;
        if params.is_heating {
            -z
        } else {
            z
        }
    })))
}

fn volt_to_temp(
    voltage: &Array1<f64>,
    calibration: &[[f64; 2]],
    degree: usize,
) -> Result<Array1<f64>> {
    if degree > 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "kfac_fit_deg",
            expected: "0, 1, or 2",
            actual: degree.to_string(),
        });
    }
    if calibration.len() <= degree {
        return Err(PyrthError::InvalidParameter {
            parameter: "calibration",
            expected: "more calibration rows than polynomial degree",
            actual: calibration.len().to_string(),
        });
    }

    let coeffs = polyfit_voltage_temperature(calibration, degree)?;
    Ok(Array1::from_iter(
        voltage.iter().map(|voltage| polyval(&coeffs, *voltage)),
    ))
}

fn polyfit_voltage_temperature(calibration: &[[f64; 2]], degree: usize) -> Result<Vec<f64>> {
    let size = degree + 1;
    let mut normal = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];

    for [temperature, voltage] in calibration {
        let powers = (0..size)
            .map(|power| voltage.powi(power as i32))
            .collect::<Vec<_>>();
        for row in 0..size {
            rhs[row] += powers[row] * temperature;
            for col in 0..size {
                normal[row][col] += powers[row] * powers[col];
            }
        }
    }

    solve_linear_system(normal, rhs)
}

fn solve_linear_system(mut matrix: Vec<Vec<f64>>, mut rhs: Vec<f64>) -> Result<Vec<f64>> {
    let len = rhs.len();
    for pivot in 0..len {
        let max_row = (pivot..len)
            .max_by(|left, right| {
                matrix[*left][pivot]
                    .abs()
                    .partial_cmp(&matrix[*right][pivot].abs())
                    .unwrap_or(std::cmp::Ordering::Equal)
            })
            .unwrap();
        if matrix[max_row][pivot] == 0.0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "calibration",
                expected: "full-rank polynomial fit",
                actual: "singular".to_string(),
            });
        }
        matrix.swap(pivot, max_row);
        rhs.swap(pivot, max_row);

        let pivot_value = matrix[pivot][pivot];
        for col in pivot..len {
            matrix[pivot][col] /= pivot_value;
        }
        rhs[pivot] /= pivot_value;

        for row in 0..len {
            if row == pivot {
                continue;
            }
            let factor = matrix[row][pivot];
            for col in pivot..len {
                matrix[row][col] -= factor * matrix[pivot][col];
            }
            rhs[row] -= factor * rhs[pivot];
        }
    }

    Ok(rhs)
}

fn polyval(coeffs: &[f64], value: f64) -> f64 {
    coeffs
        .iter()
        .enumerate()
        .map(|(power, coeff)| coeff * value.powi(power as i32))
        .sum()
}
