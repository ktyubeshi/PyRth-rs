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
    let (time, temperature) = cut_and_shift(input.time, input.value, params)?;
    let t_zero = average_range(&temperature, params.temp_0_avg_range, "temp_0_avg_range")?;
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

fn tmp_to_z(
    temperature: &Array1<f64>,
    t_zero: f64,
    params: &EvaluationParams,
) -> Result<Array1<f64>> {
    let denom = (params.power_step.abs() - params.optical_power) * params.power_scale_factor;
    if denom == 0.0 || !denom.is_finite() {
        return Err(PyrthError::InvalidParameter {
            parameter: "power_step",
            expected: "finite non-zero effective power denominator",
            actual: params.power_step.to_string(),
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
