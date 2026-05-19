use ndarray::Array1;

use crate::{
    data::TransientInput,
    error::{PyrthError, Result},
};

#[derive(Clone, Debug, PartialEq)]
pub struct T3sterRaw {
    pub time_microseconds: Array1<f64>,
    pub adc_count: Array1<f64>,
    pub lsb: f64,
    pub uref: f64,
}

pub fn parse_t3ster_raw_text(text: &str) -> Result<T3sterRaw> {
    let header_lines = text
        .lines()
        .filter_map(|line| line.trim().strip_prefix('#').map(str::trim))
        .collect::<Vec<_>>();
    if header_lines.len() <= 8 {
        return Err(PyrthError::InvalidParameter {
            parameter: "t3ster raw",
            expected: "legacy T3Ster header with LSB and reference voltage rows",
            actual: format!("{} header rows", header_lines.len()),
        });
    }

    let lsb = header_lines[6]
        .parse::<f64>()
        .map_err(|err| PyrthError::InvalidParameter {
            parameter: "t3ster raw",
            expected: "numeric LSB header value",
            actual: err.to_string(),
        })?;
    let uref = header_lines[8]
        .parse::<f64>()
        .map_err(|err| PyrthError::InvalidParameter {
            parameter: "t3ster raw",
            expected: "numeric reference voltage header value",
            actual: err.to_string(),
        })?;
    if !lsb.is_finite() || !uref.is_finite() {
        return Err(PyrthError::InvalidParameter {
            parameter: "t3ster raw",
            expected: "finite LSB and reference voltage",
            actual: format!("lsb={lsb}, uref={uref}"),
        });
    }

    let mut time = Vec::new();
    let mut adc = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed
            .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let Ok(sample_time) = parts[0].parse::<f64>() else {
            continue;
        };
        let Ok(sample_adc) = parts[1].parse::<f64>() else {
            continue;
        };
        time.push(sample_time);
        adc.push(sample_adc);
    }

    if time.is_empty() {
        return Err(PyrthError::EmptyInput);
    }

    Ok(T3sterRaw {
        time_microseconds: Array1::from(time),
        adc_count: Array1::from(adc),
        lsb,
        uref,
    })
}

pub fn parse_t3ster_power_step(text: &str) -> Result<f64> {
    for line in text.lines() {
        let trimmed = line.trim();
        let Some((key, value)) = trimmed.split_once('=') else {
            continue;
        };
        let key = key.trim();
        if key.eq_ignore_ascii_case("power") || key.eq_ignore_ascii_case("powerstep") {
            let power =
                value
                    .trim()
                    .parse::<f64>()
                    .map_err(|err| PyrthError::InvalidParameter {
                        parameter: "t3ster power",
                        expected: "numeric Power or POWERSTEP value",
                        actual: err.to_string(),
                    })?;
            if power.is_finite() {
                return Ok(power);
            }
        }
    }

    Err(PyrthError::InvalidParameter {
        parameter: "t3ster power",
        expected: "Power or POWERSTEP entry",
        actual: "missing".to_string(),
    })
}

pub fn parse_t3ster_calibration_text(text: &str) -> Result<Vec<[f64; 2]>> {
    let mut calibration = Vec::new();
    for line in text.lines() {
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }
        let parts = trimmed
            .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();
        if parts.len() < 2 {
            continue;
        }
        let Ok(temperature) = parts[0].parse::<f64>() else {
            continue;
        };
        let Ok(voltage) = parts[1].parse::<f64>() else {
            continue;
        };
        calibration.push([temperature, voltage]);
    }

    if calibration.is_empty() {
        return Err(PyrthError::InvalidParameter {
            parameter: "t3ster calibration",
            expected: "temperature/voltage calibration rows",
            actual: "none".to_string(),
        });
    }

    Ok(calibration)
}

pub fn t3ster_raw_to_temperature_input(
    raw: &T3sterRaw,
    calibration: &[[f64; 2]],
    degree: usize,
) -> Result<TransientInput> {
    if raw.time_microseconds.len() != raw.adc_count.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: raw.time_microseconds.len(),
            value_len: raw.adc_count.len(),
        });
    }

    let first_nonzero = raw
        .time_microseconds
        .iter()
        .position(|time| *time > 0.0)
        .ok_or(PyrthError::InvalidTimeAxis)?;
    let coefficients = polyfit_temperature_voltage(calibration, degree)?;

    let voltage_offset = raw.uref - 4095.0 / 2.0 * raw.lsb;
    let mut pairs = Vec::with_capacity(raw.time_microseconds.len() - first_nonzero);
    for index in first_nonzero..raw.time_microseconds.len() {
        let time = raw.time_microseconds[index] * 1e-6;
        let voltage = raw.adc_count[index] * raw.lsb + voltage_offset;
        let temperature = voltage_to_temperature(voltage, &coefficients, calibration)?;
        pairs.push((time, temperature));
    }

    TransientInput::from_pairs(pairs)
}

fn polyfit_temperature_voltage(calibration: &[[f64; 2]], degree: usize) -> Result<Vec<f64>> {
    if degree == 0 || degree > 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "kfac_fit_deg",
            expected: "1 or 2 for T3Ster calibration",
            actual: degree.to_string(),
        });
    }
    if calibration.len() <= degree {
        return Err(PyrthError::InvalidParameter {
            parameter: "t3ster calibration",
            expected: "more calibration rows than polynomial degree",
            actual: calibration.len().to_string(),
        });
    }

    let size = degree + 1;
    let mut normal = vec![vec![0.0; size]; size];
    let mut rhs = vec![0.0; size];

    for [temperature, voltage] in calibration {
        let powers = (0..size)
            .map(|power| temperature.powi(power as i32))
            .collect::<Vec<_>>();
        for row in 0..size {
            rhs[row] += powers[row] * voltage;
            for col in 0..size {
                normal[row][col] += powers[row] * powers[col];
            }
        }
    }

    solve_linear_system(normal, rhs)
}

fn voltage_to_temperature(
    voltage: f64,
    coefficients: &[f64],
    calibration: &[[f64; 2]],
) -> Result<f64> {
    match coefficients {
        [intercept, slope] => {
            if *slope == 0.0 {
                return Err(PyrthError::InvalidParameter {
                    parameter: "t3ster calibration",
                    expected: "non-zero linear slope",
                    actual: "0".to_string(),
                });
            }
            Ok((voltage - intercept) / slope)
        }
        [c0, c1, c2] => solve_quadratic_temperature(voltage, *c0, *c1, *c2, calibration),
        _ => Err(PyrthError::InvalidParameter {
            parameter: "kfac_fit_deg",
            expected: "1 or 2 for T3Ster calibration",
            actual: (coefficients.len() - 1).to_string(),
        }),
    }
}

fn solve_quadratic_temperature(
    voltage: f64,
    c0: f64,
    c1: f64,
    c2: f64,
    calibration: &[[f64; 2]],
) -> Result<f64> {
    if c2 == 0.0 {
        return voltage_to_temperature(voltage, &[c0, c1], calibration);
    }
    let discriminant = c1 * c1 - 4.0 * c2 * (c0 - voltage);
    if discriminant < 0.0 {
        return Err(PyrthError::InvalidParameter {
            parameter: "t3ster calibration",
            expected: "voltage inside quadratic calibration span",
            actual: voltage.to_string(),
        });
    }
    let sqrt_disc = discriminant.sqrt();
    let left = (-c1 - sqrt_disc) / (2.0 * c2);
    let right = (-c1 + sqrt_disc) / (2.0 * c2);
    let min_temp = calibration
        .iter()
        .map(|row| row[0])
        .fold(f64::INFINITY, f64::min)
        - 20.0;
    let max_temp = calibration
        .iter()
        .map(|row| row[0])
        .fold(f64::NEG_INFINITY, f64::max)
        + 20.0;

    [left, right]
        .into_iter()
        .filter(|value| value.is_finite())
        .min_by(|left, right| {
            distance_to_span(*left, min_temp, max_temp)
                .partial_cmp(&distance_to_span(*right, min_temp, max_temp))
                .unwrap_or(std::cmp::Ordering::Equal)
        })
        .ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "t3ster calibration",
            expected: "finite quadratic root",
            actual: voltage.to_string(),
        })
}

fn distance_to_span(value: f64, min: f64, max: f64) -> f64 {
    if value < min {
        min - value
    } else if value > max {
        value - max
    } else {
        0.0
    }
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
                parameter: "t3ster calibration",
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
