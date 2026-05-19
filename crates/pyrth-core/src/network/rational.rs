use ndarray::Array1;

use crate::{
    error::{PyrthError, Result},
    evaluation::CauerNetwork,
};

use super::cauer::{cumulative_sum, differential_structure};

#[derive(Clone, Debug, PartialEq)]
pub struct FosterRationalF64 {
    pub numerator: Vec<f64>,
    pub denominator: Vec<f64>,
}

pub fn foster_impedance_rational_f64(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
) -> Result<FosterRationalF64> {
    validate_foster_inputs(foster_resistance, foster_capacitance)?;

    let mut numerator = vec![0.0];
    let mut denominator = vec![1.0];

    for (&resistance, &capacitance) in foster_resistance.iter().zip(foster_capacitance) {
        let branch_num = [resistance];
        let branch_den = [1.0, resistance * capacitance];

        let num_left = polynomial_mul(&numerator, &branch_den);
        let num_right = polynomial_mul(&branch_num, &denominator);
        numerator = polynomial_add(&num_left, &num_right);
        denominator = polynomial_mul(&branch_den, &denominator);

        trim_trailing_zeros(&mut numerator);
        trim_trailing_zeros(&mut denominator);
    }

    Ok(FosterRationalF64 {
        numerator,
        denominator,
    })
}

pub fn cauer_from_foster_poly_long_f64(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
) -> Result<CauerNetwork> {
    let rational = foster_impedance_rational_f64(foster_resistance, foster_capacitance)?;
    let (resistance, capacitance) =
        poly_long_division_to_cauer_f64(&rational.numerator, &rational.denominator)?;
    let cumulative_resistance = cumulative_sum(&resistance);
    let cumulative_capacitance = cumulative_sum(&capacitance);
    let differential_structure =
        differential_structure(&cumulative_resistance, &cumulative_capacitance);

    Ok(CauerNetwork {
        resistance: Array1::from(resistance),
        capacitance: Array1::from(capacitance),
        cumulative_resistance: Array1::from(cumulative_resistance),
        cumulative_capacitance: Array1::from(cumulative_capacitance),
        differential_structure: Array1::from(differential_structure),
    })
}

pub(crate) fn poly_long_division_to_cauer_f64(
    numerator: &[f64],
    denominator: &[f64],
) -> Result<(Vec<f64>, Vec<f64>)> {
    if numerator.is_empty() || denominator.len() < 2 {
        return invalid_structure("non-empty numerator and denominator degree at least one");
    }

    let terms = denominator.len() - 1;
    let mut resistance = Vec::with_capacity(terms);
    let mut capacitance = Vec::with_capacity(terms);
    let mut num = numerator.to_vec();
    let mut den = denominator.to_vec();

    for _ in 0..terms {
        let (next_num, next_den, cap, res) = precision_step_f64(&num, &den)?;
        if !res.is_finite() || !cap.is_finite() || res <= 0.0 || cap <= 0.0 {
            return invalid_structure("finite positive Cauer elements");
        }
        resistance.push(res);
        capacitance.push(cap);
        num = next_num;
        den = next_den;
    }

    Ok((resistance, capacitance))
}

pub fn normalize_rational_polynomials_f64(
    numerator: &[f64],
    denominator: &[f64],
) -> Result<FosterRationalF64> {
    if numerator.is_empty() || denominator.is_empty() {
        return invalid_structure("non-empty numerator and denominator");
    }
    let lead = *denominator.last().unwrap();
    if !lead.is_finite() || lead == 0.0 {
        return invalid_structure("finite non-zero denominator leading coefficient");
    }

    let inverse = 1.0 / lead;
    let n_terms = denominator.len();
    let mut cleaned_num = Vec::with_capacity(n_terms);
    cleaned_num.push(0.0);
    for i in 0..n_terms - 1 {
        let source = n_terms - i - 2;
        cleaned_num.push(numerator.get(source).copied().unwrap_or(0.0) * inverse);
    }

    let cleaned_den = (0..n_terms)
        .map(|i| denominator[n_terms - i - 1] * inverse)
        .collect();

    Ok(FosterRationalF64 {
        numerator: cleaned_num,
        denominator: cleaned_den,
    })
}

fn precision_step_f64(
    numerator: &[f64],
    denominator: &[f64],
) -> Result<(Vec<f64>, Vec<f64>, f64, f64)> {
    let (quotient, remainder) = polynomial_division(denominator, numerator)?;
    if quotient.len() < 2 || quotient[0] == 0.0 {
        return invalid_structure("linear quotient with non-zero constant term");
    }

    let res_inv = quotient[0];
    let cap = quotient[1];
    let res = 1.0 / res_inv;

    let mut num_new = Vec::with_capacity(numerator.len());
    let mut den_new = Vec::with_capacity(numerator.len());
    for i in 0..numerator.len() {
        let rem = remainder.get(i).copied().unwrap_or(0.0);
        num_new.push(-res * rem);
        den_new.push(res_inv * numerator[i] + rem);
    }
    trim_trailing_zeros(&mut num_new);
    trim_trailing_zeros(&mut den_new);

    Ok((num_new, den_new, cap, res))
}

fn polynomial_division(numerator: &[f64], denominator: &[f64]) -> Result<(Vec<f64>, Vec<f64>)> {
    if denominator.is_empty() {
        return invalid_structure("non-empty division denominator");
    }

    let numerator_degree = degree(numerator);
    let denominator_degree = degree(denominator).ok_or_else(|| PyrthError::InvalidParameter {
        parameter: "structure_method",
        expected: "non-zero polynomial divisor",
        actual: "zero polynomial".to_string(),
    })?;

    let Some(numerator_degree) = numerator_degree else {
        return Ok((vec![0.0], vec![0.0; numerator.len()]));
    };

    let mut remainder = numerator.to_vec();
    let mut quotient = vec![0.0; numerator.len()];

    if numerator_degree >= denominator_degree {
        for k in (0..=numerator_degree - denominator_degree).rev() {
            quotient[k] = remainder[denominator_degree + k] / denominator[denominator_degree];
            for j in (k..denominator_degree + k).rev() {
                remainder[j] -= quotient[k] * denominator[j - k];
            }
        }
    }

    for item in remainder
        .iter_mut()
        .take(numerator_degree + 1)
        .skip(denominator_degree)
    {
        *item = 0.0;
    }
    trim_trailing_zeros(&mut quotient);
    trim_trailing_zeros(&mut remainder);

    Ok((quotient, remainder))
}

fn polynomial_mul(left: &[f64], right: &[f64]) -> Vec<f64> {
    let mut product = vec![0.0; left.len() + right.len() - 1];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            product[left_index + right_index] += left_value * right_value;
        }
    }
    product
}

fn polynomial_add(left: &[f64], right: &[f64]) -> Vec<f64> {
    let len = left.len().max(right.len());
    let mut sum = vec![0.0; len];
    for i in 0..len {
        sum[i] = left.get(i).copied().unwrap_or(0.0) + right.get(i).copied().unwrap_or(0.0);
    }
    sum
}

fn trim_trailing_zeros(values: &mut Vec<f64>) {
    while values.len() > 1 && values.last().is_some_and(|value| *value == 0.0) {
        values.pop();
    }
}

fn degree(values: &[f64]) -> Option<usize> {
    values.iter().rposition(|value| *value != 0.0)
}

fn validate_foster_inputs(resistance: &Array1<f64>, capacitance: &Array1<f64>) -> Result<()> {
    if resistance.len() != capacitance.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: resistance.len(),
            value_len: capacitance.len(),
        });
    }
    if resistance.is_empty() {
        return Err(PyrthError::EmptySpectrum);
    }
    if resistance
        .iter()
        .chain(capacitance.iter())
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return invalid_structure("finite positive Foster resistance and capacitance values");
    }
    Ok(())
}

fn invalid_structure<T>(expected: &'static str) -> Result<T> {
    Err(PyrthError::InvalidParameter {
        parameter: "structure_method",
        expected,
        actual: "finite-precision polynomial conversion failed".to_string(),
    })
}
