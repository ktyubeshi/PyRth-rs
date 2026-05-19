use ndarray::Array1;
#[cfg(feature = "mpfr")]
use rug::Float;

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
    cauer_network_from_elements(resistance, capacitance)
}

#[cfg(feature = "mpfr")]
pub fn cauer_from_foster_poly_long_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
) -> Result<CauerNetwork> {
    validate_foster_inputs(foster_resistance, foster_capacitance)?;
    let precision = validate_mpfr_precision(precision)?;

    let rational = foster_impedance_rational_mpfr(foster_resistance, foster_capacitance, precision);
    let (resistance, capacitance) =
        poly_long_division_to_cauer_mpfr(&rational.numerator, &rational.denominator, precision)?;
    cauer_network_from_elements(resistance, capacitance)
}

#[cfg(feature = "mpfr")]
pub fn cauer_from_foster_sobhy_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
) -> Result<CauerNetwork> {
    cauer_from_foster_j_fraction_mpfr(
        foster_resistance,
        foster_capacitance,
        precision,
        JFractionMethod::Sobhy,
    )
}

#[cfg(feature = "mpfr")]
pub fn cauer_from_foster_khatwani_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
) -> Result<CauerNetwork> {
    cauer_from_foster_j_fraction_mpfr(
        foster_resistance,
        foster_capacitance,
        precision,
        JFractionMethod::Khatwani,
    )
}

#[cfg(feature = "mpfr")]
/// Mirrors Python's Boor-Golub MPFR output before adapting it to Rust's CauerNetwork contract.
pub fn boor_golub_cauer_mpfr_raw(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
) -> Result<(Vec<f64>, Vec<f64>)> {
    validate_foster_inputs(foster_resistance, foster_capacitance)?;
    let precision = validate_mpfr_precision(precision)?;
    let foster_resistance: Vec<_> = foster_resistance
        .iter()
        .map(|value| Float::with_val(precision, *value))
        .collect();
    let foster_capacitance: Vec<_> = foster_capacitance
        .iter()
        .map(|value| Float::with_val(precision, *value))
        .collect();

    boor_golub_raw_mpfr(&foster_resistance, &foster_capacitance, precision)
}

#[cfg(feature = "mpfr")]
pub fn cauer_from_foster_boor_golub_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
) -> Result<CauerNetwork> {
    let (mut resistance, mut capacitance) =
        boor_golub_cauer_mpfr_raw(foster_resistance, foster_capacitance, precision)?;

    // Python's raw Boor-Golub sequence can end with a zero resistance sentinel.
    // Rust's CauerNetwork stores only physical, positive RC sections.
    while matches!(resistance.last(), Some(value) if value.is_finite() && *value <= 0.0) {
        resistance.pop();
        capacitance.pop();
    }

    cauer_network_from_elements(resistance, capacitance)
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

fn cauer_network_from_elements(
    resistance: Vec<f64>,
    capacitance: Vec<f64>,
) -> Result<CauerNetwork> {
    if resistance.len() != capacitance.len() {
        return Err(PyrthError::LengthMismatch {
            time_len: resistance.len(),
            value_len: capacitance.len(),
        });
    }
    if resistance
        .iter()
        .chain(capacitance.iter())
        .any(|value| !value.is_finite() || *value <= 0.0)
    {
        return invalid_structure("finite positive Cauer elements");
    }

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

#[cfg(feature = "mpfr")]
#[derive(Clone, Debug, PartialEq)]
struct FosterRationalMpfr {
    numerator: Vec<Float>,
    denominator: Vec<Float>,
}

#[cfg(feature = "mpfr")]
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
enum JFractionMethod {
    Sobhy,
    Khatwani,
}

#[cfg(feature = "mpfr")]
fn validate_mpfr_precision(precision: usize) -> Result<u32> {
    u32::try_from(precision).map_err(|_| PyrthError::InvalidParameter {
        parameter: "precision",
        expected: "less than or equal to u32::MAX",
        actual: precision.to_string(),
    })
}

#[cfg(feature = "mpfr")]
fn foster_impedance_rational_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: u32,
) -> FosterRationalMpfr {
    let mut numerator = vec![Float::with_val(precision, 0)];
    let mut denominator = vec![Float::with_val(precision, 1)];

    for (&resistance, &capacitance) in foster_resistance.iter().zip(foster_capacitance) {
        let branch_num = [Float::with_val(precision, resistance)];
        let branch_den = [
            Float::with_val(precision, 1),
            Float::with_val(precision, resistance * capacitance),
        ];

        let num_left = polynomial_mul_mpfr(&numerator, &branch_den, precision);
        let num_right = polynomial_mul_mpfr(&branch_num, &denominator, precision);
        numerator = polynomial_add_mpfr(&num_left, &num_right, precision);
        denominator = polynomial_mul_mpfr(&branch_den, &denominator, precision);

        trim_trailing_zeros_mpfr(&mut numerator);
        trim_trailing_zeros_mpfr(&mut denominator);
    }

    FosterRationalMpfr {
        numerator,
        denominator,
    }
}

#[cfg(feature = "mpfr")]
fn cauer_from_foster_j_fraction_mpfr(
    foster_resistance: &Array1<f64>,
    foster_capacitance: &Array1<f64>,
    precision: usize,
    method: JFractionMethod,
) -> Result<CauerNetwork> {
    validate_foster_inputs(foster_resistance, foster_capacitance)?;
    let precision = validate_mpfr_precision(precision)?;
    let rational = foster_impedance_rational_mpfr(foster_resistance, foster_capacitance, precision);
    let (cleaned_num, cleaned_den) =
        normalize_rational_polynomials_mpfr(&rational.numerator, &rational.denominator, precision)?;
    let n_terms = cleaned_den.len();
    let (large_h, small_h) = match method {
        JFractionMethod::Sobhy => {
            sobhy_method_mpfr(n_terms, &cleaned_num, &cleaned_den, precision)?
        }
        JFractionMethod::Khatwani => {
            let markov_parameters =
                generate_markov_params_mpfr(&cleaned_num, &cleaned_den, precision);
            khatwani_method_mpfr(n_terms, &markov_parameters, precision)?
        }
    };
    let (resistance, capacitance) =
        continued_fraction_to_cauer_mpfr(n_terms, &large_h, &small_h, precision)?;
    cauer_network_from_elements(resistance, capacitance)
}

#[cfg(feature = "mpfr")]
fn normalize_rational_polynomials_mpfr(
    numerator: &[Float],
    denominator: &[Float],
    precision: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    if numerator.is_empty() || denominator.is_empty() {
        return invalid_structure("non-empty numerator and denominator");
    }
    let lead = denominator.last().unwrap();
    if lead == &Float::with_val(precision, 0) {
        return invalid_structure("non-zero denominator leading coefficient");
    }

    let inverse = Float::with_val(precision, Float::with_val(precision, 1) / lead);
    let n_terms = denominator.len();
    let mut cleaned_num = Vec::with_capacity(n_terms);
    cleaned_num.push(Float::with_val(precision, 0));
    for i in 0..n_terms - 1 {
        let source = n_terms - i - 2;
        let value = numerator
            .get(source)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        cleaned_num.push(Float::with_val(precision, &inverse * value));
    }

    let cleaned_den = (0..n_terms)
        .map(|i| Float::with_val(precision, &inverse * &denominator[n_terms - i - 1]))
        .collect();

    Ok((cleaned_num, cleaned_den))
}

#[cfg(feature = "mpfr")]
fn generate_markov_params_mpfr(
    cleaned_num: &[Float],
    cleaned_den: &[Float],
    precision: u32,
) -> Vec<Float> {
    let n_terms = cleaned_den.len();
    let max_order = 2 * n_terms;
    let mut expansion_len = 1;
    let mut last_term = vec![Float::with_val(precision, 1)];
    let mut last_error = cleaned_den[1..].to_vec();

    while expansion_len < max_order {
        let mut pre_term = vec![Float::with_val(precision, 0); expansion_len];
        pre_term.extend(last_term.iter().cloned());

        let correction =
            polynomial_mul_mpfr_truncated(&last_error, &pre_term, max_order, precision);
        last_term = polynomial_sub_mpfr(&last_term, &correction, precision);
        last_error = polynomial_neg_mpfr(&polynomial_mul_mpfr_truncated(
            &last_error,
            &last_error,
            max_order,
            precision,
        ));

        expansion_len *= 2;
    }

    let product = polynomial_mul_mpfr(cleaned_num, &last_term, precision);
    (1..=max_order)
        .map(|i| {
            product
                .get(i)
                .cloned()
                .unwrap_or_else(|| Float::with_val(precision, 0))
        })
        .collect()
}

#[cfg(feature = "mpfr")]
fn boor_golub_raw_mpfr(
    foster_resistance: &[Float],
    foster_capacitance: &[Float],
    precision: u32,
) -> Result<(Vec<f64>, Vec<f64>)> {
    let poles: Vec<_> = foster_resistance
        .iter()
        .zip(foster_capacitance)
        .filter_map(|(resistance, capacitance)| {
            if resistance > &Float::with_val(precision, 0)
                && capacitance > &Float::with_val(precision, 0)
            {
                Some(Float::with_val(
                    precision,
                    Float::with_val(precision, -1)
                        / Float::with_val(precision, resistance * capacitance),
                ))
            } else {
                None
            }
        })
        .collect();
    if poles.is_empty() {
        return invalid_structure("at least one positive Foster pole for Boor-Golub conversion");
    }

    let pole_count = poles.len() - 1;
    let weights: Vec<_> = foster_capacitance
        .iter()
        .take(poles.len())
        .map(|capacitance| checked_div_mpfr(&Float::with_val(precision, 1), capacitance, precision))
        .collect::<Result<_>>()?;

    let mut k = vec![Float::with_val(precision, 0); 2 * poles.len()];
    let weight_sum = weights
        .iter()
        .fold(Float::with_val(precision, 0), |sum, weight| {
            Float::with_val(precision, sum + weight)
        });
    k[1] = checked_div_mpfr(&Float::with_val(precision, 1), &weight_sum, precision)?;

    if pole_count == 0 {
        return Ok((
            vec![foster_resistance[0].to_f64()],
            vec![foster_capacitance[0].to_f64()],
        ));
    }

    let mut lambda = vec![Float::with_val(precision, 0); poles.len()];
    let mut mu = vec![Float::with_val(precision, 0); poles.len()];
    let mut lambda_mu_sum = vec![Float::with_val(precision, 0); pole_count];
    let mut lambda_mu_prod = vec![Float::with_val(precision, 0); pole_count];
    let mut polynomial_basis = vec![Vec::<Float>::new(); poles.len()];
    polynomial_basis[0] = vec![Float::with_val(precision, 1)];

    for i in 0..poles.len() {
        lambda[0] = Float::with_val(precision, &lambda[0] - &weights[i] * &poles[i]);
    }
    lambda[0] = checked_div_mpfr(&lambda[0], &weight_sum, precision)?;

    k[2] = checked_div_mpfr(
        &Float::with_val(precision, 1),
        &Float::with_val(precision, &k[1] * &lambda[0]),
        precision,
    )?;
    polynomial_basis[1] = vec![lambda[0].clone(), Float::with_val(precision, 1)];

    lambda_mu_prod[0] = checked_div_mpfr(
        &weighted_self_product_mpfr(&poles, &polynomial_basis[1], &weights, precision),
        &weighted_self_product_mpfr(&poles, &polynomial_basis[0], &weights, precision),
        precision,
    )?;
    mu[1] = checked_div_mpfr(&lambda_mu_prod[0], &lambda[0], precision)?;

    for i in 2..=pole_count {
        let mut shifted_basis = Vec::with_capacity(polynomial_basis[i - 1].len() + 1);
        shifted_basis.push(Float::with_val(precision, 0));
        shifted_basis.extend(polynomial_basis[i - 1].iter().cloned());

        lambda_mu_sum[i - 1] = checked_div_mpfr(
            &weighted_inner_product_mpfr(
                &poles,
                &polynomial_basis[i - 1],
                &shifted_basis,
                &weights,
                precision,
            ),
            &weighted_self_product_mpfr(&poles, &polynomial_basis[i - 1], &weights, precision),
            precision,
        )?;
        lambda[i - 1] = Float::with_val(precision, &lambda_mu_sum[i - 1] - &mu[i - 1]);

        let first = polynomial_mul_mpfr(
            &[lambda_mu_sum[i - 1].clone(), Float::with_val(precision, 1)],
            &polynomial_basis[i - 1],
            precision,
        );
        let second = polynomial_mul_mpfr(
            &[Float::with_val(precision, -&lambda_mu_prod[i - 2])],
            &polynomial_basis[i - 2],
            precision,
        );
        polynomial_basis[i] = polynomial_add_mpfr(&first, &second, precision);

        lambda_mu_prod[i - 1] = checked_div_mpfr(
            &weighted_self_product_mpfr(&poles, &polynomial_basis[i], &weights, precision),
            &weighted_self_product_mpfr(&poles, &polynomial_basis[i - 1], &weights, precision),
            precision,
        )?;
        mu[i] = checked_div_mpfr(&lambda_mu_prod[i - 1], &lambda[i - 1], precision)?;
    }

    k[3] = checked_div_mpfr(
        &Float::with_val(precision, &k[1] * &lambda[0]),
        &mu[1],
        precision,
    )?;
    for i in 2..=pole_count {
        let mut lambdas = k[1].clone();
        let mut mus = mu[1].clone();
        for item in mu.iter().take(i).skip(2) {
            mus = Float::with_val(precision, mus * item);
        }
        for item in lambda.iter().take(i) {
            lambdas = Float::with_val(precision, lambdas * item);
        }
        k[2 * i] = checked_div_mpfr(&mus, &lambdas, precision)?;
        k[2 * i + 1] = checked_div_mpfr(
            &lambdas,
            &Float::with_val(precision, mus * &mu[i]),
            precision,
        )?;
    }

    let mut resistance = vec![0.0; poles.len()];
    let mut capacitance = vec![0.0; poles.len()];
    for i in 0..pole_count {
        resistance[i] = k[2 * i + 2].to_f64();
        capacitance[i] = k[2 * i + 1].to_f64();
    }
    capacitance[pole_count] = k[2 * pole_count + 1].to_f64();

    Ok((resistance, capacitance))
}

#[cfg(feature = "mpfr")]
fn khatwani_method_mpfr(
    n_terms: usize,
    markov_parameters: &[Float],
    precision: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    if n_terms < 2 || markov_parameters.len() < 2 * n_terms {
        return invalid_structure("sufficient Markov parameters for Khatwani conversion");
    }

    let mut a_matrix = vec![vec![Float::with_val(precision, 0); 2 * n_terms]; n_terms + 1];
    a_matrix[0][0] = Float::with_val(precision, 1);
    a_matrix[1][..(2 * n_terms)].clone_from_slice(&markov_parameters[..(2 * n_terms)]);

    let mut large_h = vec![Float::with_val(precision, 0); n_terms - 1];
    let mut small_h = vec![Float::with_val(precision, 0); n_terms - 1];

    large_h[0] = checked_div_mpfr(&a_matrix[0][0], &a_matrix[1][0], precision)?;
    small_h[0] = checked_div_mpfr(
        &Float::with_val(precision, &a_matrix[0][1] - &large_h[0] * &a_matrix[1][1]),
        &a_matrix[1][0],
        precision,
    )?;

    for i in 2..n_terms {
        for j in 0..(2 * n_terms - (i - 1) * 2) {
            a_matrix[i][j] = Float::with_val(
                precision,
                &a_matrix[i - 2][j + 2]
                    - Float::with_val(precision, &large_h[i - 2] * &a_matrix[i - 1][j + 2])
                    - Float::with_val(precision, &small_h[i - 2] * &a_matrix[i - 1][j + 1]),
            );
        }

        large_h[i - 1] = checked_div_mpfr(&a_matrix[i - 1][0], &a_matrix[i][0], precision)?;
        small_h[i - 1] = checked_div_mpfr(
            &Float::with_val(
                precision,
                &a_matrix[i - 1][1] - Float::with_val(precision, &large_h[i - 1] * &a_matrix[i][1]),
            ),
            &a_matrix[i][0],
            precision,
        )?;
    }

    Ok((large_h, small_h))
}

#[cfg(feature = "mpfr")]
fn sobhy_method_mpfr(
    n_terms: usize,
    cleaned_num: &[Float],
    cleaned_den: &[Float],
    precision: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    if n_terms < 2 || cleaned_num.len() < n_terms || cleaned_den.len() < n_terms {
        return invalid_structure("normalized rational polynomials for Sobhy conversion");
    }

    let mut a_table = vec![vec![Float::with_val(precision, 0); n_terms]; n_terms + 1];
    let mut b_table = vec![vec![Float::with_val(precision, 0); n_terms]; n_terms + 1];

    for i in 0..n_terms {
        a_table[0][i] = cleaned_den[i].clone();
        b_table[0][i] = cleaned_den[i].clone();
    }
    for i in 0..n_terms - 1 {
        a_table[1][i] = cleaned_num[i + 1].clone();
    }

    for k in 0..n_terms - 1 {
        b_table[1][k] = Float::with_val(
            precision,
            &a_table[0][k + 1]
                - checked_div_mpfr(&a_table[0][0], &a_table[1][0], precision)? * &a_table[1][k + 1],
        );
    }

    for j in 2..=n_terms {
        for k in 0..n_terms - j {
            a_table[j][k] = Float::with_val(
                precision,
                &b_table[j - 1][k + 1]
                    - checked_div_mpfr(&b_table[j - 1][0], &a_table[j - 1][0], precision)?
                        * &a_table[j - 1][k + 1],
            );
        }
        for k in 0..n_terms - j {
            b_table[j][k] = Float::with_val(
                precision,
                &a_table[j - 1][k + 1]
                    - checked_div_mpfr(&a_table[j - 1][0], &a_table[j][0], precision)?
                        * &a_table[j][k + 1],
            );
        }
    }

    let mut large_h = Vec::with_capacity(n_terms - 1);
    let mut small_h = Vec::with_capacity(n_terms - 1);
    for m in 1..n_terms {
        large_h.push(checked_div_mpfr(
            &a_table[m - 1][0],
            &a_table[m][0],
            precision,
        )?);
        small_h.push(checked_div_mpfr(&b_table[m][0], &a_table[m][0], precision)?);
    }

    Ok((large_h, small_h))
}

#[cfg(feature = "mpfr")]
fn continued_fraction_to_cauer_mpfr(
    n_terms: usize,
    large_h: &[Float],
    small_h: &[Float],
    precision: u32,
) -> Result<(Vec<f64>, Vec<f64>)> {
    if n_terms < 2 || large_h.len() < n_terms - 1 || small_h.len() < n_terms - 1 {
        return invalid_structure("H-h continued fraction coefficients");
    }

    let mut a_square = vec![Float::with_val(precision, 0); n_terms - 1];
    let mut small_b = vec![Float::with_val(precision, 0); n_terms - 1];

    a_square[0] = checked_div_mpfr(&Float::with_val(precision, 1), &large_h[0], precision)?;
    small_b[0] = Float::with_val(precision, -&small_h[0] / &large_h[0]);

    for i in 1..n_terms - 1 {
        a_square[i] = checked_div_mpfr(
            &Float::with_val(precision, -1),
            &Float::with_val(precision, &large_h[i] * &large_h[i - 1]),
            precision,
        )?;
        small_b[i] = Float::with_val(precision, -&small_h[i] / &large_h[i]);
    }

    let mut small_c = vec![Float::with_val(precision, 0); 2 * (n_terms - 1)];
    small_c[0] = checked_div_mpfr(&Float::with_val(precision, 1), &a_square[0], precision)?;
    small_c[1] = checked_div_mpfr(
        &Float::with_val(precision, -&a_square[0]),
        &small_b[0],
        precision,
    )?;

    for i in 1..n_terms - 1 {
        small_c[2 * i] = checked_div_mpfr(
            &Float::with_val(precision, 1),
            &Float::with_val(
                precision,
                &small_c[2 * i - 2] * &small_c[2 * i - 1] * &small_c[2 * i - 1] * &a_square[i],
            ),
            precision,
        )?;
        small_c[2 * i + 1] = checked_div_mpfr(
            &Float::with_val(precision, -&small_c[2 * i - 1]),
            &Float::with_val(
                precision,
                Float::with_val(precision, 1)
                    + Float::with_val(
                        precision,
                        &small_c[2 * i] * &small_c[2 * i - 1] * &small_b[i],
                    ),
            ),
            precision,
        )?;
    }

    let mut resistance = Vec::with_capacity(n_terms - 1);
    let mut capacitance = Vec::with_capacity(n_terms - 1);
    for i in 0..n_terms - 1 {
        capacitance.push(small_c[2 * i].to_f64());
        resistance.push(small_c[2 * i + 1].to_f64());
    }

    Ok((resistance, capacitance))
}

#[cfg(feature = "mpfr")]
fn poly_long_division_to_cauer_mpfr(
    numerator: &[Float],
    denominator: &[Float],
    precision: u32,
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
        let (next_num, next_den, cap, res) = precision_step_mpfr(&num, &den, precision)?;
        let cap_f64 = cap.to_f64();
        let res_f64 = res.to_f64();
        if !res_f64.is_finite() || !cap_f64.is_finite() || res_f64 <= 0.0 || cap_f64 <= 0.0 {
            return invalid_structure("finite positive Cauer elements");
        }
        resistance.push(res_f64);
        capacitance.push(cap_f64);
        num = next_num;
        den = next_den;
    }

    Ok((resistance, capacitance))
}

#[cfg(feature = "mpfr")]
fn precision_step_mpfr(
    numerator: &[Float],
    denominator: &[Float],
    precision: u32,
) -> Result<(Vec<Float>, Vec<Float>, Float, Float)> {
    let (quotient, remainder) = polynomial_division_mpfr(denominator, numerator, precision)?;
    if quotient.len() < 2 || quotient[0] == 0 {
        return invalid_structure("linear quotient with non-zero constant term");
    }

    let res_inv = quotient[0].clone();
    let cap = quotient[1].clone();
    let res = Float::with_val(precision, Float::with_val(precision, 1) / &res_inv);

    let mut num_new = Vec::with_capacity(numerator.len());
    let mut den_new = Vec::with_capacity(numerator.len());
    for i in 0..numerator.len() {
        let rem = remainder
            .get(i)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        num_new.push(Float::with_val(
            precision,
            -Float::with_val(precision, &res * &rem),
        ));
        den_new.push(Float::with_val(
            precision,
            Float::with_val(precision, &res_inv * &numerator[i]) + rem,
        ));
    }
    trim_trailing_zeros_mpfr(&mut num_new);
    trim_trailing_zeros_mpfr(&mut den_new);

    Ok((num_new, den_new, cap, res))
}

#[cfg(feature = "mpfr")]
fn polynomial_division_mpfr(
    numerator: &[Float],
    denominator: &[Float],
    precision: u32,
) -> Result<(Vec<Float>, Vec<Float>)> {
    if denominator.is_empty() {
        return invalid_structure("non-empty division denominator");
    }

    let numerator_degree = degree_mpfr(numerator);
    let denominator_degree =
        degree_mpfr(denominator).ok_or_else(|| PyrthError::InvalidParameter {
            parameter: "structure_method",
            expected: "non-zero polynomial divisor",
            actual: "zero polynomial".to_string(),
        })?;

    let Some(numerator_degree) = numerator_degree else {
        return Ok((
            vec![Float::with_val(precision, 0)],
            vec![Float::with_val(precision, 0); numerator.len()],
        ));
    };

    let mut remainder = numerator.to_vec();
    let mut quotient = vec![Float::with_val(precision, 0); numerator.len()];

    if numerator_degree >= denominator_degree {
        for k in (0..=numerator_degree - denominator_degree).rev() {
            quotient[k] = Float::with_val(
                precision,
                &remainder[denominator_degree + k] / &denominator[denominator_degree],
            );
            for j in (k..denominator_degree + k).rev() {
                let term = Float::with_val(precision, &quotient[k] * &denominator[j - k]);
                remainder[j] = Float::with_val(precision, &remainder[j] - term);
            }
        }
    }

    for item in remainder
        .iter_mut()
        .take(numerator_degree + 1)
        .skip(denominator_degree)
    {
        *item = Float::with_val(precision, 0);
    }
    trim_trailing_zeros_mpfr(&mut quotient);
    trim_trailing_zeros_mpfr(&mut remainder);

    Ok((quotient, remainder))
}

#[cfg(feature = "mpfr")]
fn polynomial_mul_mpfr(left: &[Float], right: &[Float], precision: u32) -> Vec<Float> {
    let mut product = vec![Float::with_val(precision, 0); left.len() + right.len() - 1];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let term = Float::with_val(precision, left_value * right_value);
            product[left_index + right_index] =
                Float::with_val(precision, &product[left_index + right_index] + term);
        }
    }
    product
}

#[cfg(feature = "mpfr")]
fn polynomial_add_mpfr(left: &[Float], right: &[Float], precision: u32) -> Vec<Float> {
    let len = left.len().max(right.len());
    let mut sum = vec![Float::with_val(precision, 0); len];
    for (i, value) in sum.iter_mut().enumerate() {
        let left_value = left
            .get(i)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        let right_value = right
            .get(i)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        *value = Float::with_val(precision, left_value + right_value);
    }
    sum
}

#[cfg(feature = "mpfr")]
fn polynomial_sub_mpfr(left: &[Float], right: &[Float], precision: u32) -> Vec<Float> {
    let len = left.len().max(right.len());
    let mut diff = vec![Float::with_val(precision, 0); len];
    for (i, value) in diff.iter_mut().enumerate() {
        let left_value = left
            .get(i)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        let right_value = right
            .get(i)
            .cloned()
            .unwrap_or_else(|| Float::with_val(precision, 0));
        *value = Float::with_val(precision, left_value - right_value);
    }
    trim_trailing_zeros_mpfr(&mut diff);
    diff
}

#[cfg(feature = "mpfr")]
fn polynomial_neg_mpfr(values: &[Float]) -> Vec<Float> {
    values
        .iter()
        .map(|value| Float::with_val(value.prec(), -value))
        .collect()
}

#[cfg(feature = "mpfr")]
fn horner_poly_eval_mpfr(value: &Float, polynomial: &[Float], precision: u32) -> Float {
    let mut result = Float::with_val(precision, 0);
    for coefficient in polynomial
        .iter()
        .rev()
        .take(polynomial.len().saturating_sub(1))
    {
        result = Float::with_val(
            precision,
            Float::with_val(precision, result + coefficient) * value,
        );
    }
    Float::with_val(
        precision,
        result
            + polynomial
                .first()
                .cloned()
                .unwrap_or_else(|| Float::with_val(precision, 0)),
    )
}

#[cfg(feature = "mpfr")]
fn weighted_inner_product_mpfr(
    poles: &[Float],
    left: &[Float],
    right: &[Float],
    weights: &[Float],
    precision: u32,
) -> Float {
    let mut product = Float::with_val(precision, 0);
    for i in 0..poles.len() {
        let left_value = horner_poly_eval_mpfr(&poles[i], left, precision);
        let right_value = horner_poly_eval_mpfr(&poles[i], right, precision);
        product = Float::with_val(
            precision,
            product - Float::with_val(precision, left_value * right_value * &weights[i]),
        );
    }
    product
}

#[cfg(feature = "mpfr")]
fn weighted_self_product_mpfr(
    poles: &[Float],
    polynomial: &[Float],
    weights: &[Float],
    precision: u32,
) -> Float {
    let mut product = Float::with_val(precision, 0);
    for i in 0..poles.len() {
        let value = horner_poly_eval_mpfr(&poles[i], polynomial, precision);
        product = Float::with_val(
            precision,
            product + Float::with_val(precision, &value * value * &weights[i]),
        );
    }
    product
}

#[cfg(feature = "mpfr")]
fn polynomial_mul_mpfr_truncated(
    left: &[Float],
    right: &[Float],
    max_len: usize,
    precision: u32,
) -> Vec<Float> {
    let len = (left.len() + right.len() - 1).min(max_len);
    let mut product = vec![Float::with_val(precision, 0); len];
    for (left_index, left_value) in left.iter().enumerate() {
        for (right_index, right_value) in right.iter().enumerate() {
            let index = left_index + right_index;
            if index >= max_len {
                break;
            }
            let term = Float::with_val(precision, left_value * right_value);
            product[index] = Float::with_val(precision, &product[index] + term);
        }
    }
    trim_trailing_zeros_mpfr(&mut product);
    product
}

#[cfg(feature = "mpfr")]
fn checked_div_mpfr(numerator: &Float, denominator: &Float, precision: u32) -> Result<Float> {
    if denominator == &Float::with_val(precision, 0) {
        return invalid_structure("non-zero divisor in MPFR conversion");
    }
    Ok(Float::with_val(precision, numerator / denominator))
}

#[cfg(feature = "mpfr")]
fn trim_trailing_zeros_mpfr(values: &mut Vec<Float>) {
    while values.len() > 1 && values.last().is_some_and(|value| *value == 0) {
        values.pop();
    }
}

#[cfg(feature = "mpfr")]
fn degree_mpfr(values: &[Float]) -> Option<usize> {
    values.iter().rposition(|value| *value != 0)
}

fn invalid_structure<T>(expected: &'static str) -> Result<T> {
    Err(PyrthError::InvalidParameter {
        parameter: "structure_method",
        expected,
        actual: "finite-precision polynomial conversion failed".to_string(),
    })
}
