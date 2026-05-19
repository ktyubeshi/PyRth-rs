use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::{
    config::{EvaluationParams, FourierFilter},
    evaluation::DerivativeResult,
};

pub fn response_matrix(domain: &Array1<f64>) -> Array2<f64> {
    let len = domain.len();
    let norm = domain
        .iter()
        .copied()
        .map(|value| (value - value.exp()).exp())
        .sum::<f64>();

    let mut response = Array2::zeros((len, len));
    for row in 0..len {
        for col in 0..len {
            let value = domain[row] - domain[col];
            response[(row, col)] = (value - value.exp()).exp() / norm;
        }
    }
    response
}

pub fn bayesian_deconvolution(
    response: &Array2<f64>,
    imp_deriv_interp: &Array1<f64>,
    steps: usize,
) -> Array1<f64> {
    let mut estimate = imp_deriv_interp.clone();

    for _ in 0..steps {
        let mut denom = response.dot(&estimate);
        denom.mapv_inplace(|value| if value == 0.0 { f64::INFINITY } else { value });

        let q_vec = imp_deriv_interp / &denom;
        let k_sum = q_vec.dot(response);
        estimate *= &k_sum;
    }

    estimate
}

pub fn time_spectrum_bayesian(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
) -> Array1<f64> {
    let matrix = response_matrix(&derivative.log_time_pad);
    bayesian_deconvolution(&matrix, &derivative.imp_deriv_interp, params.bay_steps)
        * derivative.log_time_delta
}

pub fn time_spectrum_fourier(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
) -> Array1<f64> {
    let len = derivative.imp_deriv_interp.len();
    if len == 0 {
        return Array1::zeros(0);
    }

    let mut planner = FftPlanner::new();
    let fft = planner.plan_fft_forward(len);
    let ifft = planner.plan_fft_inverse(len);

    let mut signal = derivative
        .imp_deriv_interp
        .iter()
        .copied()
        .map(|value| Complex::new(value, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut signal);

    let null_index = derivative
        .log_time_pad
        .iter()
        .position(|value| *value >= 0.0)
        .unwrap_or(len);
    let weight = derivative
        .log_time_pad
        .iter()
        .copied()
        .map(weight_z)
        .collect::<Vec<_>>();
    let mut shifted_weight = Vec::with_capacity(len);
    for index in 0..len {
        shifted_weight.push(weight[(index + null_index) % len]);
    }

    let mut kernel = shifted_weight
        .into_iter()
        .map(|value| Complex::new(value, 0.0))
        .collect::<Vec<_>>();
    fft.process(&mut kernel);
    for value in &mut kernel {
        *value *= derivative.log_time_delta;
    }

    let epsilon = 1e-12;
    let filter = fourier_filter(
        params.filter_name,
        len,
        derivative.log_time_delta,
        params.filter_range,
        params.filter_parameter,
    );
    let mut deconvolved = signal
        .into_iter()
        .zip(kernel)
        .zip(filter)
        .map(|((signal_value, kernel_value), filter_value)| {
            if kernel_value.norm_sqr() <= epsilon * epsilon {
                Complex::new(0.0, 0.0)
            } else {
                (signal_value / kernel_value) * filter_value
            }
        })
        .collect::<Vec<_>>();

    ifft.process(&mut deconvolved);
    let scale = derivative.log_time_delta / len as f64;
    Array1::from_iter(deconvolved.into_iter().map(|value| {
        let spectrum_value = value.re * scale;
        if spectrum_value.is_finite() {
            spectrum_value
        } else {
            0.0
        }
    }))
}

pub fn time_spectrum_lasso(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
) -> Array1<f64> {
    time_spectrum_lasso_with_alpha(derivative, params, params.lasso_alpha)
}

pub fn time_spectrum_adaptive(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
) -> Array1<f64> {
    let bayesian_prior = time_spectrum_bayesian(derivative, params);
    time_spectrum_lasso_with_weights(
        derivative,
        params,
        Some(&adaptive_lasso_weights(&bayesian_prior)),
    )
}

fn time_spectrum_lasso_with_alpha(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
    alpha: f64,
) -> Array1<f64> {
    let mut params = params.clone();
    params.lasso_alpha = alpha;
    time_spectrum_lasso_with_weights(derivative, &params, None)
}

fn time_spectrum_lasso_with_weights(
    derivative: &DerivativeResult,
    params: &EvaluationParams,
    weights: Option<&Array1<f64>>,
) -> Array1<f64> {
    let design = response_matrix(&derivative.log_time_pad);
    let (normalized_design, column_norms) = normalize_columns(&design);
    let weighted_design = match weights {
        Some(weights) => apply_column_weights(&normalized_design, weights),
        None => normalized_design,
    };
    let normalized_coefficients = nonnegative_lasso_coordinate_descent(
        &weighted_design,
        &derivative.imp_deriv_interp,
        params.lasso_alpha,
        params.lasso_max_iter,
        params.lasso_tol,
    );

    Array1::from_iter(
        normalized_coefficients
            .iter()
            .zip(column_norms.iter())
            .enumerate()
            .map(|(index, (coefficient, norm))| {
                let weight = weights.map(|weights| weights[index]).unwrap_or(1.0);
                coefficient / (norm * weight)
            }),
    ) * derivative.log_time_delta
}

fn adaptive_lasso_weights(prior: &Array1<f64>) -> Array1<f64> {
    let epsilon = 1e-6;
    let gamma = 0.8;
    prior.mapv(|value| {
        let weight = 1.0 / (value.abs() + epsilon).powf(gamma);
        weight.clamp(0.05, 20.0)
    })
}

fn apply_column_weights(design: &Array2<f64>, weights: &Array1<f64>) -> Array2<f64> {
    let mut weighted = design.clone();
    for col in 0..weighted.ncols() {
        let weight = weights[col];
        for row in 0..weighted.nrows() {
            weighted[(row, col)] /= weight;
        }
    }
    weighted
}

fn normalize_columns(design: &Array2<f64>) -> (Array2<f64>, Array1<f64>) {
    let mut normalized = design.clone();
    let mut norms = Array1::ones(design.ncols());

    for col in 0..design.ncols() {
        let norm = design
            .column(col)
            .iter()
            .map(|value| value * value)
            .sum::<f64>()
            .sqrt();
        let norm = if norm == 0.0 { 1.0 } else { norm };
        norms[col] = norm;
        for row in 0..design.nrows() {
            normalized[(row, col)] /= norm;
        }
    }

    (normalized, norms)
}

fn nonnegative_lasso_coordinate_descent(
    design: &Array2<f64>,
    target: &Array1<f64>,
    alpha: f64,
    max_iter: usize,
    tol: f64,
) -> Array1<f64> {
    let sample_count = target.len() as f64;
    let mut coefficients = Array1::zeros(design.ncols());
    let mut residual = target.clone();
    let column_norm_sq = (0..design.ncols())
        .map(|col| {
            design
                .column(col)
                .iter()
                .map(|value| value * value)
                .sum::<f64>()
        })
        .collect::<Vec<_>>();

    for _ in 0..max_iter {
        let mut max_change = 0.0_f64;
        let mut max_value = 0.0_f64;

        for col in 0..design.ncols() {
            let old = coefficients[col];
            if old != 0.0 {
                for row in 0..design.nrows() {
                    residual[row] += design[(row, col)] * old;
                }
            }

            let rho = design.column(col).dot(&residual);
            let norm_sq = column_norm_sq[col];
            let new_value = if norm_sq == 0.0 {
                0.0
            } else {
                ((rho - alpha * sample_count) / norm_sq).max(0.0)
            };

            if new_value != 0.0 {
                for row in 0..design.nrows() {
                    residual[row] -= design[(row, col)] * new_value;
                }
            }

            coefficients[col] = new_value;
            max_change = max_change.max((new_value - old).abs());
            max_value = max_value.max(new_value.abs());
        }

        if max_change <= tol * max_value.max(1.0) {
            break;
        }
    }

    coefficients
}

fn weight_z(value: f64) -> f64 {
    (value - value.exp()).exp()
}

fn fourier_filter(
    filter_name: FourierFilter,
    len: usize,
    spacing: f64,
    filter_range: f64,
    filter_parameter: f64,
) -> Vec<f64> {
    if len == 0 {
        return Vec::new();
    }
    let shifted_freq = fftshifted_frequencies(len, spacing);
    let shifted_filter = match filter_name {
        FourierFilter::Hann => {
            cosine_window_filter(&shifted_freq, filter_range, |index, n_minus_one| {
                let phase = std::f64::consts::PI * index as f64 / n_minus_one as f64;
                phase.sin() * phase.sin()
            })
        }
        FourierFilter::Rectangular => {
            cosine_window_filter(&shifted_freq, filter_range, |_index, _n_minus_one| 1.0)
        }
        FourierFilter::Gauss => gauss_filter(&shifted_freq, filter_range, filter_parameter),
        FourierFilter::Fermi => fermi_filter(&shifted_freq, filter_range, filter_parameter),
        FourierFilter::Nuttall => {
            cosine_window_filter(&shifted_freq, filter_range, |index, n_minus_one| {
                four_term_cosine(index, n_minus_one, 0.355768, 0.487396, 0.144232, 0.012604)
            })
        }
        FourierFilter::BlackmanNuttall => {
            cosine_window_filter(&shifted_freq, filter_range, |index, n_minus_one| {
                four_term_cosine(
                    index,
                    n_minus_one,
                    0.3635819,
                    0.4891775,
                    0.1365995,
                    0.0106411,
                )
            })
        }
        FourierFilter::BlackmanHarris => {
            cosine_window_filter(&shifted_freq, filter_range, |index, n_minus_one| {
                four_term_cosine(index, n_minus_one, 0.35875, 0.48829, 0.14128, 0.01168)
            })
        }
    };
    ifftshift(shifted_filter)
}

fn fftshifted_frequencies(len: usize, spacing: f64) -> Vec<f64> {
    let scale = len as f64 * spacing;
    let mut frequencies = (0..len)
        .map(|index| {
            let positive_bins = (len + 1) / 2;
            let bin = if index < positive_bins {
                index as isize
            } else {
                index as isize - len as isize
            };
            bin as f64 / scale
        })
        .collect::<Vec<_>>();
    fftshift(&mut frequencies);
    frequencies
}

fn cosine_window_filter<F>(frequency: &[f64], filter_range: f64, window: F) -> Vec<f64>
where
    F: Fn(usize, usize) -> f64,
{
    let len = frequency.len();
    let maxfreq = filter_range.abs();
    let idx_l = lower_bound(frequency, -maxfreq);
    let idx_u = lower_bound(frequency, maxfreq);
    let mut filter = vec![0.0; len];
    if idx_u <= idx_l {
        return filter;
    }

    let n_minus_one = idx_u - idx_l - 1;
    if n_minus_one == 0 {
        filter[idx_l] = 1.0;
        return filter;
    }

    for index in 0..=n_minus_one {
        filter[idx_l + index] = window(index, n_minus_one);
    }
    filter
}

fn gauss_filter(frequency: &[f64], filter_range: f64, sigma: f64) -> Vec<f64> {
    cosine_window_filter(frequency, filter_range, |index, n_minus_one| {
        if sigma == 0.0 {
            return if index == n_minus_one / 2 { 1.0 } else { 0.0 };
        }
        let center = n_minus_one as f64 / 2.0;
        let width = sigma * n_minus_one as f64 / 2.0;
        (-0.5 * ((index as f64 - center) / width).powi(2)).exp()
    })
}

fn fermi_filter(frequency: &[f64], filter_range: f64, sigma: f64) -> Vec<f64> {
    frequency
        .iter()
        .copied()
        .map(|freq| {
            if sigma == 0.0 {
                if freq.abs() < filter_range.abs() {
                    1.0
                } else {
                    0.0
                }
            } else {
                let exp_value = (-(freq.abs() - filter_range.abs()) / sigma).exp();
                exp_value / (1.0 + exp_value)
            }
        })
        .collect()
}

fn four_term_cosine(index: usize, n_minus_one: usize, a0: f64, a1: f64, a2: f64, a3: f64) -> f64 {
    let phase = 2.0 * std::f64::consts::PI * index as f64 / n_minus_one as f64;
    a0 - a1 * phase.cos() + a2 * (2.0 * phase).cos() - a3 * (3.0 * phase).cos()
}

fn lower_bound(values: &[f64], target: f64) -> usize {
    values.partition_point(|value| *value < target)
}

fn fftshift(values: &mut [f64]) {
    let mid = (values.len() + 1) / 2;
    values.rotate_left(mid);
}

fn ifftshift(mut values: Vec<f64>) -> Vec<f64> {
    let shift = values.len() / 2;
    values.rotate_left(shift);
    values
}
