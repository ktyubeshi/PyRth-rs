use ndarray::{Array1, Array2};
use rustfft::{num_complex::Complex, FftPlanner};

use crate::{config::EvaluationParams, evaluation::DerivativeResult};

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

pub fn time_spectrum_fourier(derivative: &DerivativeResult) -> Array1<f64> {
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
    let mut deconvolved = signal
        .into_iter()
        .zip(kernel)
        .map(|(signal_value, kernel_value)| {
            if kernel_value.norm_sqr() <= epsilon * epsilon {
                Complex::new(0.0, 0.0)
            } else {
                signal_value / kernel_value
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

fn weight_z(value: f64) -> f64 {
    (value - value.exp()).exp()
}
