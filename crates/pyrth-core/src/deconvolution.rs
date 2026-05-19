use ndarray::{Array1, Array2};

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
