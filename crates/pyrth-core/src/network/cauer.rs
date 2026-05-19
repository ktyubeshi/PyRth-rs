use ndarray::Array1;

use crate::{config::EvaluationParams, evaluation::CauerNetwork};

pub fn cauer_from_foster_lanczos(
    foster_capacitance: &Array1<f64>,
    foster_resistance: &Array1<f64>,
    params: &EvaluationParams,
) -> CauerNetwork {
    let (mut resistance, mut capacitance) = lanczos_inner(
        foster_capacitance.as_slice().unwrap(),
        foster_resistance.as_slice().unwrap(),
    );

    if params.blockwise_sum_width > 1 {
        (resistance, capacitance) =
            reduceat_blocks(&resistance, &capacitance, params.blockwise_sum_width);
    }

    let cumulative_resistance = cumulative_sum(&resistance);
    let cumulative_capacitance = cumulative_sum(&capacitance);
    let differential_structure =
        differential_structure(&cumulative_resistance, &cumulative_capacitance);

    CauerNetwork {
        resistance: Array1::from(resistance),
        capacitance: Array1::from(capacitance),
        cumulative_resistance: Array1::from(cumulative_resistance),
        cumulative_capacitance: Array1::from(cumulative_capacitance),
        differential_structure: Array1::from(differential_structure),
    }
}

fn lanczos_inner(cap_fost: &[f64], res_fost: &[f64]) -> (Vec<f64>, Vec<f64>) {
    let c_diag = cap_fost;
    let k_diag = res_fost
        .iter()
        .copied()
        .map(|resistance| 1.0 / resistance)
        .collect::<Vec<_>>();

    let g = vec![1.0; c_diag.len()];
    let mut r = g
        .iter()
        .zip(c_diag)
        .map(|(value, capacitance)| value / capacitance)
        .collect::<Vec<_>>();

    let beta = dot(&r, &g).sqrt();
    let mut v = vec![0.0; r.len()];

    let mut resistance = Vec::new();
    let mut capacitance = Vec::new();

    let mut u = r.iter().map(|value| value / beta).collect::<Vec<_>>();
    let alpha = -dot_k(&u, &k_diag);
    r = next_residual(&k_diag, c_diag, &u, beta, &v, alpha);
    let mut beta_next = dot_c(&r, c_diag).sqrt();
    v = u;

    capacitance.push(1.0 / beta.powf(2.0));
    resistance.push(-1.0 / (alpha * capacitance[0]));

    let mut cap_prev = capacitance[0];
    let mut res_prev = resistance[0];
    let mut cap_sum = 0.0;

    while cap_sum < 1e4 {
        u = r.iter().map(|value| value / beta_next).collect::<Vec<_>>();
        let alpha = -dot_k(&u, &k_diag);
        r = next_residual(&k_diag, c_diag, &u, beta_next, &v, alpha);
        let beta_prev = beta_next;
        beta_next = dot_c(&r, c_diag).sqrt();
        v = u;

        let cap_next = 1.0 / (beta_prev.powf(2.0) * res_prev.powf(2.0) * cap_prev);
        let res_next = -1.0 / (alpha * cap_next + 1.0 / res_prev);

        if res_next <= 0.0 || cap_next <= 0.0 {
            break;
        }

        capacitance.push(cap_next);
        resistance.push(res_next);
        cap_sum += cap_next;

        cap_prev = cap_next;
        res_prev = res_next;
    }

    (resistance, capacitance)
}

fn next_residual(
    k_diag: &[f64],
    c_diag: &[f64],
    u: &[f64],
    beta: f64,
    v: &[f64],
    alpha: f64,
) -> Vec<f64> {
    k_diag
        .iter()
        .zip(c_diag)
        .zip(u)
        .zip(v)
        .map(|(((k, c), u), v)| (-((k + alpha * c) * u) - beta * c * v) / c)
        .collect()
}

fn dot(left: &[f64], right: &[f64]) -> f64 {
    left.iter()
        .zip(right)
        .map(|(left, right)| left * right)
        .sum()
}

fn dot_k(vector: &[f64], k_diag: &[f64]) -> f64 {
    vector
        .iter()
        .zip(k_diag)
        .map(|(value, k)| value * k * value)
        .sum()
}

fn dot_c(vector: &[f64], c_diag: &[f64]) -> f64 {
    vector
        .iter()
        .zip(c_diag)
        .map(|(value, c)| value * c * value)
        .sum()
}

fn reduceat_blocks(resistance: &[f64], capacitance: &[f64], width: usize) -> (Vec<f64>, Vec<f64>) {
    let block_count = resistance.len() / width;
    let mut block_resistance = Vec::with_capacity(block_count);
    let mut block_capacitance = Vec::with_capacity(block_count);

    for block in 0..block_count {
        let start = block * width;
        let end = if block + 1 == block_count {
            resistance.len()
        } else {
            (block + 1) * width
        };
        block_resistance.push(resistance[start..end].iter().sum());
        block_capacitance.push(capacitance[start..end].iter().sum());
    }

    (block_resistance, block_capacitance)
}

fn cumulative_sum(values: &[f64]) -> Vec<f64> {
    let mut total = 0.0;
    values
        .iter()
        .map(|value| {
            total += value;
            total
        })
        .collect()
}

fn differential_structure(resistance: &[f64], capacitance: &[f64]) -> Vec<f64> {
    resistance
        .windows(2)
        .zip(capacitance.windows(2))
        .map(|(resistance, capacitance)| {
            let delta_res = resistance[1] - resistance[0];
            if delta_res == 0.0 {
                0.0
            } else {
                (capacitance[1] - capacitance[0]) / delta_res
            }
        })
        .collect()
}
