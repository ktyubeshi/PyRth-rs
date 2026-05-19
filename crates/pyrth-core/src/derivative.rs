use ndarray::Array1;

use crate::{config::EvaluationParams, error::Result, evaluation::DerivativeResult};

const ESTIMATOR_TIE_EPSILON: f64 = 1e-18;

pub fn z_fit_deriv(
    impedance: &Array1<f64>,
    log_time: &Array1<f64>,
    params: &EvaluationParams,
) -> Result<DerivativeResult> {
    let log_time_size = params.log_time_size;
    let log_time_delta = (log_time[log_time.len() - 1] - log_time[0]) / log_time_size as f64;

    let log_time_interp = Array1::from_iter(
        (0..log_time_size).map(|index| log_time[0] + index as f64 * log_time_delta),
    );

    let lent = log_time.len();
    let leni = log_time_interp.len();
    let pad_number_pre = (log_time_size as f64 * params.pad_factor_pre) as usize;
    let pad_number_after = (log_time_size as f64 * params.pad_factor_after) as usize;
    let leng = leni + pad_number_pre + pad_number_after;

    let mut global_weight = Vec::with_capacity(lent);
    for index in 0..lent - 1 {
        global_weight.push(log_time[index + 1] - log_time[index]);
    }
    global_weight.push(log_time[lent - 1] - log_time[lent - 2]);

    let mut imp_smooth = vec![0.0; leni];
    let mut imp_deriv_interp = vec![0.0; leng];

    let full_window = [-params.window_increment, 0.0, params.window_increment];
    let lower_window = [0.0, params.window_increment];
    let upper_window = [-params.window_increment, 0.0];

    let mut best_window_length: Option<f64> = None;
    let mut n_data = pad_number_pre;

    for (n, t_val) in log_time_interp.iter().copied().enumerate() {
        let mut best_poly_val = 1e200;
        let mut best_diff_val = 1e200;
        let mut best_estimator = 1e200;
        let last_window_length = best_window_length.unwrap_or(0.0);

        let initial_steps;
        let bw_steps: &[f64] = if let Some(last) = best_window_length {
            if params.minimum_window_length < last && last < params.maximum_window_length {
                &full_window
            } else if last >= params.maximum_window_length {
                &upper_window
            } else if last <= params.minimum_window_length {
                &lower_window
            } else {
                &full_window
            }
        } else {
            let steps = ((params.maximum_window_length - params.minimum_window_length)
                / params.window_increment) as usize
                + 1;
            initial_steps = (0..steps)
                .map(|index| params.minimum_window_length + index as f64 * params.window_increment)
                .collect::<Vec<_>>();
            &initial_steps
        };

        for bw in bw_steps {
            let mut window_length = last_window_length + bw;
            window_length = window_length
                .max(params.minimum_window_length)
                .min(params.maximum_window_length);

            let index = search_sorted(log_time.as_slice().expect("contiguous log_time"), t_val)
                .max(params.min_index);
            let center_time = log_time[index];

            let mut up_bound = search_sorted(
                log_time.as_slice().expect("contiguous log_time"),
                t_val + window_length,
            ) as isize
                + 1;
            let mut low_bound = search_sorted(
                log_time.as_slice().expect("contiguous log_time"),
                t_val - window_length,
            ) as isize;

            while up_bound - low_bound < params.minimum_window_size as isize {
                up_bound += 1;
                low_bound -= 1;
            }
            while low_bound < 0 {
                up_bound += 1;
                low_bound += 1;
            }
            while up_bound > lent as isize {
                up_bound -= 1;
                low_bound -= 1;
            }

            let low = low_bound as usize;
            let up = up_bound as usize;
            let t_frame = &log_time.as_slice().expect("contiguous log_time")[low..up];
            let z_frame = &impedance.as_slice().expect("contiguous impedance")[low..up];

            let center_index = search_sorted(t_frame, center_time);
            let max_dist_cd1 = center_time - t_frame[0];
            let max_dist_cd2 = t_frame[t_frame.len() - 1] - center_time;
            let max_dist = max_dist_cd1.max(max_dist_cd2);

            let mut weights = Vec::with_capacity(t_frame.len());
            for (offset, time) in t_frame.iter().copied().enumerate() {
                let frame_weight =
                    (1.0 - ((time - center_time) / max_dist).abs().powf(3.0)).powf(3.0);
                weights.push(frame_weight * global_weight[low + offset]);
            }

            let coefs = polyfit(t_frame, z_frame, &weights);
            let poly_value = coefs.slope * t_val + coefs.intercept;

            let var = params.expected_var.powf(2.0);
            let dif_spread = 0.1;

            let mut z_frame_copy = z_frame.to_vec();
            z_frame_copy[center_index] = impedance[index] - dif_spread * poly_value;
            let coefs_lower = polyfit(t_frame, &z_frame_copy, &weights);
            let polval_lower = coefs_lower.slope * t_val + coefs_lower.intercept;

            z_frame_copy[center_index] = impedance[index] + dif_spread * poly_value;
            let coefs_upper = polyfit(t_frame, &z_frame_copy, &weights);
            let polval_upper = coefs_upper.slope * t_val + coefs_upper.intercept;

            let diff_term = ((polval_upper - polval_lower) / (2.0 * dif_spread * poly_value)).abs();
            let estimator = poly_value.powf(2.0) - 2.0 * z_frame[center_index] * poly_value
                + 2.0 * var * diff_term;

            // Python/Numba can keep the earlier window when estimator values
            // differ only by roundoff noise; preserve that stable tie break.
            if estimator < best_estimator - ESTIMATOR_TIE_EPSILON {
                best_estimator = estimator;
                best_poly_val = poly_value;
                best_diff_val = coefs.slope;
                best_window_length = Some(window_length);
            }
        }

        imp_smooth[n] = best_poly_val;
        imp_deriv_interp[n_data] = if best_diff_val > 0.0 {
            best_diff_val
        } else {
            0.0
        };
        n_data += 1;
    }

    let time_start = log_time_interp[0] - pad_number_pre as f64 * log_time_delta;
    let time_stop =
        log_time_interp[log_time_interp.len() - 1] + pad_number_after as f64 * log_time_delta;

    let mut log_time_pad = Vec::with_capacity(leng);
    for index in 0..pad_number_pre {
        log_time_pad.push(
            time_start + index as f64 * (log_time_interp[0] - time_start) / pad_number_pre as f64,
        );
    }
    log_time_pad.extend(log_time_interp.iter().copied());
    if pad_number_after > 0 {
        let start = log_time_interp[log_time_interp.len() - 1] + log_time_delta;
        for index in 0..pad_number_after {
            let value = if pad_number_after == 1 {
                time_stop
            } else {
                start + index as f64 * (time_stop - start) / (pad_number_after - 1) as f64
            };
            log_time_pad.push(value);
        }
    }

    let imp_smooth_full = Array1::from_iter(log_time.iter().copied().map(|time| {
        interp_linear(
            log_time_interp.as_slice().expect("contiguous interp"),
            &imp_smooth,
            time,
        )
    }));

    Ok(DerivativeResult {
        imp_smooth: Array1::from(imp_smooth),
        imp_deriv_interp: Array1::from(imp_deriv_interp),
        log_time_interp,
        log_time_pad: Array1::from(log_time_pad),
        log_time_delta,
        imp_smooth_full,
    })
}

struct LinearCoefficients {
    slope: f64,
    intercept: f64,
}

fn polyfit(x: &[f64], y: &[f64], weights: &[f64]) -> LinearCoefficients {
    let weight_sum: f64 = weights.iter().sum();
    let weighted_mean_x = weights
        .iter()
        .zip(x)
        .map(|(weight, x)| weight * x)
        .sum::<f64>()
        / weight_sum;
    let weighted_mean_y = weights
        .iter()
        .zip(y)
        .map(|(weight, y)| weight * y)
        .sum::<f64>()
        / weight_sum;

    let numerator = weights
        .iter()
        .zip(x)
        .zip(y)
        .map(|((weight, x), y)| weight * (x - weighted_mean_x) * (y - weighted_mean_y))
        .sum::<f64>();
    let denominator = weights
        .iter()
        .zip(x)
        .map(|(weight, x)| weight * (x - weighted_mean_x).powf(2.0))
        .sum::<f64>();

    LinearCoefficients {
        slope: numerator / denominator,
        intercept: weighted_mean_y - numerator / denominator * weighted_mean_x,
    }
}

fn search_sorted(values: &[f64], target: f64) -> usize {
    values.partition_point(|value| *value < target)
}

fn interp_linear(x: &[f64], y: &[f64], target: f64) -> f64 {
    let upper = search_sorted(x, target);
    if upper == 0 {
        return y[0];
    }
    if upper >= x.len() {
        return y[y.len() - 1];
    }
    let lower = upper - 1;
    let fraction = (target - x[lower]) / (x[upper] - x[lower]);
    y[lower] + fraction * (y[upper] - y[lower])
}
