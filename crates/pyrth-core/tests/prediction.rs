use ndarray::Array1;
use pyrth_core::{predict_temperature, EvaluationParams, TransientInput};

#[test]
fn standard_temperature_prediction_returns_finite_series() {
    let input = TransientInput::from_pairs([
        (1e-6, 0.1),
        (1e-5, 0.2),
        (1e-4, 0.3),
        (1e-3, 0.42),
        (1e-2, 0.5),
    ])
    .unwrap();
    let power = power_input(&[(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)]);
    let mut params = EvaluationParams::default();
    params.log_time_size = 16;
    params.minimum_window_size = 2;
    params.min_index = 1;
    params.bay_steps = 2;
    params.calc_struc = false;

    let result = predict_temperature(input, power, &params, 1e-3).unwrap();

    assert!(result.base.derivative.is_some());
    assert!(!result.predicted_temperature.is_empty());
    assert_eq!(result.lin_time.len(), result.predicted_temperature.len());
    assert_eq!(result.lin_time.len(), result.power_function_int.len());
    assert!(result
        .predicted_temperature
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .power_function_int
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .impulse_response_int
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn prediction_rejects_invalid_sampling_period() {
    let input = TransientInput::from_pairs([(1e-6, 0.1), (1e-5, 0.2), (1e-4, 0.3)]).unwrap();
    let power = power_input(&[(0.0, 0.0), (0.02, 1.0)]);
    let params = EvaluationParams::default();

    let err = predict_temperature(input, power, &params, 0.0).unwrap_err();

    assert!(err.to_string().contains("lin_sampling_period"));
}

fn power_input(pairs: &[(f64, f64)]) -> TransientInput {
    let (time, value): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
    TransientInput {
        time: Array1::from(time),
        value: Array1::from(value),
    }
}
