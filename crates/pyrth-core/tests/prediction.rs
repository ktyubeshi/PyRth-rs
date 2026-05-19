use ndarray::Array1;
use pyrth_core::{
    foster_impulse_response_on, predict_temperature, predict_temperature_from_optimization_result,
    predict_temperature_from_rc_parameters, EvaluationParams, OptimizationResult, RcParameters,
    TransientInput,
};

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

#[test]
fn rc_temperature_prediction_matches_single_foster_step_response() {
    let parameters = RcParameters::from_slices(&[2.0], &[0.005]).unwrap();
    let reference_time = Array1::from_iter((0..=200).map(|index| index as f64 * 5e-4));
    let power = power_input(&[(0.0, 1.0), (0.08, 1.0)]);

    let result =
        predict_temperature_from_rc_parameters(&power, &parameters, &reference_time, 5e-4).unwrap();

    let sample_index = result
        .lin_time
        .iter()
        .position(|time| *time >= 0.02)
        .unwrap();
    let sample_time = result.lin_time[sample_index];
    let expected = parameters.impedance_at(sample_time).unwrap();

    assert!((result.predicted_temperature[sample_index] - expected).abs() < 0.04);
    assert_eq!(result.lin_time.len(), result.predicted_temperature.len());
    assert_eq!(result.lin_time.len(), result.power_function_int.len());
    assert!(result
        .impulse_response_int
        .iter()
        .all(|value| value.is_finite()));
}

#[test]
fn foster_impulse_response_uses_rc_time_constants() {
    let parameters = RcParameters::from_slices(&[2.0], &[0.005]).unwrap();
    let time = Array1::from(vec![0.0, 0.01]);

    let impulse = foster_impulse_response_on(&parameters, &time).unwrap();

    assert!((impulse[0] - 200.0).abs() < 1e-12);
    assert!((impulse[1] - 200.0 / std::f64::consts::E).abs() < 1e-12);
}

#[test]
fn optimization_result_prediction_reuses_rc_parameters() {
    let parameters = RcParameters::from_slices(&[2.0], &[0.005]).unwrap();
    let optimization = OptimizationResult {
        parameters: parameters.clone(),
        residual_norm: 0.0,
        iterations: 1,
    };
    let reference_time = Array1::from_iter((0..=200).map(|index| index as f64 * 5e-4));
    let power = power_input(&[(0.0, 1.0), (0.08, 1.0)]);

    let from_parameters =
        predict_temperature_from_rc_parameters(&power, &parameters, &reference_time, 5e-4).unwrap();
    let from_optimization =
        predict_temperature_from_optimization_result(&power, &optimization, &reference_time, 5e-4)
            .unwrap();

    assert_eq!(from_parameters.lin_time, from_optimization.lin_time);
    assert_eq!(
        from_parameters.predicted_temperature,
        from_optimization.predicted_temperature
    );
}

fn power_input(pairs: &[(f64, f64)]) -> TransientInput {
    let (time, value): (Vec<_>, Vec<_>) = pairs.iter().copied().unzip();
    TransientInput {
        time: Array1::from(time),
        value: Array1::from(value),
    }
}
