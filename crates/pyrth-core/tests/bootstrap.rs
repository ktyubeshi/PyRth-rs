use pyrth_core::{
    bootstrap_from_impedance_data, bootstrap_from_theoretical, theoretical_impedance_input,
    DeconvMode, EvaluationParams, TheoreticalModel,
};

#[test]
fn bootstrap_from_theoretical_returns_deterministic_finite_means() {
    let model = TheoreticalModel::from_slices(&[0.5, 1.5], &[0.02, 0.2]).unwrap();
    let mut params = EvaluationParams::default();
    params.deconv_mode = DeconvMode::Fourier;
    params.log_time_size = 64;
    params.minimum_window_size = 8;
    params.min_index = 1;
    params.calc_struc = false;

    let result = bootstrap_from_theoretical(&model, 1e-5, 10.0, 96, 4, 1e-6, &params, 42).unwrap();
    let repeated =
        bootstrap_from_theoretical(&model, 1e-5, 10.0, 96, 4, 1e-6, &params, 42).unwrap();

    assert_eq!(result.successful_repetitions, 4);
    assert_eq!(result.impedance_mean.len(), 96);
    assert_eq!(result.impedance_p10.len(), 96);
    assert_eq!(result.impedance_median.len(), 96);
    assert_eq!(result.impedance_p90.len(), 96);
    assert_eq!(result.time_spectrum_mean.len(), 64);
    assert_eq!(result.time_spectrum_p10.len(), 64);
    assert_eq!(result.time_spectrum_median.len(), 64);
    assert_eq!(result.time_spectrum_p90.len(), 64);
    assert!(result.impedance_mean.iter().all(|value| value.is_finite()));
    assert!(result.impedance_p10.iter().all(|value| value.is_finite()));
    assert!(result
        .impedance_median
        .iter()
        .all(|value| value.is_finite()));
    assert!(result.impedance_p90.iter().all(|value| value.is_finite()));
    assert!(result
        .time_spectrum_mean
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .time_spectrum_p10
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .time_spectrum_median
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .time_spectrum_p90
        .iter()
        .all(|value| value.is_finite()));
    for index in 0..result.impedance_mean.len() {
        assert!(result.impedance_p10[index] <= result.impedance_median[index]);
        assert!(result.impedance_median[index] <= result.impedance_p90[index]);
    }
    assert_eq!(result, repeated);
}

#[test]
fn bootstrap_from_impedance_data_returns_seeded_deterministic_bands() {
    let input = theoretical_impedance_input(&[0.5, 1.5], &[0.02, 0.2], 1e-5, 10.0, 96).unwrap();
    let mut params = EvaluationParams::default();
    params.deconv_mode = DeconvMode::Fourier;
    params.log_time_size = 64;
    params.minimum_window_size = 8;
    params.min_index = 1;
    params.calc_struc = false;

    let result = bootstrap_from_impedance_data(&input, 5, 1e-6, &params, 7).unwrap();
    let repeated = bootstrap_from_impedance_data(&input, 5, 1e-6, &params, 7).unwrap();

    assert_eq!(result, repeated);
    assert_eq!(result.successful_repetitions, 5);
    assert_eq!(result.impedance_mean.len(), input.value.len());
    assert_eq!(result.time_spectrum_mean.len(), params.log_time_size);
    assert!(result.impedance_mean.iter().all(|value| value.is_finite()));
    assert!(result
        .time_spectrum_mean
        .iter()
        .all(|value| value.is_finite()));
    for index in 0..result.impedance_mean.len() {
        assert!(result.impedance_p10[index] <= result.impedance_median[index]);
        assert!(result.impedance_median[index] <= result.impedance_p90[index]);
    }
    for index in 0..result.time_spectrum_mean.len() {
        assert!(result.time_spectrum_p10[index] <= result.time_spectrum_median[index]);
        assert!(result.time_spectrum_median[index] <= result.time_spectrum_p90[index]);
    }
}
