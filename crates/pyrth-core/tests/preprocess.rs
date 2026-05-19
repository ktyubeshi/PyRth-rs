use approx::assert_relative_eq;
use pyrth_core::{
    make_impedance_data, parse_t3ster_calibration_text, parse_t3ster_power_step,
    parse_t3ster_raw_text, t3ster_raw_to_temperature_input, EvaluationParams, InputMode,
    PyrthError, TransientInput,
};

fn temperature_params() -> EvaluationParams {
    EvaluationParams {
        input_mode: InputMode::Temperature,
        ..EvaluationParams::default()
    }
}

#[test]
fn temperature_extrapolation_prepends_sqrt_time_fit() {
    let input =
        TransientInput::from_pairs([(1.0, 12.0), (4.0, 14.0), (9.0, 16.0), (16.0, 18.0)]).unwrap();
    let mut params = temperature_params();
    params.extrapolate = true;
    params.lower_fit_limit = Some(4.0);
    params.upper_fit_limit = Some(16.0);

    let data = make_impedance_data(input, &params).unwrap();

    assert_eq!(data.time.len(), 18);
    assert_relative_eq!(data.time[0], 1e-4, epsilon = 1e-14);
    assert_relative_eq!(data.time[15], 4.0, epsilon = 1e-12);
    assert_relative_eq!(data.time[16], 9.0, epsilon = 1e-12);
    assert_relative_eq!(data.time[17], 16.0, epsilon = 1e-12);

    for (time, impedance) in data.time.iter().zip(data.impedance.iter()) {
        assert_relative_eq!(*impedance, -2.0 * time.sqrt(), epsilon = 1e-12);
    }
}

#[test]
fn temperature_extrapolation_requires_fit_limits() {
    let input = TransientInput::from_pairs([(1.0, 12.0), (4.0, 14.0), (9.0, 16.0)]).unwrap();
    let mut params = temperature_params();
    params.extrapolate = true;

    let err = make_impedance_data(input, &params).unwrap_err();

    assert!(matches!(
        err,
        PyrthError::InvalidParameter {
            parameter: "lower_fit_limit",
            ..
        }
    ));
}

#[test]
fn temperature_default_keeps_existing_cut_and_average_path() {
    let input =
        TransientInput::from_pairs([(1.0, 20.0), (2.0, 18.0), (4.0, 16.0), (8.0, 14.0)]).unwrap();
    let mut params = temperature_params();
    params.data_cut_lower = 1;
    params.data_cut_upper = Some(3);
    params.temp_0_avg_range = (0, 1);

    let data = make_impedance_data(input, &params).unwrap();

    assert_eq!(data.time.as_slice().unwrap(), &[1.0, 3.0]);
    assert_eq!(data.impedance.as_slice().unwrap(), &[0.0, 2.0]);
}

#[test]
fn t3ster_text_parsers_load_legacy_sample() {
    let raw = parse_t3ster_raw_text(include_str!(
        "../../../tests/data/t3ster/T25_I-m5m-I-h600m_100s.raw"
    ))
    .unwrap();
    let calibration =
        parse_t3ster_calibration_text(include_str!("../../../tests/data/t3ster/calib.tco"))
            .unwrap();
    let power = parse_t3ster_power_step(include_str!(
        "../../../tests/data/t3ster/T25_I-m5m-I-h600m_100s.pwr"
    ))
    .unwrap();

    assert_eq!(raw.time_microseconds.len(), raw.adc_count.len());
    assert!(raw.time_microseconds.len() > 100);
    assert_relative_eq!(raw.lsb, 2.4414e-5, epsilon = 1e-12);
    assert_relative_eq!(raw.uref, 2.5706, epsilon = 1e-12);
    assert_eq!(calibration.len(), 5);
    assert_relative_eq!(power, 1.754057, epsilon = 1e-12);
}

#[test]
fn t3ster_raw_converts_to_temperature_input() {
    let raw = parse_t3ster_raw_text(include_str!(
        "../../../tests/data/t3ster/T25_I-m5m-I-h600m_100s.raw"
    ))
    .unwrap();
    let calibration =
        parse_t3ster_calibration_text(include_str!("../../../tests/data/t3ster/calib.tco"))
            .unwrap();

    let input = t3ster_raw_to_temperature_input(&raw, &calibration, 2).unwrap();

    assert_eq!(input.time.len(), input.value.len());
    assert!(input.time.len() > 100);
    assert_relative_eq!(input.time[0], 1e-6, epsilon = 1e-12);
    assert!(input
        .time
        .windows(2)
        .into_iter()
        .all(|pair| pair[0] < pair[1]));
    assert!(input.value.iter().all(|value| value.is_finite()));
}
