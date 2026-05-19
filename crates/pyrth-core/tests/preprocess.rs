use approx::assert_relative_eq;
use pyrth_core::{make_impedance_data, EvaluationParams, InputMode, PyrthError, TransientInput};

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
