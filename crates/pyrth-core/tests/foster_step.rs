use approx::assert_relative_eq;
use pyrth_core::{foster_step_response_input, FosterStepResponseModel, PyrthError};

#[test]
fn single_rc_matches_known_step_response_values() {
    let input = foster_step_response_input(&[2.0], &[3.0], 1.0, 6.0, 2).unwrap();

    assert_relative_eq!(
        input.value[0],
        2.0 * (1.0_f64 - (-1.0_f64 / 6.0_f64).exp()),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        input.value[1],
        2.0 * (1.0_f64 - (-1.0_f64).exp()),
        epsilon = 1e-14
    );
}

#[test]
fn multiple_rc_step_response_is_monotonic() {
    let input =
        foster_step_response_input(&[1.0, 3.0, 5.0], &[0.5, 2.0, 4.0], 1e-3, 1e3, 128).unwrap();

    assert_eq!(input.time.len(), 128);
    assert!(input
        .time
        .windows(2)
        .into_iter()
        .all(|pair| pair[0] < pair[1]));
    assert!(input
        .value
        .windows(2)
        .into_iter()
        .all(|pair| pair[0] <= pair[1]));
    assert!(input.value[input.value.len() - 1] <= 9.0);
}

#[test]
fn invalid_input_is_rejected() {
    assert!(matches!(
        foster_step_response_input(&[], &[], 1e-6, 1.0, 16),
        Err(PyrthError::EmptyInput)
    ));
    assert!(matches!(
        foster_step_response_input(&[1.0], &[1.0, 2.0], 1e-6, 1.0, 16),
        Err(PyrthError::LengthMismatch { .. })
    ));
    assert!(matches!(
        foster_step_response_input(&[0.0], &[1.0], 1e-6, 1.0, 16),
        Err(PyrthError::InvalidParameter {
            parameter: "resistances",
            ..
        })
    ));
    assert!(matches!(
        foster_step_response_input(&[1.0], &[-1.0], 1e-6, 1.0, 16),
        Err(PyrthError::InvalidParameter {
            parameter: "capacitances",
            ..
        })
    ));
    assert!(matches!(
        foster_step_response_input(&[1.0], &[1.0], 1.0, 1.0, 16),
        Err(PyrthError::InvalidParameter {
            parameter: "time_range",
            ..
        })
    ));
    assert!(matches!(
        foster_step_response_input(&[1.0], &[1.0], 1e-6, 1.0, 1),
        Err(PyrthError::InvalidParameter {
            parameter: "time_size",
            ..
        })
    ));
}

#[test]
fn model_can_evaluate_individual_time_points() {
    let model = FosterStepResponseModel::from_slices(&[2.0, 4.0], &[3.0, 5.0]).unwrap();
    let expected = 2.0 * (1.0_f64 - (-1.0_f64 / 6.0_f64).exp())
        + 4.0 * (1.0_f64 - (-1.0_f64 / 20.0_f64).exp());

    assert_relative_eq!(
        model.step_response_at(1.0).unwrap(),
        expected,
        epsilon = 1e-14
    );
}
