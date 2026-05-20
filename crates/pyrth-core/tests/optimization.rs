use approx::assert_relative_eq;
use ndarray::array;
use pyrth_core::{
    impedance_residual_norm, optimize_rc_parameters, relative_l2_norm, EvaluationResult,
    FosterNetwork, FosterStepResponseModel, ImpedanceData, OptimizationConfig, PyrthError,
    RcParameterBounds, RcParameters,
};

fn foster_result(resistance: Vec<f64>, capacitance: Vec<f64>) -> EvaluationResult {
    let len = resistance.len();
    EvaluationResult {
        impedance: ImpedanceData {
            time: array![1.0],
            impedance: array![0.0],
            log_time: array![0.0],
        },
        derivative: None,
        time_spectrum: None,
        foster: Some(FosterNetwork {
            resistance: ndarray::Array1::from(resistance),
            capacitance: ndarray::Array1::from(capacitance),
            tau: ndarray::Array1::ones(len),
        }),
        cauer: None,
    }
}

#[test]
fn rc_parameters_reject_non_finite_and_non_positive_values() {
    assert!(matches!(
        RcParameters::from_slices(&[1.0, f64::NAN], &[2.0, 3.0]),
        Err(PyrthError::InvalidParameter {
            parameter: "resistance",
            ..
        })
    ));
    assert!(matches!(
        RcParameters::from_slices(&[1.0, 2.0], &[2.0, 0.0]),
        Err(PyrthError::InvalidParameter {
            parameter: "capacitance",
            ..
        })
    ));
}

#[test]
fn length_mismatch_is_rejected() {
    assert!(matches!(
        RcParameters::from_slices(&[1.0], &[2.0, 3.0]),
        Err(PyrthError::InvalidParameter {
            parameter: "capacitance",
            ..
        })
    ));
    assert!(matches!(
        relative_l2_norm(&array![1.0, 2.0], &array![1.0]),
        Err(PyrthError::InvalidParameter {
            parameter: "candidate",
            ..
        })
    ));
}

#[test]
fn flattened_parameters_roundtrip() {
    let parameters = RcParameters::from_slices(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    let flattened = parameters.to_flattened().unwrap();
    let roundtrip = RcParameters::from_flattened(&flattened).unwrap();

    assert_eq!(flattened, array![1.0, 2.0, 3.0, 4.0]);
    assert_eq!(roundtrip, parameters);
}

#[test]
fn foster_parameters_can_be_extracted_from_evaluation_result() {
    let result = foster_result(vec![1.0, 2.0], vec![3.0, 4.0]);
    let parameters = RcParameters::from_evaluation(&result).unwrap();

    assert_eq!(parameters.resistance, array![1.0, 2.0]);
    assert_eq!(parameters.capacitance, array![3.0, 4.0]);
}

#[test]
fn bounds_check_reports_containment() {
    let lower = RcParameters::from_slices(&[1.0, 2.0], &[3.0, 4.0]).unwrap();
    let upper = RcParameters::from_slices(&[2.0, 3.0], &[4.0, 5.0]).unwrap();
    let bounds = RcParameterBounds::new(lower, upper).unwrap();
    let inside = RcParameters::from_slices(&[1.5, 2.5], &[3.5, 4.5]).unwrap();
    let outside = RcParameters::from_slices(&[1.5, 3.5], &[3.5, 4.5]).unwrap();

    assert!(bounds.contains(&inside).unwrap());
    assert!(!bounds.contains(&outside).unwrap());
}

#[test]
fn relative_l2_norm_matches_known_value() {
    let norm = relative_l2_norm(&array![1.0, 2.0, 3.0], &array![1.0, 1.0, 5.0]).unwrap();

    assert_relative_eq!(norm, 5.0_f64.sqrt() / 14.0_f64.sqrt(), epsilon = 1e-14);
}

#[test]
fn impedance_residual_is_zero_for_matching_theoretical_model() {
    let model = FosterStepResponseModel::from_slices(&[2.0, 4.0], &[3.0, 5.0]).unwrap();
    let input = model.to_transient_input(1e-3, 1e3, 64).unwrap();
    let residual = impedance_residual_norm(&input, &model).unwrap();

    assert_relative_eq!(residual, 0.0, epsilon = 1e-14);
}

#[test]
fn optimize_rc_parameters_improves_shifted_initial_guess() {
    let model = FosterStepResponseModel::from_slices(&[1.0, 3.0], &[0.4, 2.0]).unwrap();
    let input = model.to_transient_input(1e-3, 1e2, 48).unwrap();
    let initial = RcParameters::from_slices(&[0.75, 3.5], &[0.65, 1.5]).unwrap();
    let lower = RcParameters::from_slices(&[0.5, 2.0], &[0.2, 1.0]).unwrap();
    let upper = RcParameters::from_slices(&[1.5, 4.0], &[1.0, 3.0]).unwrap();
    let bounds = RcParameterBounds::new(lower, upper).unwrap();
    let config = OptimizationConfig {
        max_iter: 64,
        initial_step: 0.25,
        min_step: 1e-4,
        shrink_factor: 0.5,
    };
    let initial_residual =
        impedance_residual_norm(&input, &initial.to_foster_step_response_model().unwrap()).unwrap();

    let result = optimize_rc_parameters(&input, &initial, &bounds, config).unwrap();

    assert!(result.residual_norm < initial_residual);
    assert!(result.iterations > 0);
}

#[test]
fn optimize_rc_parameters_clamps_result_to_bounds() {
    let model = FosterStepResponseModel::from_slices(&[1.0, 3.0], &[0.4, 2.0]).unwrap();
    let input = model.to_transient_input(1e-3, 1e2, 32).unwrap();
    let initial = RcParameters::from_slices(&[0.1, 10.0], &[0.1, 10.0]).unwrap();
    let lower = RcParameters::from_slices(&[0.5, 2.0], &[0.2, 1.0]).unwrap();
    let upper = RcParameters::from_slices(&[1.5, 4.0], &[1.0, 3.0]).unwrap();
    let bounds = RcParameterBounds::new(lower, upper).unwrap();

    let result = optimize_rc_parameters(
        &input,
        &initial,
        &bounds,
        OptimizationConfig {
            max_iter: 8,
            initial_step: 0.5,
            min_step: 0.01,
            shrink_factor: 0.5,
        },
    )
    .unwrap();

    assert!(bounds.contains(&result.parameters).unwrap());
}

#[test]
fn optimize_rc_parameters_rejects_invalid_config() {
    let model = FosterStepResponseModel::from_slices(&[1.0], &[0.4]).unwrap();
    let input = model.to_transient_input(1e-3, 1e1, 16).unwrap();
    let initial = RcParameters::from_slices(&[1.0], &[0.4]).unwrap();
    let lower = RcParameters::from_slices(&[0.5], &[0.2]).unwrap();
    let upper = RcParameters::from_slices(&[1.5], &[1.0]).unwrap();
    let bounds = RcParameterBounds::new(lower, upper).unwrap();

    assert!(matches!(
        optimize_rc_parameters(
            &input,
            &initial,
            &bounds,
            OptimizationConfig {
                max_iter: 0,
                initial_step: 0.1,
                min_step: 0.01,
                shrink_factor: 0.5,
            },
        ),
        Err(PyrthError::InvalidParameter {
            parameter: "max_iter",
            ..
        })
    ));
    assert!(matches!(
        optimize_rc_parameters(
            &input,
            &initial,
            &bounds,
            OptimizationConfig {
                max_iter: 8,
                initial_step: 0.1,
                min_step: 0.01,
                shrink_factor: 1.0,
            },
        ),
        Err(PyrthError::InvalidParameter {
            parameter: "shrink_factor",
            ..
        })
    ));
}
