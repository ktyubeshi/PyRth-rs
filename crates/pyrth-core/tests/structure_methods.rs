use approx::assert_relative_eq;
use ndarray::array;
#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
use pyrth_core::network::{
    boor_golub_cauer_mpfr_raw, cauer_from_foster_boor_golub_mpfr,
    cauer_from_foster_khatwani_mpfr, cauer_from_foster_poly_long_mpfr,
    cauer_from_foster_sobhy_mpfr,
};
use pyrth_core::{
    error::PyrthError,
    network::{cauer_from_foster_poly_long_f64, foster_impedance_rational_f64},
    DeconvMode, EvaluationParams, StructureMethod, TransientInput,
};

#[test]
fn foster_impedance_rational_f64_matches_two_parallel_branches() {
    let resistance = array![2.0, 3.0];
    let capacitance = array![5.0, 7.0];

    let rational = foster_impedance_rational_f64(&resistance, &capacitance).unwrap();

    assert_relative_eq!(rational.numerator[0], 5.0);
    assert_relative_eq!(rational.numerator[1], 72.0);
    assert_relative_eq!(rational.denominator[0], 1.0);
    assert_relative_eq!(rational.denominator[1], 31.0);
    assert_relative_eq!(rational.denominator[2], 210.0);
}

#[test]
fn poly_long_f64_converts_single_foster_branch() {
    let resistance = array![2.0];
    let capacitance = array![3.0];

    let cauer = cauer_from_foster_poly_long_f64(&resistance, &capacitance).unwrap();

    assert_eq!(cauer.resistance.len(), 1);
    assert_eq!(cauer.capacitance.len(), 1);
    assert_relative_eq!(cauer.resistance[0], 2.0);
    assert_relative_eq!(cauer.capacitance[0], 3.0);
    assert_relative_eq!(cauer.cumulative_resistance[0], 2.0);
    assert_relative_eq!(cauer.cumulative_capacitance[0], 3.0);
    assert!(cauer.differential_structure.is_empty());
}

#[test]
fn poly_long_f64_preserves_total_resistance_for_two_branches() {
    let resistance = array![2.0, 3.0];
    let capacitance = array![5.0, 7.0];

    let cauer = cauer_from_foster_poly_long_f64(&resistance, &capacitance).unwrap();

    assert_eq!(cauer.resistance.len(), 2);
    assert_eq!(cauer.capacitance.len(), 2);
    assert!(cauer.resistance.iter().all(|value| *value > 0.0));
    assert!(cauer.capacitance.iter().all(|value| *value > 0.0));
    assert_relative_eq!(cauer.cumulative_resistance[1], 5.0, epsilon = 1e-12);
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn poly_long_mpfr_converts_single_foster_branch() {
    let resistance = array![2.0];
    let capacitance = array![3.0];

    let cauer = cauer_from_foster_poly_long_mpfr(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer.resistance.len(), 1);
    assert_eq!(cauer.capacitance.len(), 1);
    assert_relative_eq!(cauer.resistance[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(cauer.capacitance[0], 3.0, epsilon = 1e-12);
    assert_relative_eq!(cauer.cumulative_resistance[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(cauer.cumulative_capacitance[0], 3.0, epsilon = 1e-12);
    assert!(cauer.differential_structure.is_empty());
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn poly_long_mpfr_preserves_total_resistance_for_two_branches() {
    let resistance = array![2.0, 3.0];
    let capacitance = array![5.0, 7.0];

    let cauer = cauer_from_foster_poly_long_mpfr(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer.resistance.len(), 2);
    assert_eq!(cauer.capacitance.len(), 2);
    assert!(cauer
        .resistance
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert!(cauer
        .capacitance
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
    assert_relative_eq!(cauer.cumulative_resistance[1], 5.0, epsilon = 1e-12);
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn sobhy_mpfr_converts_single_foster_branch() {
    let resistance = array![2.0];
    let capacitance = array![3.0];

    let cauer = cauer_from_foster_sobhy_mpfr(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer.resistance.len(), 1);
    assert_eq!(cauer.capacitance.len(), 1);
    assert_relative_eq!(cauer.resistance[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(cauer.capacitance[0], 3.0, epsilon = 1e-12);
    assert!(cauer.differential_structure.is_empty());
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn khatwani_mpfr_converts_single_foster_branch() {
    let resistance = array![2.0];
    let capacitance = array![3.0];

    let cauer = cauer_from_foster_khatwani_mpfr(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer.resistance.len(), 1);
    assert_eq!(cauer.capacitance.len(), 1);
    assert_relative_eq!(cauer.resistance[0], 2.0, epsilon = 1e-12);
    assert_relative_eq!(cauer.capacitance[0], 3.0, epsilon = 1e-12);
    assert!(cauer.differential_structure.is_empty());
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn boor_golub_mpfr_raw_preserves_python_shape_for_two_branches() {
    let resistance = array![2.0, 3.0];
    let capacitance = array![5.0, 7.0];

    let (cauer_resistance, cauer_capacitance) =
        boor_golub_cauer_mpfr_raw(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer_resistance.len(), 2);
    assert_eq!(cauer_capacitance.len(), 2);
    assert!(cauer_resistance[0].is_finite() && cauer_resistance[0] > 0.0);
    assert_relative_eq!(cauer_resistance[1], 0.0, epsilon = 1e-12);
    assert!(cauer_capacitance
        .iter()
        .all(|value| value.is_finite() && *value > 0.0));
}

#[cfg(all(feature = "mpfr", not(target_env = "msvc")))]
#[test]
fn boor_golub_mpfr_cauer_drops_python_trailing_zero_resistance() {
    let resistance = array![2.0, 3.0];
    let capacitance = array![5.0, 7.0];

    let cauer = cauer_from_foster_boor_golub_mpfr(&resistance, &capacitance, 250).unwrap();

    assert_eq!(cauer.resistance.len(), 1);
    assert_eq!(cauer.capacitance.len(), 1);
    assert!(cauer.resistance[0].is_finite() && cauer.resistance[0] > 0.0);
    assert!(cauer.capacitance[0].is_finite() && cauer.capacitance[0] > 0.0);
    assert_relative_eq!(cauer.cumulative_resistance[0], cauer.resistance[0]);
    assert_relative_eq!(cauer.cumulative_capacitance[0], cauer.capacitance[0]);
    assert!(cauer.differential_structure.is_empty());
}

#[cfg(not(feature = "mpfr"))]
#[test]
fn boor_golub_stays_explicitly_unsupported_in_evaluate() {
    let input = TransientInput::from_pairs([(1e-6, 0.0), (1e-5, 0.1), (1e-4, 0.2)]).unwrap();

    let unsupported_methods = [StructureMethod::BoorGolub];

    for structure_method in unsupported_methods {
        let mut params = EvaluationParams {
            deconv_mode: DeconvMode::Bayesian,
            structure_method,
            log_time_size: 8,
            bay_steps: 2,
            min_index: 1,
            minimum_window_size: 2,
            calc_struc: true,
            ..EvaluationParams::default()
        };
        params.minimum_window_length = 0.1;
        params.maximum_window_length = 0.2;
        params.window_increment = 0.1;

        let err = pyrth_core::evaluate(input.clone(), &params).unwrap_err();

        assert!(matches!(
            err,
            PyrthError::UnsupportedStructureMethod(mode) if mode == structure_method.to_string()
        ));
    }
}

#[cfg(not(feature = "mpfr"))]
#[test]
fn mpfr_structure_methods_stay_unsupported_in_evaluate_without_mpfr_feature() {
    let input = TransientInput::from_pairs([(1e-6, 0.0), (1e-5, 0.1), (1e-4, 0.2)]).unwrap();

    let unsupported_methods = [
        StructureMethod::Sobhy,
        StructureMethod::Khatwani,
        StructureMethod::PolyLong,
    ];

    for structure_method in unsupported_methods {
        let mut params = EvaluationParams {
            deconv_mode: DeconvMode::Bayesian,
            structure_method,
            log_time_size: 8,
            bay_steps: 2,
            min_index: 1,
            minimum_window_size: 2,
            calc_struc: true,
            ..EvaluationParams::default()
        };
        params.minimum_window_length = 0.1;
        params.maximum_window_length = 0.2;
        params.window_increment = 0.1;

        let err = pyrth_core::evaluate(input.clone(), &params).unwrap_err();

        assert!(matches!(
            err,
            PyrthError::UnsupportedStructureMethod(mode) if mode == structure_method.to_string()
        ));
    }
}
