use approx::assert_relative_eq;
use ndarray::array;
use pyrth_core::{
    compare_evaluations, CauerNetwork, EvaluationResult, FosterNetwork, ImpedanceData, PyrthError,
};

fn result(
    time_spectrum: Option<Vec<f64>>,
    foster_resistance: Option<Vec<f64>>,
    cauer_cumulative_resistance: Option<Vec<f64>>,
    differential_structure: Option<Vec<f64>>,
) -> EvaluationResult {
    EvaluationResult {
        impedance: ImpedanceData {
            time: array![1.0],
            impedance: array![0.0],
            log_time: array![0.0],
        },
        derivative: None,
        time_spectrum: time_spectrum.map(ndarray::Array1::from),
        foster: foster_resistance.map(|resistance| {
            let len = resistance.len();
            FosterNetwork {
                resistance: ndarray::Array1::from(resistance),
                capacitance: ndarray::Array1::ones(len),
                tau: ndarray::Array1::ones(len),
            }
        }),
        cauer: cauer_cumulative_resistance.map(|cumulative_resistance| {
            let len = cumulative_resistance.len();
            CauerNetwork {
                resistance: ndarray::Array1::ones(len),
                capacitance: ndarray::Array1::ones(len),
                cumulative_resistance: ndarray::Array1::from(cumulative_resistance),
                cumulative_capacitance: ndarray::Array1::ones(len),
                differential_structure: ndarray::Array1::from(
                    differential_structure.unwrap_or_default(),
                ),
            }
        }),
    }
}

#[test]
fn compares_time_spectrum_structure_and_cauer_resistance() {
    let reference = result(
        Some(vec![1.0, 2.0, 3.0]),
        Some(vec![99.0]),
        Some(vec![1.0, 3.0]),
        Some(vec![2.0, 4.0]),
    );
    let candidate = result(
        Some(vec![1.0, 1.0, 5.0, 99.0]),
        Some(vec![100.0]),
        Some(vec![1.0, 2.5]),
        Some(vec![1.0, 5.0, 99.0]),
    );

    let comparison = compare_evaluations(&reference, &candidate).unwrap();

    assert_relative_eq!(
        comparison.time_const_norm,
        (5.0_f64.sqrt()) / (14.0_f64.sqrt()),
        epsilon = 1e-14
    );
    assert_relative_eq!(
        comparison.structure_norm,
        (2.0_f64.sqrt()) / (20.0_f64.sqrt()),
        epsilon = 1e-14
    );
    assert_relative_eq!(comparison.total_resistance_diff, 0.5, epsilon = 1e-14);
}

#[test]
fn uses_foster_resistance_when_cauer_is_missing() {
    let reference = result(Some(vec![1.0]), Some(vec![1.0, 2.0, 3.0]), None, None);
    let candidate = result(Some(vec![1.0]), Some(vec![1.0, 2.5]), None, None);

    let comparison = compare_evaluations(&reference, &candidate).unwrap();

    assert_eq!(comparison.time_const_norm, 0.0);
    assert_eq!(comparison.structure_norm, 0.0);
    assert_relative_eq!(comparison.total_resistance_diff, 2.5, epsilon = 1e-14);
}

#[test]
fn missing_time_spectrum_is_rejected() {
    let reference = result(None, Some(vec![1.0]), None, None);
    let candidate = result(Some(vec![1.0]), Some(vec![1.0]), None, None);

    assert!(matches!(
        compare_evaluations(&reference, &candidate),
        Err(PyrthError::InvalidParameter {
            parameter: "reference.time_spectrum",
            ..
        })
    ));
}

#[test]
fn missing_resistance_source_is_rejected() {
    let reference = result(Some(vec![1.0]), None, None, None);
    let candidate = result(Some(vec![1.0]), Some(vec![1.0]), None, None);

    assert!(matches!(
        compare_evaluations(&reference, &candidate),
        Err(PyrthError::InvalidParameter {
            parameter: "total_resistance",
            ..
        })
    ));
}
