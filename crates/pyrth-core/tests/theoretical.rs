use approx::{assert_abs_diff_eq, assert_relative_eq};
use ndarray::Array1;
use pyrth_core::{structure_params_to_func, theoretical_module, time_const_to_impedance};
use serde::Deserialize;

#[derive(Deserialize)]
struct TheoreticalFixture {
    params: TheoreticalFixtureParams,
    theo_log_time: Vec<f64>,
    theo_int_cau_res: Vec<f64>,
    theo_int_cau_cap: Vec<f64>,
    theo_diff_struc: Vec<f64>,
    theo_time_const: Vec<f64>,
    theo_imp_deriv: Vec<f64>,
    theo_impedance: Vec<f64>,
}

#[derive(Deserialize)]
struct TheoreticalFixtureParams {
    theo_time: [f64; 2],
    theo_time_size: usize,
    theo_delta: f64,
    theo_resistances: Vec<f64>,
    theo_capacitances: Vec<f64>,
}

#[test]
fn structure_params_to_func_preserves_section_boundaries() {
    let resistances = Array1::from(vec![0.1, 0.2, 0.3]);
    let capacitances = Array1::from(vec![1e-9, 1e-8, 1e-7]);

    let (res, cap) = structure_params_to_func(6, &resistances, &capacitances).unwrap();

    assert_eq!(res[0], 0.0);
    assert_relative_eq!(res[res.len() - 1], 0.6, epsilon = 1e-14);
    assert!(res.iter().any(|value| (*value - 0.1).abs() < 1e-14));
    assert!(res.iter().any(|value| (*value - 0.3).abs() < 1e-14));
    assert_eq!(res.len(), 8);
    assert_relative_eq!(cap[0], 0.0, epsilon = 1e-20);
    assert_relative_eq!(cap[cap.len() - 1], 1.11e-7, epsilon = 1e-20);
}

#[test]
fn theoretical_module_returns_python_compatible_keys_lengths() {
    let result = theoretical_module(
        &[0.1, 0.2, 0.3],
        &[1e-9, 1e-8, 1e-7],
        5e-16,
        0.02,
        1000,
        std::f64::consts::PI / 90.0,
    )
    .unwrap();

    assert_eq!(result.log_time.len(), 1000);
    assert_eq!(result.time_const_spectrum.len(), 1000);
    assert_eq!(result.impedance_derivative.len(), 1000);
    assert_eq!(result.impedance.len(), 1000);
    assert_eq!(result.cumulative_resistance.len(), 1001);
    assert_eq!(result.cumulative_capacitance.len(), 1001);
    assert_eq!(result.differential_structure.len(), 1000);
    assert!(result
        .time_const_spectrum
        .iter()
        .all(|value| value.is_finite()));
    assert!(result
        .impedance_derivative
        .iter()
        .all(|value| value.is_finite()));
    assert!(result.impedance.iter().all(|value| value.is_finite()));
    assert_abs_diff_eq!(result.impedance[0], 0.0, epsilon = 1e-14);
}

#[test]
fn theoretical_module_matches_python_generated_basic_fixture() {
    let fixture: TheoreticalFixture =
        serde_json::from_str(include_str!("fixtures/theoretical_case_basic.json")).unwrap();
    let result = theoretical_module(
        &fixture.params.theo_resistances,
        &fixture.params.theo_capacitances,
        fixture.params.theo_time[0],
        fixture.params.theo_time[1],
        fixture.params.theo_time_size,
        fixture.params.theo_delta,
    )
    .unwrap();

    assert_vec_close(
        &result.log_time.to_vec(),
        &fixture.theo_log_time,
        1e-14,
        1e-14,
    );
    assert_vec_close(
        &result.cumulative_resistance.to_vec(),
        &fixture.theo_int_cau_res,
        1e-12,
        1e-14,
    );
    assert_vec_close(
        &result.cumulative_capacitance.to_vec(),
        &fixture.theo_int_cau_cap,
        1e-12,
        1e-14,
    );
    assert_vec_close(
        &result.differential_structure.to_vec(),
        &fixture.theo_diff_struc,
        1e-12,
        1e-14,
    );
    assert_vec_close(
        &result.time_const_spectrum.to_vec(),
        &fixture.theo_time_const,
        1e-8,
        1e-12,
    );
    assert_vec_close(
        &result.impedance_derivative.to_vec(),
        &fixture.theo_imp_deriv,
        1e-8,
        1e-12,
    );
    assert_vec_close(
        &result.impedance.to_vec(),
        &fixture.theo_impedance,
        1e-8,
        1e-12,
    );
}

#[test]
fn time_const_to_impedance_uses_python_weight_slice() {
    let log_time = Array1::linspace(-2.0, 2.0, 5);
    let time_const = Array1::from(vec![0.0, 1.0, 2.0, 1.0, 0.0]);

    let (deriv, impedance) = time_const_to_impedance(&log_time, &time_const).unwrap();

    assert_eq!(deriv.len(), 5);
    assert_eq!(impedance.len(), 5);
    assert_abs_diff_eq!(impedance[0], 0.0);
    assert!(deriv.iter().all(|value| value.is_finite()));
    assert!(impedance.iter().all(|value| value.is_finite()));
}

fn assert_vec_close(actual: &[f64], expected: &[f64], rtol: f64, atol: f64) {
    assert_eq!(actual.len(), expected.len());
    for (index, (actual, expected)) in actual.iter().zip(expected.iter()).enumerate() {
        let tolerance = atol + rtol * expected.abs();
        assert!(
            (actual - expected).abs() <= tolerance,
            "index {index}: actual={actual}, expected={expected}, tolerance={tolerance}"
        );
    }
}
