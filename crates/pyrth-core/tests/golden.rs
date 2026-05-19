use std::{fs, path::PathBuf};

use approx::{assert_relative_eq, relative_eq};
use ndarray::Array1;
use pyrth_core::{
    evaluate, DeconvMode, EvaluationParams, InputMode, StructureMethod, TransientInput,
};
use serde::Deserialize;

#[derive(Debug, Deserialize)]
struct GoldenFixture {
    input: GoldenInput,
    parameters: GoldenParams,
    reference: GoldenReference,
}

#[derive(Debug, Deserialize)]
struct GoldenInput {
    mode: String,
    data: Vec<[f64; 2]>,
}

#[derive(Debug, Deserialize)]
struct GoldenParams {
    deconv_mode: String,
    struc_method: String,
    precision: usize,
    log_time_size: usize,
    bay_steps: usize,
    pad_factor_pre: f64,
    pad_factor_after: f64,
    minimum_window_length: f64,
    maximum_window_length: f64,
    minimum_window_size: usize,
    window_increment: f64,
    expected_var: f64,
    min_index: usize,
    timespec_interpolate_factor: f64,
    blockwise_sum_width: usize,
}

#[derive(Debug, Deserialize)]
struct GoldenReference {
    time: Vec<f64>,
    impedance: Vec<f64>,
    log_time: Vec<f64>,
    imp_deriv_interp: Vec<f64>,
    imp_smooth: Vec<f64>,
    imp_smooth_full: Vec<f64>,
    log_time_delta: f64,
    log_time_interp: Vec<f64>,
    log_time_pad: Vec<f64>,
    time_spec: Vec<f64>,
}

fn fixture_path(name: &str) -> PathBuf {
    PathBuf::from(env!("CARGO_MANIFEST_DIR"))
        .join("..")
        .join("..")
        .join("tests")
        .join("golden")
        .join("fixtures")
        .join(name)
}

fn read_fixture(name: &str) -> GoldenFixture {
    let path = fixture_path(name);
    let raw = fs::read_to_string(&path).unwrap_or_else(|err| {
        panic!("failed to read fixture {}: {err}", path.display());
    });
    serde_json::from_str(&raw).unwrap_or_else(|err| {
        panic!("failed to parse fixture {}: {err}", path.display());
    })
}

fn params_from_fixture(fixture: &GoldenFixture) -> EvaluationParams {
    let params = &fixture.parameters;
    EvaluationParams {
        input_mode: InputMode::from_label(&fixture.input.mode).unwrap(),
        deconv_mode: DeconvMode::from_label(&params.deconv_mode).unwrap(),
        structure_method: StructureMethod::from_label(&params.struc_method).unwrap(),
        precision: params.precision,
        log_time_size: params.log_time_size,
        bay_steps: params.bay_steps,
        pad_factor_pre: params.pad_factor_pre,
        pad_factor_after: params.pad_factor_after,
        minimum_window_length: params.minimum_window_length,
        maximum_window_length: params.maximum_window_length,
        minimum_window_size: params.minimum_window_size,
        window_increment: params.window_increment,
        expected_var: params.expected_var,
        min_index: params.min_index,
        timespec_interpolate_factor: params.timespec_interpolate_factor,
        blockwise_sum_width: params.blockwise_sum_width,
    }
}

fn assert_array_close(name: &str, actual: &Array1<f64>, expected: &[f64]) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            relative_eq!(actual, expected, epsilon = 1e-12, max_relative = 1e-10,),
            "mismatch in {name}[{index}]: actual={actual}, expected={expected}"
        );
    }
}

#[test]
fn impedance_and_derivative_match_python_golden() {
    for fixture_name in [
        "mosfet_tim_bayesian_lanczos.json",
        "mosfet_dry_bayesian_lanczos.json",
        "led_bayesian_lanczos.json",
    ] {
        assert_impedance_matches(fixture_name);
    }

    for fixture_name in [
        "mosfet_tim_bayesian_lanczos.json",
        "mosfet_dry_bayesian_lanczos.json",
    ] {
        assert_derivative_matches(fixture_name);
        assert_time_spectrum_matches(fixture_name);
    }
}

fn evaluate_fixture(fixture_name: &str) -> (GoldenFixture, pyrth_core::EvaluationResult) {
    let fixture = read_fixture(fixture_name);
    let params = params_from_fixture(&fixture);
    let input =
        TransientInput::from_pairs(fixture.input.data.iter().map(|pair| (pair[0], pair[1])))
            .unwrap();

    let result = evaluate(input, &params).unwrap();
    (fixture, result)
}

fn assert_impedance_matches(fixture_name: &str) {
    let (fixture, result) = evaluate_fixture(fixture_name);
    assert_array_close(
        &format!("{fixture_name}:time"),
        &result.impedance.time,
        &fixture.reference.time,
    );
    assert_array_close(
        &format!("{fixture_name}:impedance"),
        &result.impedance.impedance,
        &fixture.reference.impedance,
    );
    assert_array_close(
        &format!("{fixture_name}:log_time"),
        &result.impedance.log_time,
        &fixture.reference.log_time,
    );
}

fn assert_derivative_matches(fixture_name: &str) {
    let (fixture, result) = evaluate_fixture(fixture_name);
    let derivative = result.derivative.as_ref().unwrap();
    assert_relative_eq!(
        derivative.log_time_delta,
        fixture.reference.log_time_delta,
        epsilon = 1e-12,
        max_relative = 1e-10
    );
    assert_array_close(
        &format!("{fixture_name}:log_time_interp"),
        &derivative.log_time_interp,
        &fixture.reference.log_time_interp,
    );
    assert_array_close(
        &format!("{fixture_name}:log_time_pad"),
        &derivative.log_time_pad,
        &fixture.reference.log_time_pad,
    );
    assert_array_close(
        &format!("{fixture_name}:imp_smooth"),
        &derivative.imp_smooth,
        &fixture.reference.imp_smooth,
    );
    assert_array_close(
        &format!("{fixture_name}:imp_smooth_full"),
        &derivative.imp_smooth_full,
        &fixture.reference.imp_smooth_full,
    );
    assert_array_close(
        &format!("{fixture_name}:imp_deriv_interp"),
        &derivative.imp_deriv_interp,
        &fixture.reference.imp_deriv_interp,
    );
}

fn assert_time_spectrum_matches(fixture_name: &str) {
    let (fixture, result) = evaluate_fixture(fixture_name);
    let time_spectrum = result.time_spectrum.as_ref().unwrap();
    assert_array_close_with_tolerance(
        &format!("{fixture_name}:time_spec"),
        time_spectrum,
        &fixture.reference.time_spec,
        1e-8,
        1e-6,
    );
}

fn assert_array_close_with_tolerance(
    name: &str,
    actual: &Array1<f64>,
    expected: &[f64],
    epsilon: f64,
    max_relative: f64,
) {
    assert_eq!(actual.len(), expected.len(), "{name} length mismatch");
    for (index, (actual, expected)) in actual.iter().zip(expected).enumerate() {
        assert!(
            relative_eq!(
                actual,
                expected,
                epsilon = epsilon,
                max_relative = max_relative,
            ),
            "mismatch in {name}[{index}]: actual={actual}, expected={expected}"
        );
    }
}
