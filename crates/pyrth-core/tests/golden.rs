use std::{fs, path::PathBuf};

use approx::{assert_relative_eq, relative_eq};
use ndarray::Array1;
use pyrth_core::{
    cauer_from_foster_lanczos, evaluate, export_csv, DeconvMode, EvaluationParams, InputMode,
    PyrthError, StructureMethod, TransientInput,
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
    int_cau_cap: Vec<f64>,
    int_cau_res: Vec<f64>,
    log_time_delta: f64,
    log_time_interp: Vec<f64>,
    log_time_pad: Vec<f64>,
    therm_capa_fost: Vec<f64>,
    therm_resist_fost: Vec<f64>,
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
        calc_struc: true,
        only_make_z: false,
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
        let (fixture, result) = evaluate_fixture(fixture_name);
        assert_impedance_matches(fixture_name, &fixture, &result);
    }

    for fixture_name in [
        "mosfet_tim_bayesian_lanczos.json",
        "mosfet_dry_bayesian_lanczos.json",
    ] {
        assert_golden_foster_to_cauer_matches(fixture_name);
    }

    for fixture_name in [
        "mosfet_tim_bayesian_lanczos.json",
        "mosfet_dry_bayesian_lanczos.json",
    ] {
        let (fixture, result) = evaluate_fixture(fixture_name);
        assert_derivative_matches(fixture_name, &fixture, &result);
        assert_time_spectrum_matches(fixture_name, &fixture, &result);
        assert_foster_matches(fixture_name, &fixture, &result);
    }
}

fn evaluate_fixture(fixture_name: &str) -> (GoldenFixture, pyrth_core::EvaluationResult) {
    let fixture = read_fixture(fixture_name);
    let params = params_from_fixture(&fixture);
    evaluate_fixture_with_params(fixture, params)
}

fn evaluate_fixture_with_params(
    fixture: GoldenFixture,
    params: EvaluationParams,
) -> (GoldenFixture, pyrth_core::EvaluationResult) {
    let input =
        TransientInput::from_pairs(fixture.input.data.iter().map(|pair| (pair[0], pair[1])))
            .unwrap();

    let result = evaluate(input, &params).unwrap();
    (fixture, result)
}

#[test]
fn evaluation_flags_gate_pipeline_stages() {
    let fixture = read_fixture("mosfet_tim_bayesian_lanczos.json");
    let mut params = params_from_fixture(&fixture);
    params.only_make_z = true;
    let (_, result) = evaluate_fixture_with_params(fixture, params);

    assert!(result.derivative.is_none());
    assert!(result.time_spectrum.is_none());
    assert!(result.foster.is_none());
    assert!(result.cauer.is_none());

    let fixture = read_fixture("mosfet_tim_bayesian_lanczos.json");
    let mut params = params_from_fixture(&fixture);
    params.calc_struc = false;
    let (_, result) = evaluate_fixture_with_params(fixture, params);

    assert!(result.derivative.is_some());
    assert!(result.time_spectrum.is_some());
    assert!(result.foster.is_some());
    assert!(result.cauer.is_none());
}

#[test]
fn invalid_evaluation_params_return_errors() {
    let input = TransientInput::from_pairs([(1e-6, 0.1), (1e-5, 0.2)]).unwrap();
    let mut params = EvaluationParams::default();
    params.log_time_size = 0;

    let err = evaluate(input, &params).unwrap_err();
    assert!(matches!(
        err,
        PyrthError::InvalidParameter {
            parameter: "log_time_size",
            ..
        }
    ));

    let input = TransientInput::from_pairs([(1e-6, 0.1)]).unwrap();
    let params = EvaluationParams::default();
    let err = evaluate(input, &params).unwrap_err();
    assert!(matches!(
        err,
        PyrthError::InvalidParameter {
            parameter: "data",
            ..
        }
    ));

    let input = TransientInput::from_pairs([(1e-6, 0.1), (1e-5, 0.2), (1e-4, 0.3)]).unwrap();
    let params = EvaluationParams::default();
    let err = evaluate(input, &params).unwrap_err();
    assert!(matches!(
        err,
        PyrthError::InvalidParameter {
            parameter: "min_index",
            ..
        }
    ));
}

#[test]
fn csv_export_writes_available_pipeline_outputs() {
    let fixture = read_fixture("mosfet_tim_bayesian_lanczos.json");
    let mut params = params_from_fixture(&fixture);
    params.only_make_z = true;
    let (_, result) = evaluate_fixture_with_params(fixture, params);

    let output_dir = std::env::temp_dir().join(format!(
        "pyrth_core_only_make_z_export_{}",
        std::process::id()
    ));
    if output_dir.exists() {
        fs::remove_dir_all(&output_dir).unwrap();
    }

    let files = export_csv(&result, &output_dir).unwrap();

    assert!(files.impedance.exists());
    assert!(files.imp_deriv.is_none());
    assert!(files.time_spec.is_none());
    assert!(files.foster.is_none());
    assert!(files.cauer.is_none());
    assert!(files.diff_struc.is_none());

    let header = fs::read_to_string(files.impedance)
        .unwrap()
        .lines()
        .next()
        .unwrap()
        .to_string();
    assert_eq!(header, "time,impedance");
}

fn assert_impedance_matches(
    fixture_name: &str,
    fixture: &GoldenFixture,
    result: &pyrth_core::EvaluationResult,
) {
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

fn assert_derivative_matches(
    fixture_name: &str,
    fixture: &GoldenFixture,
    result: &pyrth_core::EvaluationResult,
) {
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

fn assert_time_spectrum_matches(
    fixture_name: &str,
    fixture: &GoldenFixture,
    result: &pyrth_core::EvaluationResult,
) {
    let time_spectrum = result.time_spectrum.as_ref().unwrap();
    assert_array_close_with_tolerance(
        &format!("{fixture_name}:time_spec"),
        time_spectrum,
        &fixture.reference.time_spec,
        1e-8,
        1e-6,
    );
}

fn assert_foster_matches(
    fixture_name: &str,
    fixture: &GoldenFixture,
    result: &pyrth_core::EvaluationResult,
) {
    let foster = result.foster.as_ref().unwrap();
    assert_array_close_with_tolerance(
        &format!("{fixture_name}:therm_resist_fost"),
        &foster.resistance,
        &fixture.reference.therm_resist_fost,
        1e-8,
        1e-6,
    );
    assert_array_close_with_tolerance(
        &format!("{fixture_name}:therm_capa_fost"),
        &foster.capacitance,
        &fixture.reference.therm_capa_fost,
        1e-8,
        1e-6,
    );
}

fn assert_golden_foster_to_cauer_matches(fixture_name: &str) {
    let fixture = read_fixture(fixture_name);
    let params = params_from_fixture(&fixture);
    let cauer = cauer_from_foster_lanczos(
        &Array1::from(fixture.reference.therm_capa_fost.clone()),
        &Array1::from(fixture.reference.therm_resist_fost.clone()),
        &params,
    );

    assert_prefix_close_with_tolerance(
        &format!("{fixture_name}:int_cau_res"),
        &cauer.cumulative_resistance,
        &fixture.reference.int_cau_res,
        1,
        1e-7,
        1e-4,
    );
    assert_prefix_close_with_tolerance(
        &format!("{fixture_name}:int_cau_cap"),
        &cauer.cumulative_capacitance,
        &fixture.reference.int_cau_cap,
        1,
        1e-7,
        1e-4,
    );
    assert_cauer_is_physical(fixture_name, &cauer);
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

fn assert_prefix_close_with_tolerance(
    name: &str,
    actual: &Array1<f64>,
    expected: &[f64],
    prefix_len: usize,
    epsilon: f64,
    max_relative: f64,
) {
    assert!(
        actual.len() >= prefix_len,
        "{name} shorter than prefix length: {} < {prefix_len}",
        actual.len()
    );
    assert!(
        expected.len() >= prefix_len,
        "{name} expected shorter than prefix length: {} < {prefix_len}",
        expected.len()
    );
    for index in 0..prefix_len {
        let actual = actual[index];
        let expected = expected[index];
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

fn assert_cauer_is_physical(fixture_name: &str, cauer: &pyrth_core::CauerNetwork) {
    assert!(
        !cauer.resistance.is_empty(),
        "{fixture_name}: Cauer resistance is empty"
    );
    assert!(
        !cauer.capacitance.is_empty(),
        "{fixture_name}: Cauer capacitance is empty"
    );
    assert!(
        cauer
            .resistance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "{fixture_name}: Cauer resistance contains invalid values"
    );
    assert!(
        cauer
            .capacitance
            .iter()
            .all(|value| value.is_finite() && *value >= 0.0),
        "{fixture_name}: Cauer capacitance contains invalid values"
    );
    assert!(
        cauer
            .cumulative_resistance
            .windows(2)
            .into_iter()
            .all(|pair| pair[0] <= pair[1]),
        "{fixture_name}: cumulative resistance is not monotonic"
    );
    assert!(
        cauer
            .cumulative_capacitance
            .iter()
            .all(|value| value.is_finite()),
        "{fixture_name}: cumulative capacitance contains invalid values"
    );
}
