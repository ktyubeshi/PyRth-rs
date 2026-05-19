//! Python extension crate for the PyRth Rust port.

use std::fs;

use pyo3::{exceptions::PyValueError, prelude::*, types::PyDict};
pub use pyrth_core::*;

#[pyclass]
struct Evaluation;

#[pymethods]
impl Evaluation {
    #[new]
    fn new() -> Self {
        Self
    }

    fn standard_module(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let only_make_z = extract_bool(parameters, "only_make_z")?.unwrap_or(false);
        let calc_struc = extract_bool(parameters, "calc_struc")?.unwrap_or(true);
        let input_mode = extract_string(parameters, "input_mode")?;
        let deconv_mode =
            extract_string(parameters, "deconv_mode")?.or(extract_string(parameters, "deconv")?);
        let filter_name =
            extract_string(parameters, "filter_name")?.or(extract_string(parameters, "filter")?);
        let filter_range = extract_f64(parameters, "filter_range")?;
        let filter_parameter = extract_f64(parameters, "filter_parameter")?;
        let log_time_size = extract_usize(parameters, "log_time_size")?;
        let bay_steps = extract_usize(parameters, "bay_steps")?;
        let blockwise_sum_width = extract_usize(parameters, "blockwise_sum_width")?;
        let power_step = extract_f64(parameters, "power_step")?;
        let power_scale_factor = extract_f64(parameters, "power_scale_factor")?;
        let optical_power = extract_f64(parameters, "optical_power")?;
        let is_heating = extract_bool(parameters, "is_heating")?.unwrap_or(false);
        let kfac_fit_deg = extract_usize(parameters, "kfac_fit_deg")?;
        let calibration = extract_calibration(parameters)?;
        let data_cut_lower = extract_usize(parameters, "data_cut_lower")?;
        let data_cut_upper = extract_usize(parameters, "data_cut_upper")?;
        let temp_0_avg_range = extract_usize_pair(parameters, "temp_0_avg_range")?;
        let extrapolate = extract_bool(parameters, "extrapolate")?.unwrap_or(false);
        let lower_fit_limit = extract_f64(parameters, "lower_fit_limit")?;
        let upper_fit_limit = extract_f64(parameters, "upper_fit_limit")?;
        let structure_method =
            extract_first_string(parameters, &["struc_method", "structure_method"])?;
        let precision = extract_usize(parameters, "precision")?;
        let min_index = extract_usize(parameters, "min_index")?;
        let minimum_window_size = extract_usize(parameters, "minimum_window_size")?;
        let timespec_interpolate_factor = extract_f64(parameters, "timespec_interpolate_factor")?;
        let lasso_alpha = extract_f64(parameters, "lasso_alpha")?;
        let lasso_max_iter = extract_usize(parameters, "lasso_max_iter")?;
        let lasso_tol = extract_f64(parameters, "lasso_tol")?;
        let pad_factor_pre = extract_f64(parameters, "pad_factor_pre")?;
        let pad_factor_after = extract_f64(parameters, "pad_factor_after")?;
        let minimum_window_length = extract_f64(parameters, "minimum_window_length")?;
        let maximum_window_length = extract_f64(parameters, "maximum_window_length")?;
        let window_increment = extract_f64(parameters, "window_increment")?;
        let expected_var = extract_f64(parameters, "expected_var")?;

        let mut overrides = EvalOverrides {
            input_mode,
            deconv_mode,
            structure_method,
            precision,
            filter_name,
            filter_range,
            filter_parameter,
            only_make_z,
            calc_struc,
            log_time_size,
            bay_steps,
            min_index,
            minimum_window_size,
            timespec_interpolate_factor,
            blockwise_sum_width,
            lasso_alpha,
            lasso_max_iter,
            lasso_tol,
            pad_factor_pre,
            pad_factor_after,
            minimum_window_length,
            maximum_window_length,
            window_increment,
            expected_var,
            power_step,
            power_scale_factor,
            optical_power,
            is_heating,
            kfac_fit_deg,
            calibration,
            data_cut_lower,
            data_cut_upper,
            temp_0_avg_range,
            extrapolate,
            lower_fit_limit,
            upper_fit_limit,
        };
        let input = extract_transient_input(parameters, &mut overrides)?;

        evaluate_transient_input_with_params(py, input, overrides)
    }
}

fn extract_transient_input(
    parameters: &Bound<'_, PyDict>,
    overrides: &mut EvalOverrides,
) -> PyResult<pyrth_core::TransientInput> {
    if let Some(data) = parameters.get_item("data")? {
        let data = data.extract::<Vec<(f64, f64)>>().map_err(|err| {
            PyValueError::new_err(format!("data must be a sequence of pairs: {err}"))
        })?;
        return pyrth_core::TransientInput::from_pairs(data)
            .map_err(|err| PyValueError::new_err(err.to_string()));
    }

    if !matches!(
        overrides.input_mode.as_deref(),
        Some(mode) if mode.eq_ignore_ascii_case("t3ster")
    ) {
        return Err(PyValueError::new_err(
            "data must be provided unless input_mode is 't3ster' with file paths",
        ));
    }

    let raw_path = extract_first_string(parameters, &["infile", "input"])?
        .ok_or_else(|| PyValueError::new_err("input_mode='t3ster' requires infile or input"))?;
    let calibration_path = extract_first_string(parameters, &["infile_tco", "t3ster_calibration"])?
        .ok_or_else(|| {
            PyValueError::new_err("input_mode='t3ster' requires infile_tco or t3ster_calibration")
        })?;
    let power_path = extract_first_string(parameters, &["infile_pwr", "t3ster_power"])?;

    let raw_text = fs::read_to_string(&raw_path).map_err(|err| {
        PyValueError::new_err(format!("failed to read T3Ster raw file {raw_path}: {err}"))
    })?;
    let calibration_text = fs::read_to_string(&calibration_path).map_err(|err| {
        PyValueError::new_err(format!(
            "failed to read T3Ster calibration file {calibration_path}: {err}"
        ))
    })?;
    let raw = pyrth_core::parse_t3ster_raw_text(&raw_text)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let calibration = pyrth_core::parse_t3ster_calibration_text(&calibration_text)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    if let Some(power_path) = power_path {
        let power_text = fs::read_to_string(&power_path).map_err(|err| {
            PyValueError::new_err(format!(
                "failed to read T3Ster power file {power_path}: {err}"
            ))
        })?;
        overrides.power_step = Some(
            pyrth_core::parse_t3ster_power_step(&power_text)
                .map_err(|err| PyValueError::new_err(err.to_string()))?,
        );
    }

    overrides.input_mode = Some("temp".to_string());
    pyrth_core::t3ster_raw_to_temperature_input(
        &raw,
        &calibration,
        overrides
            .kfac_fit_deg
            .unwrap_or_else(|| pyrth_core::EvaluationParams::default().kfac_fit_deg),
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn extract_first_string(parameters: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<Option<String>> {
    for key in keys {
        if let Some(value) = extract_string(parameters, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

#[pyfunction]
#[pyo3(signature = (data, only_make_z=false, calc_struc=true))]
fn evaluate_impedance(
    py: Python<'_>,
    data: Vec<(f64, f64)>,
    only_make_z: bool,
    calc_struc: bool,
) -> PyResult<PyObject> {
    evaluate_impedance_with_params(
        py,
        data,
        EvalOverrides {
            input_mode: None,
            deconv_mode: None,
            structure_method: None,
            precision: None,
            filter_name: None,
            filter_range: None,
            filter_parameter: None,
            only_make_z,
            calc_struc,
            log_time_size: None,
            bay_steps: None,
            min_index: None,
            minimum_window_size: None,
            timespec_interpolate_factor: None,
            blockwise_sum_width: None,
            lasso_alpha: None,
            lasso_max_iter: None,
            lasso_tol: None,
            pad_factor_pre: None,
            pad_factor_after: None,
            minimum_window_length: None,
            maximum_window_length: None,
            window_increment: None,
            expected_var: None,
            power_step: None,
            power_scale_factor: None,
            optical_power: None,
            is_heating: false,
            kfac_fit_deg: None,
            calibration: None,
            data_cut_lower: None,
            data_cut_upper: None,
            temp_0_avg_range: None,
            extrapolate: false,
            lower_fit_limit: None,
            upper_fit_limit: None,
        },
    )
}

struct EvalOverrides {
    input_mode: Option<String>,
    deconv_mode: Option<String>,
    structure_method: Option<String>,
    precision: Option<usize>,
    filter_name: Option<String>,
    filter_range: Option<f64>,
    filter_parameter: Option<f64>,
    only_make_z: bool,
    calc_struc: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    min_index: Option<usize>,
    minimum_window_size: Option<usize>,
    timespec_interpolate_factor: Option<f64>,
    blockwise_sum_width: Option<usize>,
    lasso_alpha: Option<f64>,
    lasso_max_iter: Option<usize>,
    lasso_tol: Option<f64>,
    pad_factor_pre: Option<f64>,
    pad_factor_after: Option<f64>,
    minimum_window_length: Option<f64>,
    maximum_window_length: Option<f64>,
    window_increment: Option<f64>,
    expected_var: Option<f64>,
    power_step: Option<f64>,
    power_scale_factor: Option<f64>,
    optical_power: Option<f64>,
    is_heating: bool,
    kfac_fit_deg: Option<usize>,
    calibration: Option<Vec<[f64; 2]>>,
    data_cut_lower: Option<usize>,
    data_cut_upper: Option<usize>,
    temp_0_avg_range: Option<(usize, usize)>,
    extrapolate: bool,
    lower_fit_limit: Option<f64>,
    upper_fit_limit: Option<f64>,
}

fn evaluate_impedance_with_params(
    py: Python<'_>,
    data: Vec<(f64, f64)>,
    overrides: EvalOverrides,
) -> PyResult<PyObject> {
    let input = pyrth_core::TransientInput::from_pairs(data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    evaluate_impedance_with_input(py, input, overrides)
}

fn evaluate_transient_input_with_params(
    py: Python<'_>,
    input: pyrth_core::TransientInput,
    overrides: EvalOverrides,
) -> PyResult<PyObject> {
    evaluate_impedance_with_input(py, input, overrides)
}

fn evaluate_impedance_with_input(
    py: Python<'_>,
    input: pyrth_core::TransientInput,
    overrides: EvalOverrides,
) -> PyResult<PyObject> {
    let mut params = pyrth_core::EvaluationParams::default();
    if let Some(input_mode) = overrides.input_mode {
        params.input_mode = pyrth_core::InputMode::from_label(&input_mode)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(deconv_mode) = overrides.deconv_mode {
        params.deconv_mode = pyrth_core::DeconvMode::from_label(&deconv_mode)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(structure_method) = overrides.structure_method {
        params.structure_method = pyrth_core::StructureMethod::from_label(&structure_method)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(precision) = overrides.precision {
        params.precision = precision;
    }
    if let Some(filter_name) = overrides.filter_name {
        params.filter_name = pyrth_core::FourierFilter::from_label(&filter_name)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(filter_range) = overrides.filter_range {
        params.filter_range = filter_range;
    }
    if let Some(filter_parameter) = overrides.filter_parameter {
        params.filter_parameter = filter_parameter;
    }
    params.only_make_z = overrides.only_make_z;
    params.calc_struc = overrides.calc_struc;
    if let Some(log_time_size) = overrides.log_time_size {
        params.log_time_size = log_time_size;
    }
    if let Some(bay_steps) = overrides.bay_steps {
        params.bay_steps = bay_steps;
    }
    if let Some(min_index) = overrides.min_index {
        params.min_index = min_index;
    }
    if let Some(minimum_window_size) = overrides.minimum_window_size {
        params.minimum_window_size = minimum_window_size;
    }
    if let Some(timespec_interpolate_factor) = overrides.timespec_interpolate_factor {
        params.timespec_interpolate_factor = timespec_interpolate_factor;
    }
    if let Some(blockwise_sum_width) = overrides.blockwise_sum_width {
        params.blockwise_sum_width = blockwise_sum_width;
    }
    if let Some(lasso_alpha) = overrides.lasso_alpha {
        params.lasso_alpha = lasso_alpha;
    }
    if let Some(lasso_max_iter) = overrides.lasso_max_iter {
        params.lasso_max_iter = lasso_max_iter;
    }
    if let Some(lasso_tol) = overrides.lasso_tol {
        params.lasso_tol = lasso_tol;
    }
    if let Some(pad_factor_pre) = overrides.pad_factor_pre {
        params.pad_factor_pre = pad_factor_pre;
    }
    if let Some(pad_factor_after) = overrides.pad_factor_after {
        params.pad_factor_after = pad_factor_after;
    }
    if let Some(minimum_window_length) = overrides.minimum_window_length {
        params.minimum_window_length = minimum_window_length;
    }
    if let Some(maximum_window_length) = overrides.maximum_window_length {
        params.maximum_window_length = maximum_window_length;
    }
    if let Some(window_increment) = overrides.window_increment {
        params.window_increment = window_increment;
    }
    if let Some(expected_var) = overrides.expected_var {
        params.expected_var = expected_var;
    }
    if let Some(power_step) = overrides.power_step {
        params.power_step = power_step;
    }
    if let Some(power_scale_factor) = overrides.power_scale_factor {
        params.power_scale_factor = power_scale_factor;
    }
    if let Some(optical_power) = overrides.optical_power {
        params.optical_power = optical_power;
    }
    params.is_heating = overrides.is_heating;
    if let Some(kfac_fit_deg) = overrides.kfac_fit_deg {
        params.kfac_fit_deg = kfac_fit_deg;
    }
    if let Some(calibration) = overrides.calibration {
        params.calibration = Some(calibration);
    }
    if let Some(data_cut_lower) = overrides.data_cut_lower {
        params.data_cut_lower = data_cut_lower;
    }
    if let Some(data_cut_upper) = overrides.data_cut_upper {
        params.data_cut_upper = Some(data_cut_upper);
    }
    if let Some(temp_0_avg_range) = overrides.temp_0_avg_range {
        params.temp_0_avg_range = temp_0_avg_range;
    }
    params.extrapolate = overrides.extrapolate;
    params.lower_fit_limit = overrides.lower_fit_limit;
    params.upper_fit_limit = overrides.upper_fit_limit;

    let result = pyrth_core::evaluate(input, &params)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let output = PyDict::new(py);
    output.set_item("time", result.impedance.time.to_vec())?;
    output.set_item("impedance", result.impedance.impedance.to_vec())?;
    output.set_item("log_time", result.impedance.log_time.to_vec())?;

    if let Some(derivative) = result.derivative {
        let derivative_dict = PyDict::new(py);
        derivative_dict.set_item("imp_smooth", derivative.imp_smooth.to_vec())?;
        derivative_dict.set_item("imp_deriv_interp", derivative.imp_deriv_interp.to_vec())?;
        derivative_dict.set_item("log_time_interp", derivative.log_time_interp.to_vec())?;
        derivative_dict.set_item("imp_smooth_full", derivative.imp_smooth_full.to_vec())?;
        derivative_dict.set_item("log_time_pad", derivative.log_time_pad.to_vec())?;
        derivative_dict.set_item("log_time_delta", derivative.log_time_delta)?;
        output.set_item("derivative", derivative_dict)?;
    }

    if let Some(time_spectrum) = result.time_spectrum {
        output.set_item("time_spec", time_spectrum.to_vec())?;
    }

    if let Some(foster) = result.foster {
        let foster_dict = PyDict::new(py);
        foster_dict.set_item("resistance", foster.resistance.to_vec())?;
        foster_dict.set_item("capacitance", foster.capacitance.to_vec())?;
        foster_dict.set_item("tau", foster.tau.to_vec())?;
        output.set_item("foster", foster_dict)?;
    }

    if let Some(cauer) = result.cauer {
        let cauer_dict = PyDict::new(py);
        cauer_dict.set_item("resistance", cauer.resistance.to_vec())?;
        cauer_dict.set_item("capacitance", cauer.capacitance.to_vec())?;
        cauer_dict.set_item(
            "cumulative_resistance",
            cauer.cumulative_resistance.to_vec(),
        )?;
        cauer_dict.set_item(
            "cumulative_capacitance",
            cauer.cumulative_capacitance.to_vec(),
        )?;
        cauer_dict.set_item(
            "differential_structure",
            cauer.differential_structure.to_vec(),
        )?;
        output.set_item("cauer", cauer_dict)?;
    }

    Ok(output.into())
}

#[pymodule]
fn pyrth_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Evaluation>()?;
    m.add_function(wrap_pyfunction!(evaluate_impedance, m)?)?;
    Ok(())
}

fn extract_bool(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<bool>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<bool>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be bool: {err}")))
}

fn extract_usize(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<usize>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<usize>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a positive integer: {err}")))
}

fn extract_f64(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<f64>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<f64>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a number: {err}")))
}

fn extract_string(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<String>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a string: {err}")))
}

fn extract_calibration(parameters: &Bound<'_, PyDict>) -> PyResult<Option<Vec<[f64; 2]>>> {
    parameters
        .get_item("calibration")?
        .map(|value| {
            let pairs = value.extract::<Vec<(f64, f64)>>()?;
            Ok(pairs
                .into_iter()
                .map(|(temperature, voltage)| [temperature, voltage])
                .collect())
        })
        .transpose()
        .map_err(|err: PyErr| {
            PyValueError::new_err(format!(
                "calibration must be a sequence of (temperature, voltage) pairs: {err}"
            ))
        })
}

fn extract_usize_pair(
    parameters: &Bound<'_, PyDict>,
    key: &str,
) -> PyResult<Option<(usize, usize)>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<(usize, usize)>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a pair of integers: {err}")))
}
