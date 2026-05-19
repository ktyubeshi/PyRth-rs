//! Python extension crate for the PyRth Rust port.

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
        let data = parameters
            .get_item("data")?
            .ok_or_else(|| PyValueError::new_err("data must be provided"))?
            .extract::<Vec<(f64, f64)>>()
            .map_err(|err| {
                PyValueError::new_err(format!("data must be a sequence of pairs: {err}"))
            })?;

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

        evaluate_impedance_with_params(
            py,
            data,
            EvalOverrides {
                input_mode,
                deconv_mode,
                filter_name,
                filter_range,
                filter_parameter,
                only_make_z,
                calc_struc,
                log_time_size,
                bay_steps,
                blockwise_sum_width,
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
            },
        )
    }
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
            filter_name: None,
            filter_range: None,
            filter_parameter: None,
            only_make_z,
            calc_struc,
            log_time_size: None,
            bay_steps: None,
            blockwise_sum_width: None,
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
    filter_name: Option<String>,
    filter_range: Option<f64>,
    filter_parameter: Option<f64>,
    only_make_z: bool,
    calc_struc: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    blockwise_sum_width: Option<usize>,
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

    let mut params = pyrth_core::EvaluationParams::default();
    if let Some(input_mode) = overrides.input_mode {
        params.input_mode = pyrth_core::InputMode::from_label(&input_mode)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(deconv_mode) = overrides.deconv_mode {
        params.deconv_mode = pyrth_core::DeconvMode::from_label(&deconv_mode)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
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
    if let Some(blockwise_sum_width) = overrides.blockwise_sum_width {
        params.blockwise_sum_width = blockwise_sum_width;
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
