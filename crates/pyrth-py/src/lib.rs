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
        let log_time_size = extract_usize(parameters, "log_time_size")?;
        let bay_steps = extract_usize(parameters, "bay_steps")?;
        let blockwise_sum_width = extract_usize(parameters, "blockwise_sum_width")?;

        evaluate_impedance_with_params(
            py,
            data,
            EvalOverrides {
                only_make_z,
                calc_struc,
                log_time_size,
                bay_steps,
                blockwise_sum_width,
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
            only_make_z,
            calc_struc,
            log_time_size: None,
            bay_steps: None,
            blockwise_sum_width: None,
        },
    )
}

struct EvalOverrides {
    only_make_z: bool,
    calc_struc: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    blockwise_sum_width: Option<usize>,
}

fn evaluate_impedance_with_params(
    py: Python<'_>,
    data: Vec<(f64, f64)>,
    overrides: EvalOverrides,
) -> PyResult<PyObject> {
    let input = pyrth_core::TransientInput::from_pairs(data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let mut params = pyrth_core::EvaluationParams::default();
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
