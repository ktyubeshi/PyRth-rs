//! Python extension crate for the PyRth Rust port.

use std::{collections::HashMap, fs, path::Path, sync::RwLock};

use pyo3::{
    exceptions::{PyKeyError, PyValueError},
    prelude::*,
    types::{PyAny, PyDict, PyList},
};
pub use pyrth_core::*;

#[pyclass(name = "StructureFunction")]
#[derive(Clone)]
struct PyStructureFunction {
    label: String,
    result: pyrth_core::EvaluationResult,
}

#[pymethods]
impl PyStructureFunction {
    #[getter]
    fn label(&self) -> String {
        self.label.clone()
    }

    #[getter]
    fn data_handlers(&self) -> Vec<String> {
        data_handlers_for_result(&self.result)
    }

    #[getter]
    fn time(&self) -> Vec<f64> {
        self.result.impedance.time.to_vec()
    }

    #[getter]
    fn impedance(&self) -> Vec<f64> {
        self.result.impedance.impedance.to_vec()
    }

    #[getter]
    fn log_time(&self) -> Vec<f64> {
        self.result.impedance.log_time.to_vec()
    }

    #[getter]
    fn time_spec(&self) -> Option<Vec<f64>> {
        self.result
            .time_spectrum
            .as_ref()
            .map(|time_spectrum| time_spectrum.to_vec())
    }

    #[getter]
    fn log_time_interp(&self) -> Option<Vec<f64>> {
        self.result
            .derivative
            .as_ref()
            .map(|derivative| derivative.log_time_interp.to_vec())
    }

    #[getter]
    fn imp_deriv_interp(&self) -> Option<Vec<f64>> {
        self.result
            .derivative
            .as_ref()
            .map(|derivative| derivative.imp_deriv_interp.to_vec())
    }

    #[getter]
    fn therm_resist_fost(&self) -> Option<Vec<f64>> {
        self.result
            .foster
            .as_ref()
            .map(|foster| foster.resistance.to_vec())
    }

    #[getter]
    fn therm_capa_fost(&self) -> Option<Vec<f64>> {
        self.result
            .foster
            .as_ref()
            .map(|foster| foster.capacitance.to_vec())
    }

    #[getter]
    fn cau_res(&self) -> Option<Vec<f64>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer.resistance.to_vec())
    }

    #[getter]
    fn cau_cap(&self) -> Option<Vec<f64>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer.capacitance.to_vec())
    }

    #[getter]
    fn int_cau_res(&self) -> Option<Vec<f64>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer.cumulative_resistance.to_vec())
    }

    #[getter]
    fn int_cau_cap(&self) -> Option<Vec<f64>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer.cumulative_capacitance.to_vec())
    }

    #[getter]
    fn diff_struc(&self) -> Option<Vec<f64>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer.differential_structure.to_vec())
    }

    #[getter]
    fn foster(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.result
            .foster
            .as_ref()
            .map(|foster| foster_network_to_dict(py, foster))
            .transpose()
    }

    #[getter]
    fn cauer(&self, py: Python<'_>) -> PyResult<Option<PyObject>> {
        self.result
            .cauer
            .as_ref()
            .map(|cauer| cauer_network_to_dict(py, cauer))
            .transpose()
    }

    fn to_dict(&self, py: Python<'_>) -> PyResult<PyObject> {
        evaluation_result_to_dict(py, self.result.clone())
    }

    fn keys(&self) -> Vec<String> {
        evaluation_result_keys(&self.result)
    }

    #[pyo3(signature = (key, default=None))]
    fn get(&self, py: Python<'_>, key: &str, default: Option<PyObject>) -> PyResult<PyObject> {
        match self.result_item(py, key)? {
            Some(value) => Ok(value),
            None => Ok(default.unwrap_or_else(|| py.None())),
        }
    }

    fn __getitem__(&self, py: Python<'_>, key: &str) -> PyResult<PyObject> {
        self.result_item(py, key)?
            .ok_or_else(|| PyKeyError::new_err(key.to_string()))
    }

    fn __contains__(&self, key: &str) -> bool {
        evaluation_result_keys(&self.result)
            .iter()
            .any(|candidate| candidate == key)
    }

    fn __len__(&self) -> usize {
        evaluation_result_keys(&self.result).len()
    }

    fn __iter__(&self, py: Python<'_>) -> PyResult<PyObject> {
        let keys = PyList::new(py, evaluation_result_keys(&self.result))?;
        Ok(keys.call_method0("__iter__")?.unbind())
    }

    fn __repr__(&self) -> String {
        format!(
            "StructureFunction(label={:?}, keys={:?})",
            self.label,
            evaluation_result_keys(&self.result)
        )
    }
}

impl PyStructureFunction {
    fn result_item(&self, py: Python<'_>, key: &str) -> PyResult<Option<PyObject>> {
        let output = evaluation_result_to_dict(py, self.result.clone())?;
        let output = output.bind(py).downcast::<PyDict>().map_err(|err| {
            PyValueError::new_err(format!("StructureFunction conversion failed: {err}"))
        })?;
        Ok(output.get_item(key)?.map(|value| value.unbind()))
    }
}

#[pyclass]
struct Evaluation {
    last_result: RwLock<Option<pyrth_core::EvaluationResult>>,
    modules: RwLock<HashMap<String, EvaluationModule>>,
    module_counters: RwLock<HashMap<String, usize>>,
}

#[derive(Clone)]
enum EvaluationModule {
    Structure(pyrth_core::EvaluationResult),
    TemperaturePrediction(TemperaturePredictionModule),
}

#[derive(Clone)]
struct TemperaturePredictionModule {
    time: Vec<f64>,
    temperature: Vec<f64>,
}

#[pymethods]
impl Evaluation {
    #[new]
    fn new() -> Self {
        Self {
            last_result: RwLock::new(None),
            modules: RwLock::new(HashMap::new()),
            module_counters: RwLock::new(HashMap::new()),
        }
    }

    fn standard(&self, py: Python<'_>, parameters: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        self.standard_module(py, parameters)
    }

    fn standard_module_set(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        if parameters.get_item("iterable_keywords")?.is_some() {
            return self.standard_module_sweep(py, parameters);
        }
        self.standard_module(py, parameters)
    }

    fn theoretical(&self, py: Python<'_>, parameters: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        self.theoretical_module(py, parameters)
    }

    fn theoretical_module(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        theoretical_module_from_parameters(py, parameters)
    }

    fn bootstrap(&self, py: Python<'_>, parameters: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        if let Some(data) = extract_pairs(parameters, "data")? {
            let input = pyrth_core::TransientInput::from_pairs(data)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            let repetitions = require_first_usize(parameters, &["repetitions"])?;
            let noise_std = bootstrap_noise_std_from_data(parameters, &input)?;
            let seed = extract_u64(parameters, "seed")?
                .or(extract_u64(parameters, "random_seed")?)
                .unwrap_or(0);
            let params = bootstrap_evaluation_params_from_parameters(parameters)?;
            let result = pyrth_core::bootstrap_from_impedance_data(
                &input,
                repetitions,
                noise_std,
                &params,
                seed,
            )
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
            return bootstrap_result_to_dict(py, result);
        }

        let resistance = require_first_vec_f64(
            parameters,
            &["resistance", "theoretical_resistance", "theo_resistances"],
        )?;
        let capacitance = require_first_vec_f64(
            parameters,
            &[
                "capacitance",
                "theoretical_capacitance",
                "theo_capacitances",
            ],
        )?;
        let (time_start, time_end) = theoretical_time_range(parameters)?;
        let time_size = extract_usize(parameters, "time_size")?
            .or(extract_usize(parameters, "theo_time_size")?)
            .ok_or_else(|| PyValueError::new_err("time_size or theo_time_size is required"))?;
        let repetitions = require_first_usize(parameters, &["repetitions"])?;
        let noise_std = bootstrap_noise_std(
            parameters,
            &resistance,
            &capacitance,
            time_start,
            time_end,
            time_size,
        )?;
        let seed = extract_u64(parameters, "seed")?
            .or(extract_u64(parameters, "random_seed")?)
            .unwrap_or(0);

        bootstrap_theoretical(
            py,
            resistance,
            capacitance,
            time_start,
            time_end,
            time_size,
            repetitions,
            noise_std,
            seed,
        )
    }

    fn optimization(&self, py: Python<'_>, parameters: &Bound<'_, PyDict>) -> PyResult<PyObject> {
        let result = optimization_result_from_parameters(parameters)?;
        optimization_result_to_dict(py, result)
    }

    fn temperature_prediction(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let prediction = predict_temperature_response_data_from_parameters(parameters)?;
        if let Some(label) = extract_string(parameters, "label")? {
            self.register_temperature_prediction_module(label, prediction.clone())?;
        }
        temperature_prediction_to_dict(py, prediction.time, prediction.temperature)
    }

    #[pyo3(signature = (reference, candidate=None))]
    fn comparison(
        &self,
        py: Python<'_>,
        reference: &Bound<'_, PyAny>,
        candidate: Option<&Bound<'_, PyAny>>,
    ) -> PyResult<PyObject> {
        let (reference, candidate) = match candidate {
            Some(candidate) => (reference.clone(), candidate.clone()),
            None => comparison_inputs_from_wrapper(reference)?,
        };
        let reference = evaluation_result_from_any_or_params(py, self, &reference)?;
        let candidate = evaluation_result_from_any_or_params(py, self, &candidate)?;
        let result = pyrth_core::compare_evaluations(&reference, &candidate)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        let output = PyDict::new(py);
        output.set_item("time_const_norm", result.time_const_norm)?;
        output.set_item("structure_norm", result.structure_norm)?;
        output.set_item("total_resistance_diff", result.total_resistance_diff)?;
        Ok(output.into())
    }

    fn comparison_module(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let evaluation_type = extract_string(parameters, "evaluation_type")?
            .ok_or_else(|| PyValueError::new_err("evaluation_type is required"))?;
        if evaluation_type.eq_ignore_ascii_case("bootstrap") {
            return self.comparison_module_bootstrap(py, parameters);
        }
        if evaluation_type.eq_ignore_ascii_case("optimization") {
            return self.comparison_module_optimization(py, parameters);
        }
        if !evaluation_type.eq_ignore_ascii_case("standard") {
            return Err(PyValueError::new_err(
                "comparison_module currently supports evaluation_type='standard', 'bootstrap', or 'optimization' only",
            ));
        }
        let base_label = extract_string(parameters, "label")?
            .ok_or_else(|| PyValueError::new_err("label is required"))?;
        let iterable_keywords = require_string_list(parameters, "iterable_keywords")?;
        if iterable_keywords.is_empty() {
            return Err(PyValueError::new_err(
                "iterable_keywords must contain at least one key",
            ));
        }

        let theoretical = theoretical_input_from_parameters(parameters)?;
        let data = transient_input_to_pairs(&theoretical);
        let mut iterables = Vec::with_capacity(iterable_keywords.len());
        for keyword in &iterable_keywords {
            iterables.push(require_object_list(parameters, keyword)?);
        }
        let set_len = iterables[0].len();
        if iterables.iter().any(|items| items.len() != set_len) {
            return Err(PyValueError::new_err(
                "Iterables do not have the same length",
            ));
        }

        let reference_parameters = comparison_variant_parameters(
            py,
            parameters,
            &iterable_keywords,
            &iterables,
            0,
            &data,
            format!("{base_label}_reference"),
        )?;
        let reference = evaluation_result_from_result_or_params(py, self, &reference_parameters)?;

        let mut time_const_comparison = Vec::with_capacity(set_len);
        let mut structure_comparison = Vec::with_capacity(set_len);
        let mut total_resist_diff = Vec::with_capacity(set_len);
        for index in 0..set_len {
            let candidate_parameters = comparison_variant_parameters(
                py,
                parameters,
                &iterable_keywords,
                &iterables,
                index,
                &data,
                format!("{base_label}_{}", index),
            )?;
            let candidate =
                evaluation_result_from_result_or_params(py, self, &candidate_parameters)?;
            let comparison = pyrth_core::compare_evaluations(&reference, &candidate)
                .map_err(|err| PyValueError::new_err(err.to_string()))?;
            time_const_comparison.push(comparison.time_const_norm);
            structure_comparison.push(comparison.structure_norm);
            total_resist_diff.push(comparison.total_resistance_diff);
        }

        let mod_values = iterables[0]
            .iter()
            .map(|value| value.clone_ref(py))
            .collect::<Vec<_>>();
        let output = PyDict::new(py);
        output.set_item("time_const_comparison", time_const_comparison)?;
        output.set_item("structure_comparison", structure_comparison)?;
        output.set_item("total_resist_diff", total_resist_diff)?;
        output.set_item("mod_key_display_name", iterable_keywords.join("_"))?;
        output.set_item("mod_value_list", PyList::new(py, mod_values)?)?;
        Ok(output.into())
    }

    fn standard_module(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let label = extract_string(parameters, "label")?.unwrap_or_else(|| "no_label".to_string());
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
        let extrapolate = extract_bool(parameters, "extrapolate")?;
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

        let (_output, result) = evaluate_transient_input_with_result(py, input, overrides)?;
        *self
            .last_result
            .write()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation result state"))? =
            Some(result);
        let result = self
            .last_result
            .read()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation result state"))?
            .as_ref()
            .cloned()
            .ok_or_else(|| PyValueError::new_err("failed to store Evaluation result state"))?;
        let final_label = self.register_module(label, result.clone())?;
        Ok(Py::new(
            py,
            PyStructureFunction {
                label: final_label,
                result,
            },
        )?
        .into_bound(py)
        .into_any()
        .unbind())
    }

    fn module_labels(&self) -> PyResult<Vec<String>> {
        let mut labels = self
            .modules
            .read()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .keys()
            .cloned()
            .collect::<Vec<_>>();
        labels.sort();
        Ok(labels)
    }

    fn module_count(&self) -> PyResult<usize> {
        Ok(self
            .modules
            .read()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .len())
    }

    fn module(&self, py: Python<'_>, label: &str) -> PyResult<Py<PyStructureFunction>> {
        let result = self
            .modules
            .read()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .get(label)
            .cloned()
            .ok_or_else(|| PyValueError::new_err(format!("module '{label}' was not found")))?;
        match result {
            EvaluationModule::Structure(result) => Py::new(
                py,
                PyStructureFunction {
                    label: label.to_string(),
                    result,
                },
            ),
            EvaluationModule::TemperaturePrediction(result) => Err(PyValueError::new_err(
                format!(
                    "module '{label}' is a temperature_prediction result with {} samples, not a StructureFunction",
                    result.time.len()
                ),
            )),
        }
    }

    fn standard_module_sweep(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let evaluation_type = extract_string(parameters, "evaluation_type")?
            .ok_or_else(|| PyValueError::new_err("evaluation_type is required"))?;
        if !evaluation_type.eq_ignore_ascii_case("standard") {
            return Err(PyValueError::new_err(
                "standard_module_set currently supports evaluation_type='standard' only",
            ));
        }
        let base_label = extract_string(parameters, "label")?
            .ok_or_else(|| PyValueError::new_err("label is required"))?;
        let iterable_keywords = require_string_list(parameters, "iterable_keywords")?;
        if iterable_keywords.is_empty() {
            return Err(PyValueError::new_err(
                "iterable_keywords must contain at least one key",
            ));
        }

        let mut iterables = Vec::with_capacity(iterable_keywords.len());
        for keyword in &iterable_keywords {
            iterables.push(require_object_list(parameters, keyword)?);
        }
        let set_len = iterables[0].len();
        if iterables.iter().any(|items| items.len() != set_len) {
            return Err(PyValueError::new_err(
                "Iterables do not have the same length",
            ));
        }

        let mut modules = Vec::with_capacity(set_len);
        for index in 0..set_len {
            let variant = clone_dict(py, parameters)?;
            let label_suffix = format!("{}_{}", iterable_keywords[0], index);
            variant.set_item("label", format!("{base_label}_{label_suffix}"))?;
            for (keyword, values) in iterable_keywords.iter().zip(iterables.iter()) {
                variant.set_item(keyword, values[index].clone_ref(py))?;
            }
            modules.push(self.standard_module(py, &variant)?);
        }

        Ok(PyList::new(py, modules)?.into())
    }

    fn comparison_module_optimization(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let base_label = extract_string(parameters, "label")?
            .ok_or_else(|| PyValueError::new_err("label is required"))?;
        let iterable_keywords = require_string_list(parameters, "iterable_keywords")?;
        if iterable_keywords.is_empty() {
            return Err(PyValueError::new_err(
                "iterable_keywords must contain at least one key",
            ));
        }

        let input = if let Some(data) = extract_pairs(parameters, "data")? {
            pyrth_core::TransientInput::from_pairs(data)
                .map_err(|err| PyValueError::new_err(err.to_string()))?
        } else {
            theoretical_input_from_parameters(parameters)?
        };
        let data = transient_input_to_pairs(&input);

        let mut iterables = Vec::with_capacity(iterable_keywords.len());
        for keyword in &iterable_keywords {
            iterables.push(require_object_list(parameters, keyword)?);
        }
        let set_len = iterables[0].len();
        if iterables.iter().any(|items| items.len() != set_len) {
            return Err(PyValueError::new_err(
                "Iterables do not have the same length",
            ));
        }

        let reference_parameters = comparison_variant_parameters(
            py,
            parameters,
            &iterable_keywords,
            &iterables,
            0,
            &data,
            format!("{base_label}_reference"),
        )?;
        let reference = optimization_result_from_parameters(&reference_parameters)?;

        let mut time_const_comparison = Vec::with_capacity(set_len);
        let mut structure_comparison = Vec::with_capacity(set_len);
        let mut total_resist_diff = Vec::with_capacity(set_len);
        for index in 0..set_len {
            let candidate_parameters = comparison_variant_parameters(
                py,
                parameters,
                &iterable_keywords,
                &iterables,
                index,
                &data,
                format!("{base_label}_{}", index),
            )?;
            let candidate = optimization_result_from_parameters(&candidate_parameters)?;
            time_const_comparison
                .push(optimization_impedance_norm(&input, &reference, &candidate)?);
            structure_comparison.push(0.0);
            total_resist_diff.push(
                (reference.parameters.resistance.iter().sum::<f64>()
                    - candidate.parameters.resistance.iter().sum::<f64>())
                .abs(),
            );
        }

        let mod_values = iterables[0]
            .iter()
            .map(|value| value.clone_ref(py))
            .collect::<Vec<_>>();
        let output = PyDict::new(py);
        output.set_item("time_const_comparison", time_const_comparison)?;
        output.set_item("structure_comparison", structure_comparison)?;
        output.set_item("total_resist_diff", total_resist_diff)?;
        output.set_item("mod_key_display_name", iterable_keywords.join("_"))?;
        output.set_item("mod_value_list", PyList::new(py, mod_values)?)?;
        Ok(output.into())
    }

    fn comparison_module_bootstrap(
        &self,
        py: Python<'_>,
        parameters: &Bound<'_, PyDict>,
    ) -> PyResult<PyObject> {
        let base_label = extract_string(parameters, "label")?
            .ok_or_else(|| PyValueError::new_err("label is required"))?;
        let iterable_keywords = require_string_list(parameters, "iterable_keywords")?;
        if iterable_keywords.is_empty() {
            return Err(PyValueError::new_err(
                "iterable_keywords must contain at least one key",
            ));
        }

        let base_input = if let Some(data) = extract_pairs(parameters, "data")? {
            Some(
                pyrth_core::TransientInput::from_pairs(data)
                    .map_err(|err| PyValueError::new_err(err.to_string()))?,
            )
        } else {
            None
        };
        let base_data = base_input.as_ref().map(transient_input_to_pairs);

        let mut iterables = Vec::with_capacity(iterable_keywords.len());
        for keyword in &iterable_keywords {
            iterables.push(require_object_list(parameters, keyword)?);
        }
        let set_len = iterables[0].len();
        if iterables.iter().any(|items| items.len() != set_len) {
            return Err(PyValueError::new_err(
                "Iterables do not have the same length",
            ));
        }

        let reference_parameters = comparison_bootstrap_variant_parameters(
            py,
            parameters,
            &iterable_keywords,
            &iterables,
            0,
            base_data.as_deref(),
            format!("{base_label}_reference"),
        )?;
        let reference = bootstrap_result_from_parameters(&reference_parameters)?;

        let mut time_const_comparison = Vec::with_capacity(set_len);
        let mut structure_comparison = Vec::with_capacity(set_len);
        let mut total_resist_diff = Vec::with_capacity(set_len);
        for index in 0..set_len {
            let candidate_parameters = comparison_bootstrap_variant_parameters(
                py,
                parameters,
                &iterable_keywords,
                &iterables,
                index,
                base_data.as_deref(),
                format!("{base_label}_{}", index),
            )?;
            let candidate = bootstrap_result_from_parameters(&candidate_parameters)?;
            time_const_comparison.push(bootstrap_time_spectrum_norm(&reference, &candidate)?);
            structure_comparison.push(0.0);
            total_resist_diff.push(bootstrap_total_resistance_diff(&reference, &candidate)?);
        }

        let mod_values = iterables[0]
            .iter()
            .map(|value| value.clone_ref(py))
            .collect::<Vec<_>>();
        let output = PyDict::new(py);
        output.set_item("time_const_comparison", time_const_comparison)?;
        output.set_item("structure_comparison", structure_comparison)?;
        output.set_item("total_resist_diff", total_resist_diff)?;
        output.set_item("mod_key_display_name", iterable_keywords.join("_"))?;
        output.set_item("mod_value_list", PyList::new(py, mod_values)?)?;
        Ok(output.into())
    }

    #[pyo3(signature = (output_dir="output/csv"))]
    fn save_as_csv(&self, py: Python<'_>, output_dir: &str) -> PyResult<PyObject> {
        self.export_registered_modules(py, output_dir, ExportKind::Csv)
    }

    #[pyo3(signature = (output_dir="output/figures"))]
    fn save_figures(&self, py: Python<'_>, output_dir: &str) -> PyResult<PyObject> {
        self.export_registered_modules(py, output_dir, ExportKind::Figure)
    }

    #[pyo3(signature = (output_dir="output"))]
    fn save_all(&self, py: Python<'_>, output_dir: &str) -> PyResult<PyObject> {
        let output = PyDict::new(py);
        output.set_item("csv", self.save_as_csv(py, &format!("{output_dir}/csv"))?)?;
        output.set_item(
            "figures",
            self.save_figures(py, &format!("{output_dir}/figures"))?,
        )?;
        Ok(output.into())
    }
}

#[derive(Clone, Copy)]
enum ExportKind {
    Csv,
    Figure,
}

impl Evaluation {
    fn export_registered_modules(
        &self,
        py: Python<'_>,
        output_dir: &str,
        export_kind: ExportKind,
    ) -> PyResult<PyObject> {
        let modules = self
            .modules
            .read()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .clone();
        if modules.is_empty() {
            return Err(PyValueError::new_err(
                "save_as_csv requires at least one previous module-producing call",
            ));
        };

        let output = PyDict::new(py);
        let mut labels = modules.keys().cloned().collect::<Vec<_>>();
        labels.sort();
        for label in labels {
            let result = modules
                .get(&label)
                .ok_or_else(|| PyValueError::new_err("failed to read Evaluation module"))?;
            let module_output_dir = Path::new(output_dir).join(&label);
            let files = match (result, export_kind) {
                (EvaluationModule::Structure(result), ExportKind::Csv) => {
                    let files = pyrth_core::export_csv(result, &module_output_dir)
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    exported_csv_files_to_dict(py, files)?
                }
                (EvaluationModule::Structure(result), ExportKind::Figure) => {
                    let files = pyrth_core::export_svg_figures(result, &module_output_dir)
                        .map_err(|err| PyValueError::new_err(err.to_string()))?;
                    exported_figure_files_to_dict(py, files)?
                }
                (EvaluationModule::TemperaturePrediction(result), ExportKind::Csv) => {
                    export_temperature_prediction_csv(py, result, &module_output_dir)?
                }
                (EvaluationModule::TemperaturePrediction(result), ExportKind::Figure) => {
                    export_temperature_prediction_svg(py, result, &module_output_dir)?
                }
            };
            output.set_item(label, files)?;
        }

        Ok(output.into())
    }

    fn register_module(
        &self,
        label: String,
        result: pyrth_core::EvaluationResult,
    ) -> PyResult<String> {
        let final_label = {
            let mut counters = self
                .module_counters
                .write()
                .map_err(|_| PyValueError::new_err("failed to lock Evaluation module counters"))?;
            let counter = counters.entry(label.clone()).or_insert(0);
            let final_label = if *counter == 0 {
                label
            } else {
                format!("{label}_{counter}")
            };
            *counter += 1;
            final_label
        };

        self.modules
            .write()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .insert(final_label.clone(), EvaluationModule::Structure(result));
        Ok(final_label)
    }

    fn register_temperature_prediction_module(
        &self,
        label: String,
        result: TemperaturePredictionModule,
    ) -> PyResult<String> {
        let final_label = self.next_module_label(label)?;
        self.modules
            .write()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation modules"))?
            .insert(
                final_label.clone(),
                EvaluationModule::TemperaturePrediction(result),
            );
        Ok(final_label)
    }

    fn next_module_label(&self, label: String) -> PyResult<String> {
        let mut counters = self
            .module_counters
            .write()
            .map_err(|_| PyValueError::new_err("failed to lock Evaluation module counters"))?;
        let counter = counters.entry(label.clone()).or_insert(0);
        let final_label = if *counter == 0 {
            label
        } else {
            format!("{label}_{counter}")
        };
        *counter += 1;
        Ok(final_label)
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

fn comparison_inputs_from_wrapper<'py>(
    parameters: &Bound<'py, PyAny>,
) -> PyResult<(Bound<'py, PyAny>, Bound<'py, PyAny>)> {
    let parameters = parameters.downcast::<PyDict>().map_err(|err| {
        PyValueError::new_err(format!(
            "comparison requires either (reference, candidate) arguments or one dict with reference and candidate: {err}"
        ))
    })?;
    let reference = require_first_item(
        parameters,
        &["reference", "reference_result", "reference_parameters"],
        "comparison requires either (reference, candidate) arguments or one dict with reference and candidate",
    )?;
    let candidate = require_first_item(
        parameters,
        &["candidate", "candidate_result", "candidate_parameters"],
        "comparison requires either (reference, candidate) arguments or one dict with reference and candidate",
    )?;

    Ok((reference, candidate))
}

fn comparison_variant_parameters<'py>(
    py: Python<'py>,
    parameters: &Bound<'py, PyDict>,
    iterable_keywords: &[String],
    iterables: &[Vec<PyObject>],
    index: usize,
    data: &[(f64, f64)],
    label: String,
) -> PyResult<Bound<'py, PyDict>> {
    let variant = clone_dict(py, parameters)?;
    variant.set_item("data", data.to_vec())?;
    variant.set_item("label", label)?;
    for (keyword, values) in iterable_keywords.iter().zip(iterables.iter()) {
        variant.set_item(keyword, values[index].clone_ref(py))?;
    }
    Ok(variant)
}

fn comparison_bootstrap_variant_parameters<'py>(
    py: Python<'py>,
    parameters: &Bound<'py, PyDict>,
    iterable_keywords: &[String],
    iterables: &[Vec<PyObject>],
    index: usize,
    data: Option<&[(f64, f64)]>,
    label: String,
) -> PyResult<Bound<'py, PyDict>> {
    let variant = clone_dict(py, parameters)?;
    variant.set_item("label", label)?;
    if let Some(data) = data {
        variant.set_item("data", data.to_vec())?;
    }
    for (keyword, values) in iterable_keywords.iter().zip(iterables.iter()) {
        variant.set_item(keyword, values[index].clone_ref(py))?;
    }
    Ok(variant)
}

fn theoretical_input_from_parameters(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::TransientInput> {
    let resistance = require_first_vec_f64(
        parameters,
        &["resistance", "theoretical_resistance", "theo_resistances"],
    )?;
    let capacitance = require_first_vec_f64(
        parameters,
        &[
            "capacitance",
            "theoretical_capacitance",
            "theo_capacitances",
        ],
    )?;
    let (time_start, time_end) = theoretical_time_range_or_default(parameters)?;
    let time_size = extract_usize(parameters, "time_size")?
        .or(extract_usize(parameters, "theo_time_size")?)
        .ok_or_else(|| PyValueError::new_err("time_size or theo_time_size is required"))?;

    let delta = extract_f64(parameters, "theo_delta")?
        .or(extract_f64(parameters, "delta")?)
        .unwrap_or(std::f64::consts::PI / 360.0);
    let result = pyrth_core::theoretical_module(
        &resistance,
        &capacitance,
        time_start,
        time_end,
        time_size,
        delta,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    pyrth_core::TransientInput::new(result.log_time.mapv(f64::exp), result.impedance)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn theoretical_time_range(parameters: &Bound<'_, PyDict>) -> PyResult<(f64, f64)> {
    if let Some(theo_time) = extract_vec_f64(parameters, "theo_time")? {
        if theo_time.len() != 2 {
            return Err(PyValueError::new_err(
                "theo_time must contain exactly two values",
            ));
        }
        return Ok((theo_time[0], theo_time[1]));
    }

    Ok((
        require_first_f64(parameters, &["time_start"])?,
        require_first_f64(parameters, &["time_end"])?,
    ))
}

fn theoretical_time_range_or_default(parameters: &Bound<'_, PyDict>) -> PyResult<(f64, f64)> {
    if let Some(theo_time) = extract_vec_f64(parameters, "theo_time")? {
        if theo_time.len() != 2 {
            return Err(PyValueError::new_err(
                "theo_time must contain exactly two values",
            ));
        }
        return Ok((theo_time[0], theo_time[1]));
    }
    let time_start = extract_f64(parameters, "time_start")?;
    let time_end = extract_f64(parameters, "time_end")?;
    Ok((time_start.unwrap_or(4e-8), time_end.unwrap_or(1e3)))
}

fn bootstrap_noise_std(
    parameters: &Bound<'_, PyDict>,
    resistance: &[f64],
    capacitance: &[f64],
    time_start: f64,
    time_end: f64,
    time_size: usize,
) -> PyResult<f64> {
    if let Some(noise_std) = extract_f64(parameters, "noise_std")? {
        return Ok(noise_std);
    }
    let signal_to_noise_ratio = require_first_f64(parameters, &["signal_to_noise_ratio"])?;
    if !signal_to_noise_ratio.is_finite() || signal_to_noise_ratio <= 0.0 {
        return Err(PyValueError::new_err(
            "signal_to_noise_ratio must be finite and greater than zero",
        ));
    }
    let input = pyrth_core::foster_step_response_input(
        resistance,
        capacitance,
        time_start,
        time_end,
        time_size,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let last_impedance = input
        .value
        .last()
        .copied()
        .ok_or_else(|| PyValueError::new_err("theoretical impedance is empty"))?;
    Ok(last_impedance / signal_to_noise_ratio)
}

fn bootstrap_noise_std_from_data(
    parameters: &Bound<'_, PyDict>,
    input: &pyrth_core::TransientInput,
) -> PyResult<f64> {
    if let Some(noise_std) = extract_f64(parameters, "noise_std")? {
        return Ok(noise_std);
    }
    let signal_to_noise_ratio = require_first_f64(parameters, &["signal_to_noise_ratio"])?;
    if !signal_to_noise_ratio.is_finite() || signal_to_noise_ratio <= 0.0 {
        return Err(PyValueError::new_err(
            "signal_to_noise_ratio must be finite and greater than zero",
        ));
    }
    let last_impedance = input
        .value
        .last()
        .copied()
        .ok_or_else(|| PyValueError::new_err("bootstrap data is empty"))?;
    Ok(last_impedance / signal_to_noise_ratio)
}

fn bootstrap_evaluation_params_from_parameters(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::EvaluationParams> {
    let mut params = pyrth_core::EvaluationParams::default();
    params.calc_struc = extract_bool(parameters, "calc_struc")?.unwrap_or(false);
    if let Some(deconv_mode) =
        extract_string(parameters, "deconv_mode")?.or(extract_string(parameters, "deconv")?)
    {
        params.deconv_mode = pyrth_core::DeconvMode::from_label(&deconv_mode)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(log_time_size) = extract_usize(parameters, "log_time_size")? {
        params.log_time_size = log_time_size;
    }
    if let Some(min_index) = extract_usize(parameters, "min_index")? {
        params.min_index = min_index;
    }
    if let Some(minimum_window_size) = extract_usize(parameters, "minimum_window_size")? {
        params.minimum_window_size = minimum_window_size;
    }
    if let Some(bay_steps) = extract_usize(parameters, "bay_steps")? {
        params.bay_steps = bay_steps;
    }
    if let Some(filter_name) = extract_string(parameters, "filter_name")? {
        params.filter_name = pyrth_core::FourierFilter::from_label(&filter_name)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
    }
    if let Some(filter_range) = extract_f64(parameters, "filter_range")? {
        params.filter_range = filter_range;
    }
    if let Some(filter_parameter) = extract_f64(parameters, "filter_parameter")? {
        params.filter_parameter = filter_parameter;
    }
    Ok(params)
}

fn bootstrap_result_from_parameters(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::BootstrapResult> {
    let repetitions = require_first_usize(parameters, &["repetitions"])?;
    let seed = extract_u64(parameters, "seed")?
        .or(extract_u64(parameters, "random_seed")?)
        .unwrap_or(0);
    let params = bootstrap_evaluation_params_from_parameters(parameters)?;

    if let Some(data) = extract_pairs(parameters, "data")? {
        let input = pyrth_core::TransientInput::from_pairs(data)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let noise_std = bootstrap_noise_std_from_data(parameters, &input)?;
        return pyrth_core::bootstrap_from_impedance_data(
            &input,
            repetitions,
            noise_std,
            &params,
            seed,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()));
    }

    let resistance = require_first_vec_f64(
        parameters,
        &["resistance", "theoretical_resistance", "theo_resistances"],
    )?;
    let capacitance = require_first_vec_f64(
        parameters,
        &[
            "capacitance",
            "theoretical_capacitance",
            "theo_capacitances",
        ],
    )?;
    let (time_start, time_end) = theoretical_time_range_or_default(parameters)?;
    let time_size = extract_usize(parameters, "time_size")?
        .or(extract_usize(parameters, "theo_time_size")?)
        .ok_or_else(|| PyValueError::new_err("time_size or theo_time_size is required"))?;
    let noise_std = bootstrap_noise_std(
        parameters,
        &resistance,
        &capacitance,
        time_start,
        time_end,
        time_size,
    )?;
    let model = pyrth_core::FosterStepResponseModel::from_slices(&resistance, &capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    pyrth_core::bootstrap_from_foster_step_response(
        &model,
        time_start,
        time_end,
        time_size,
        repetitions,
        noise_std,
        &params,
        seed,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn bootstrap_time_spectrum_norm(
    reference: &pyrth_core::BootstrapResult,
    candidate: &pyrth_core::BootstrapResult,
) -> PyResult<f64> {
    pyrth_core::relative_l2_norm(&reference.time_spectrum_mean, &candidate.time_spectrum_mean)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn bootstrap_total_resistance_diff(
    reference: &pyrth_core::BootstrapResult,
    candidate: &pyrth_core::BootstrapResult,
) -> PyResult<f64> {
    let reference_total = reference
        .impedance_mean
        .last()
        .copied()
        .ok_or_else(|| PyValueError::new_err("reference bootstrap impedance_mean is empty"))?;
    let candidate_total = candidate
        .impedance_mean
        .last()
        .copied()
        .ok_or_else(|| PyValueError::new_err("candidate bootstrap impedance_mean is empty"))?;
    Ok((reference_total - candidate_total).abs())
}

fn transient_input_to_pairs(input: &pyrth_core::TransientInput) -> Vec<(f64, f64)> {
    input
        .time
        .iter()
        .zip(input.value.iter())
        .map(|(time, value)| (*time, *value))
        .collect()
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
            extrapolate: None,
            lower_fit_limit: None,
            upper_fit_limit: None,
        },
    )
}

fn theoretical_module_from_parameters(
    py: Python<'_>,
    parameters: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let resistance = require_first_vec_f64(
        parameters,
        &["theo_resistances", "resistance", "theoretical_resistance"],
    )?;
    let capacitance = require_first_vec_f64(
        parameters,
        &[
            "theo_capacitances",
            "capacitance",
            "theoretical_capacitance",
        ],
    )?;
    let (time_start, time_end) = theoretical_time_range_or_default(parameters)?;
    let time_size = extract_usize(parameters, "theo_time_size")?
        .or(extract_usize(parameters, "time_size")?)
        .unwrap_or(30000);
    let delta = extract_f64(parameters, "theo_delta")?
        .or(extract_f64(parameters, "delta")?)
        .unwrap_or(std::f64::consts::PI / 360.0);
    let result = pyrth_core::theoretical_module(
        &resistance,
        &capacitance,
        time_start,
        time_end,
        time_size,
        delta,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;
    theoretical_structure_result_to_dict(py, result)
}

#[pyfunction]
fn foster_step_response(
    py: Python<'_>,
    resistance: Vec<f64>,
    capacitance: Vec<f64>,
    time_start: f64,
    time_end: f64,
    time_size: usize,
) -> PyResult<PyObject> {
    let model = pyrth_core::FosterStepResponseModel::from_slices(&resistance, &capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let input = model
        .to_transient_input(time_start, time_end, time_size)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let output = PyDict::new(py);
    output.set_item("time", input.time.to_vec())?;
    output.set_item("impedance", input.value.to_vec())?;
    Ok(output.into())
}

#[pyfunction]
fn theoretical_impedance(
    py: Python<'_>,
    resistance: Vec<f64>,
    capacitance: Vec<f64>,
    time_start: f64,
    time_end: f64,
    time_size: usize,
) -> PyResult<PyObject> {
    foster_step_response(py, resistance, capacitance, time_start, time_end, time_size)
}

#[pyfunction]
#[pyo3(signature = (resistance, capacitance, time_start, time_end, time_size, repetitions, noise_std, seed=0))]
fn bootstrap_theoretical(
    py: Python<'_>,
    resistance: Vec<f64>,
    capacitance: Vec<f64>,
    time_start: f64,
    time_end: f64,
    time_size: usize,
    repetitions: usize,
    noise_std: f64,
    seed: u64,
) -> PyResult<PyObject> {
    let model = pyrth_core::FosterStepResponseModel::from_slices(&resistance, &capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let mut params = pyrth_core::EvaluationParams::default();
    params.calc_struc = false;
    let result = pyrth_core::bootstrap_from_foster_step_response(
        &model,
        time_start,
        time_end,
        time_size,
        repetitions,
        noise_std,
        &params,
        seed,
    )
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    bootstrap_result_to_dict(py, result)
}

#[pyfunction]
#[pyo3(signature = (
    data,
    initial_resistance,
    initial_capacitance,
    lower_resistance,
    lower_capacitance,
    upper_resistance,
    upper_capacitance,
    max_iter=128,
    initial_step=0.1,
    min_step=1e-6,
    shrink_factor=0.5
))]
fn optimize_rc(
    py: Python<'_>,
    data: Vec<(f64, f64)>,
    initial_resistance: Vec<f64>,
    initial_capacitance: Vec<f64>,
    lower_resistance: Vec<f64>,
    lower_capacitance: Vec<f64>,
    upper_resistance: Vec<f64>,
    upper_capacitance: Vec<f64>,
    max_iter: usize,
    initial_step: f64,
    min_step: f64,
    shrink_factor: f64,
) -> PyResult<PyObject> {
    let input = pyrth_core::TransientInput::from_pairs(data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let initial = pyrth_core::RcParameters::from_slices(&initial_resistance, &initial_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let lower = pyrth_core::RcParameters::from_slices(&lower_resistance, &lower_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let upper = pyrth_core::RcParameters::from_slices(&upper_resistance, &upper_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let bounds = pyrth_core::RcParameterBounds::new(lower, upper)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let config = pyrth_core::OptimizationConfig {
        max_iter,
        initial_step,
        min_step,
        shrink_factor,
    };

    let result = pyrth_core::optimize_rc_parameters(&input, &initial, &bounds, config)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let output = PyDict::new(py);
    output.set_item("resistance", result.parameters.resistance.to_vec())?;
    output.set_item("capacitance", result.parameters.capacitance.to_vec())?;
    output.set_item("residual_norm", result.residual_norm)?;
    output.set_item("iterations", result.iterations)?;
    Ok(output.into())
}

fn optimization_result_from_parameters(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::OptimizationResult> {
    let data = require_pairs(parameters, "data")?;
    let initial_resistance = require_first_vec_f64(parameters, &["initial_resistance"])?;
    let initial_capacitance = require_first_vec_f64(parameters, &["initial_capacitance"])?;
    let lower_resistance = require_first_vec_f64(parameters, &["lower_resistance"])?;
    let lower_capacitance = require_first_vec_f64(parameters, &["lower_capacitance"])?;
    let upper_resistance = require_first_vec_f64(parameters, &["upper_resistance"])?;
    let upper_capacitance = require_first_vec_f64(parameters, &["upper_capacitance"])?;
    let max_iter = extract_usize(parameters, "max_iter")?.unwrap_or(128);
    let initial_step = extract_f64(parameters, "initial_step")?.unwrap_or(0.1);
    let min_step = extract_f64(parameters, "min_step")?.unwrap_or(1e-6);
    let shrink_factor = extract_f64(parameters, "shrink_factor")?.unwrap_or(0.5);

    let input = pyrth_core::TransientInput::from_pairs(data)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let initial = pyrth_core::RcParameters::from_slices(&initial_resistance, &initial_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let lower = pyrth_core::RcParameters::from_slices(&lower_resistance, &lower_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let upper = pyrth_core::RcParameters::from_slices(&upper_resistance, &upper_capacitance)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let bounds = pyrth_core::RcParameterBounds::new(lower, upper)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let config = pyrth_core::OptimizationConfig {
        max_iter,
        initial_step,
        min_step,
        shrink_factor,
    };

    pyrth_core::optimize_rc_parameters(&input, &initial, &bounds, config)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

fn optimization_result_to_dict(
    py: Python<'_>,
    result: pyrth_core::OptimizationResult,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("resistance", result.parameters.resistance.to_vec())?;
    output.set_item("capacitance", result.parameters.capacitance.to_vec())?;
    output.set_item("residual_norm", result.residual_norm)?;
    output.set_item("iterations", result.iterations)?;
    Ok(output.into())
}

fn optimization_impedance_norm(
    input: &pyrth_core::TransientInput,
    reference: &pyrth_core::OptimizationResult,
    candidate: &pyrth_core::OptimizationResult,
) -> PyResult<f64> {
    let reference_impedance = reference
        .parameters
        .impedance_on(&input.time)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let candidate_impedance = candidate
        .parameters
        .impedance_on(&input.time)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    pyrth_core::relative_l2_norm(&reference_impedance, &candidate_impedance)
        .map_err(|err| PyValueError::new_err(err.to_string()))
}

#[pyfunction]
#[pyo3(signature = (source, power_data=None, lin_sampling_period=1.0))]
fn predict_temperature_response(
    py: Python<'_>,
    source: &Bound<'_, PyAny>,
    power_data: Option<Vec<(f64, f64)>>,
    lin_sampling_period: f64,
) -> PyResult<PyObject> {
    if let Ok(parameters) = source.downcast::<PyDict>() {
        return predict_temperature_response_from_parameters(py, parameters);
    }

    let impulse_response = source.extract::<Vec<(f64, f64)>>().map_err(|err| {
        PyValueError::new_err(format!(
            "impulse_response must be a sequence of pairs: {err}"
        ))
    })?;
    let power_data = power_data.ok_or_else(|| PyValueError::new_err("power_data is required"))?;
    let input = pyrth_core::TransientInput::from_pairs(impulse_response)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;
    let power = transient_input_from_pairs_unchecked(power_data);
    let mut params = pyrth_core::EvaluationParams::default();
    params.calc_struc = false;

    let result = pyrth_core::predict_temperature(input, power, &params, lin_sampling_period)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    temperature_prediction_to_dict(
        py,
        result.lin_time.to_vec(),
        result.predicted_temperature.to_vec(),
    )
}

fn predict_temperature_response_from_parameters(
    py: Python<'_>,
    parameters: &Bound<'_, PyDict>,
) -> PyResult<PyObject> {
    let prediction = predict_temperature_response_data_from_parameters(parameters)?;
    temperature_prediction_to_dict(py, prediction.time, prediction.temperature)
}

fn predict_temperature_response_data_from_parameters(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<TemperaturePredictionModule> {
    let power_data = require_first_pairs(parameters, &["power_data", "power"])?;
    let lin_sampling_period = extract_f64(parameters, "lin_sampling_period")?.unwrap_or(1.0);

    if let Some(impulse_response) = extract_first_pairs(parameters, &["impulse_response"])? {
        let input = pyrth_core::TransientInput::from_pairs(impulse_response)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let power = transient_input_from_pairs_unchecked(power_data);
        let mut params = pyrth_core::EvaluationParams::default();
        params.calc_struc = false;

        let result = pyrth_core::predict_temperature(input, power, &params, lin_sampling_period)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        return Ok(TemperaturePredictionModule {
            time: result.lin_time.to_vec(),
            temperature: result.predicted_temperature.to_vec(),
        });
    }

    if parameters.get_item("data")?.is_some()
        && parameters.get_item("initial_resistance")?.is_some()
    {
        let reference_time = require_first_vec_f64(parameters, &["reference_time"])?;
        let power = transient_input_from_pairs_unchecked(power_data);
        let optimization = optimization_result_from_parameters(parameters)?;
        let result = pyrth_core::predict_temperature_from_optimization_result(
            &power,
            &optimization,
            &reference_time.into(),
            lin_sampling_period,
        )
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

        return Ok(TemperaturePredictionModule {
            time: result.lin_time.to_vec(),
            temperature: result.predicted_temperature.to_vec(),
        });
    }

    if let Some(impulse_response) = extract_first_pairs(parameters, &["data"])? {
        let input = pyrth_core::TransientInput::from_pairs(impulse_response)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let power = transient_input_from_pairs_unchecked(power_data);
        let mut params = pyrth_core::EvaluationParams::default();
        params.calc_struc = false;

        let result = pyrth_core::predict_temperature(input, power, &params, lin_sampling_period)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;

        return Ok(TemperaturePredictionModule {
            time: result.lin_time.to_vec(),
            temperature: result.predicted_temperature.to_vec(),
        });
    }

    let reference_time = require_first_vec_f64(parameters, &["reference_time"])?;
    let power = transient_input_from_pairs_unchecked(power_data);

    let result = if let Some(optimization_result) = parameters.get_item("optimization_result")? {
        let optimization_result = optimization_result.downcast::<PyDict>().map_err(|err| {
            PyValueError::new_err(format!("optimization_result must be a dict: {err}"))
        })?;
        let resistance = require_first_vec_f64(optimization_result, &["resistance"])?;
        let capacitance = require_first_vec_f64(optimization_result, &["capacitance"])?;
        let rc_parameters = pyrth_core::RcParameters::from_slices(&resistance, &capacitance)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        let optimization = pyrth_core::OptimizationResult {
            parameters: rc_parameters,
            residual_norm: extract_f64(optimization_result, "residual_norm")?.unwrap_or(0.0),
            iterations: extract_usize(optimization_result, "iterations")?.unwrap_or(0),
        };
        pyrth_core::predict_temperature_from_optimization_result(
            &power,
            &optimization,
            &reference_time.into(),
            lin_sampling_period,
        )
    } else {
        let resistance = require_first_vec_f64(parameters, &["resistance"])?;
        let capacitance = require_first_vec_f64(parameters, &["capacitance"])?;
        let rc_parameters = pyrth_core::RcParameters::from_slices(&resistance, &capacitance)
            .map_err(|err| PyValueError::new_err(err.to_string()))?;
        pyrth_core::predict_temperature_from_rc_parameters(
            &power,
            &rc_parameters,
            &reference_time.into(),
            lin_sampling_period,
        )
    }
    .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(TemperaturePredictionModule {
        time: result.lin_time.to_vec(),
        temperature: result.predicted_temperature.to_vec(),
    })
}

fn temperature_prediction_to_dict(
    py: Python<'_>,
    time: Vec<f64>,
    temperature: Vec<f64>,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("time", time)?;
    output.set_item("temperature", temperature)?;
    Ok(output.into())
}

fn export_temperature_prediction_csv(
    py: Python<'_>,
    result: &TemperaturePredictionModule,
    output_dir: &Path,
) -> PyResult<PyObject> {
    fs::create_dir_all(output_dir).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let path = output_dir.join("temperature_prediction.csv");
    let mut csv = String::from("time,temperature\n");
    for (time, temperature) in result.time.iter().zip(&result.temperature) {
        csv.push_str(&format!("{time:.17e},{temperature:.17e}\n"));
    }
    fs::write(&path, csv).map_err(|err| PyValueError::new_err(err.to_string()))?;

    let output = PyDict::new(py);
    output.set_item("temperature_prediction", path.to_string_lossy().to_string())?;
    Ok(output.into())
}

fn export_temperature_prediction_svg(
    py: Python<'_>,
    result: &TemperaturePredictionModule,
    output_dir: &Path,
) -> PyResult<PyObject> {
    fs::create_dir_all(output_dir).map_err(|err| PyValueError::new_err(err.to_string()))?;
    let path = output_dir.join("temperature_prediction.svg");
    let points = result
        .time
        .iter()
        .zip(&result.temperature)
        .filter(|(time, temperature)| time.is_finite() && temperature.is_finite())
        .map(|(time, temperature)| (*time, *temperature))
        .collect::<Vec<_>>();
    fs::write(&path, render_temperature_prediction_svg(&points))
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    let output = PyDict::new(py);
    output.set_item("temperature_prediction", path.to_string_lossy().to_string())?;
    Ok(output.into())
}

fn render_temperature_prediction_svg(points: &[(f64, f64)]) -> String {
    const WIDTH: f64 = 720.0;
    const HEIGHT: f64 = 420.0;
    const LEFT: f64 = 72.0;
    const TOP: f64 = 42.0;
    const RIGHT: f64 = 24.0;
    const BOTTOM: f64 = 58.0;

    let plot_width = WIDTH - LEFT - RIGHT;
    let plot_height = HEIGHT - TOP - BOTTOM;
    let (min_x, max_x, min_y, max_y) = bounds_for_pairs(points);
    let polyline = points
        .iter()
        .map(|(x, y)| {
            let px = LEFT + normalize_svg_value(*x, min_x, max_x) * plot_width;
            let py = TOP + (1.0 - normalize_svg_value(*y, min_y, max_y)) * plot_height;
            format!("{px:.3},{py:.3}")
        })
        .collect::<Vec<_>>()
        .join(" ");

    format!(
        r##"<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {WIDTH:.0} {HEIGHT:.0}" role="img" data-x-scale="linear" data-y-scale="linear">
  <title>Temperature prediction</title>
  <rect width="100%" height="100%" fill="#ffffff"/>
  <text x="{LEFT:.0}" y="24" font-family="sans-serif" font-size="18" fill="#111111">Temperature prediction</text>
  <line x1="{LEFT:.0}" y1="{:.0}" x2="{:.0}" y2="{:.0}" stroke="#222222" stroke-width="1"/>
  <line x1="{LEFT:.0}" y1="{TOP:.0}" x2="{LEFT:.0}" y2="{:.0}" stroke="#222222" stroke-width="1"/>
  <polyline fill="none" stroke="#b45309" stroke-width="2" points="{}"/>
  <text x="{:.0}" y="{:.0}" font-family="sans-serif" font-size="12" fill="#333333">time</text>
  <text x="12" y="{:.0}" font-family="sans-serif" font-size="12" fill="#333333" transform="rotate(-90 12,{:.0})">temperature</text>
</svg>
"##,
        TOP + plot_height,
        LEFT + plot_width,
        TOP + plot_height,
        TOP + plot_height,
        polyline,
        LEFT + plot_width / 2.0,
        HEIGHT - 16.0,
        TOP + plot_height / 2.0,
        TOP + plot_height / 2.0,
    )
}

fn bounds_for_pairs(points: &[(f64, f64)]) -> (f64, f64, f64, f64) {
    if points.is_empty() {
        return (0.0, 1.0, 0.0, 1.0);
    }

    let (mut min_x, mut max_x) = (points[0].0, points[0].0);
    let (mut min_y, mut max_y) = (points[0].1, points[0].1);
    for (x, y) in points.iter().copied() {
        min_x = min_x.min(x);
        max_x = max_x.max(x);
        min_y = min_y.min(y);
        max_y = max_y.max(y);
    }
    if min_x == max_x {
        min_x -= 0.5;
        max_x += 0.5;
    }
    if min_y == max_y {
        min_y -= 0.5;
        max_y += 0.5;
    }
    (min_x, max_x, min_y, max_y)
}

fn normalize_svg_value(value: f64, min: f64, max: f64) -> f64 {
    ((value - min) / (max - min)).clamp(0.0, 1.0)
}

fn transient_input_from_pairs_unchecked(pairs: Vec<(f64, f64)>) -> pyrth_core::TransientInput {
    let (time, value): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
    pyrth_core::TransientInput {
        time: time.into(),
        value: value.into(),
    }
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
    extrapolate: Option<bool>,
    lower_fit_limit: Option<f64>,
    upper_fit_limit: Option<f64>,
}

fn evaluation_result_from_result_or_params(
    py: Python<'_>,
    evaluation: &Evaluation,
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::EvaluationResult> {
    if looks_like_evaluation_result(parameters)? {
        return evaluation_result_from_dict(parameters);
    }

    let output = evaluation.standard_module(py, parameters)?;
    if let Ok(module) = output.bind(py).extract::<PyRef<'_, PyStructureFunction>>() {
        return Ok(module.result.clone());
    }
    let output = output.bind(py).downcast::<PyDict>().map_err(|err| {
        PyValueError::new_err(format!(
            "comparison parameters did not evaluate to a dict: {err}"
        ))
    })?;
    evaluation_result_from_dict(output)
}

fn evaluation_result_from_any_or_params(
    py: Python<'_>,
    evaluation: &Evaluation,
    source: &Bound<'_, PyAny>,
) -> PyResult<pyrth_core::EvaluationResult> {
    if let Ok(module) = source.extract::<PyRef<'_, PyStructureFunction>>() {
        return Ok(module.result.clone());
    }

    let parameters = source.downcast::<PyDict>().map_err(|err| {
        PyValueError::new_err(format!(
            "comparison input must be a dict or StructureFunction: {err}"
        ))
    })?;
    evaluation_result_from_result_or_params(py, evaluation, parameters)
}

fn looks_like_evaluation_result(parameters: &Bound<'_, PyDict>) -> PyResult<bool> {
    if parameters.get_item("data")?.is_some()
        || parameters.get_item("input")?.is_some()
        || parameters.get_item("infile")?.is_some()
    {
        return Ok(false);
    }

    Ok(parameters.get_item("time_spec")?.is_some()
        || parameters.get_item("time_spectrum")?.is_some()
        || parameters.get_item("foster")?.is_some()
        || parameters.get_item("cauer")?.is_some()
        || (parameters.get_item("time")?.is_some() && parameters.get_item("impedance")?.is_some()))
}

fn evaluation_result_from_dict(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<pyrth_core::EvaluationResult> {
    let time = extract_vec_f64(parameters, "time")?.unwrap_or_default();
    let impedance = extract_vec_f64(parameters, "impedance")?.unwrap_or_default();
    let log_time = extract_vec_f64(parameters, "log_time")?.unwrap_or_default();
    let time_spectrum =
        extract_first_vec_f64(parameters, &["time_spec", "time_spectrum"])?.map(Into::into);
    let foster = extract_foster_network(parameters)?;
    let cauer = extract_cauer_network(parameters)?;

    Ok(pyrth_core::EvaluationResult {
        impedance: pyrth_core::ImpedanceData {
            time: time.into(),
            impedance: impedance.into(),
            log_time: log_time.into(),
        },
        derivative: None,
        time_spectrum,
        foster,
        cauer,
    })
}

fn extract_foster_network(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<Option<pyrth_core::FosterNetwork>> {
    let Some(value) = parameters.get_item("foster")? else {
        return Ok(None);
    };
    let foster = value
        .downcast::<PyDict>()
        .map_err(|err| PyValueError::new_err(format!("foster must be a dict: {err}")))?;
    let resistance = require_first_vec_f64(foster, &["resistance"])?;
    let capacitance = require_first_vec_f64(foster, &["capacitance"])?;
    let tau = require_first_vec_f64(foster, &["tau"])?;

    Ok(Some(pyrth_core::FosterNetwork {
        resistance: resistance.into(),
        capacitance: capacitance.into(),
        tau: tau.into(),
    }))
}

fn extract_cauer_network(
    parameters: &Bound<'_, PyDict>,
) -> PyResult<Option<pyrth_core::CauerNetwork>> {
    let Some(value) = parameters.get_item("cauer")? else {
        return Ok(None);
    };
    let cauer = value
        .downcast::<PyDict>()
        .map_err(|err| PyValueError::new_err(format!("cauer must be a dict: {err}")))?;
    let resistance = require_first_vec_f64(cauer, &["resistance"])?;
    let capacitance = require_first_vec_f64(cauer, &["capacitance"])?;
    let cumulative_resistance = require_first_vec_f64(cauer, &["cumulative_resistance"])?;
    let cumulative_capacitance = require_first_vec_f64(cauer, &["cumulative_capacitance"])?;
    let differential_structure = require_first_vec_f64(cauer, &["differential_structure"])?;

    Ok(Some(pyrth_core::CauerNetwork {
        resistance: resistance.into(),
        capacitance: capacitance.into(),
        cumulative_resistance: cumulative_resistance.into(),
        cumulative_capacitance: cumulative_capacitance.into(),
        differential_structure: differential_structure.into(),
    }))
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

fn evaluate_transient_input_with_result(
    py: Python<'_>,
    input: pyrth_core::TransientInput,
    overrides: EvalOverrides,
) -> PyResult<(PyObject, pyrth_core::EvaluationResult)> {
    let result = evaluate_result_with_input(input, overrides)?;
    let output = evaluation_result_to_dict(py, result.clone())?;
    Ok((output, result))
}

fn evaluate_impedance_with_input(
    py: Python<'_>,
    input: pyrth_core::TransientInput,
    overrides: EvalOverrides,
) -> PyResult<PyObject> {
    let result = evaluate_result_with_input(input, overrides)?;
    evaluation_result_to_dict(py, result)
}

fn evaluate_result_with_input(
    input: pyrth_core::TransientInput,
    overrides: EvalOverrides,
) -> PyResult<pyrth_core::EvaluationResult> {
    let mut params = pyrth_core::EvaluationParams::python_compatible();
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
    if let Some(extrapolate) = overrides.extrapolate {
        params.extrapolate = extrapolate;
    }
    params.lower_fit_limit = overrides.lower_fit_limit;
    params.upper_fit_limit = overrides.upper_fit_limit;

    let result = pyrth_core::evaluate(input, &params)
        .map_err(|err| PyValueError::new_err(err.to_string()))?;

    Ok(result)
}

fn evaluation_result_to_dict(
    py: Python<'_>,
    result: pyrth_core::EvaluationResult,
) -> PyResult<PyObject> {
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
        output.set_item("imp_deriv_interp", derivative.imp_deriv_interp.to_vec())?;
        output.set_item("log_time_interp", derivative.log_time_interp.to_vec())?;
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
        output.set_item("therm_resist_fost", foster.resistance.to_vec())?;
        output.set_item("therm_capa_fost", foster.capacitance.to_vec())?;
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
        output.set_item("cau_res", cauer.resistance.to_vec())?;
        output.set_item("cau_cap", cauer.capacitance.to_vec())?;
        output.set_item("int_cau_res", cauer.cumulative_resistance.to_vec())?;
        output.set_item("int_cau_cap", cauer.cumulative_capacitance.to_vec())?;
        output.set_item("diff_struc", cauer.differential_structure.to_vec())?;
    }

    Ok(output.into())
}

fn theoretical_structure_result_to_dict(
    py: Python<'_>,
    result: pyrth_core::TheoreticalStructureResult,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("theo_log_time", result.log_time.to_vec())?;
    output.set_item("theo_int_cau_res", result.cumulative_resistance.to_vec())?;
    output.set_item("theo_int_cau_cap", result.cumulative_capacitance.to_vec())?;
    output.set_item("theo_diff_struc", result.differential_structure.to_vec())?;
    output.set_item("theo_time_const", result.time_const_spectrum.to_vec())?;
    output.set_item("theo_imp_deriv", result.impedance_derivative.to_vec())?;
    output.set_item("theo_impedance", result.impedance.to_vec())?;
    output.set_item("time", result.log_time.mapv(f64::exp).to_vec())?;
    output.set_item("impedance", result.impedance.to_vec())?;
    Ok(output.into())
}

fn bootstrap_result_to_dict(
    py: Python<'_>,
    result: pyrth_core::BootstrapResult,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("impedance_mean", result.impedance_mean.to_vec())?;
    output.set_item("impedance_p10", result.impedance_p10.to_vec())?;
    output.set_item("impedance_median", result.impedance_median.to_vec())?;
    output.set_item("impedance_p90", result.impedance_p90.to_vec())?;
    output.set_item("time_spectrum_mean", result.time_spectrum_mean.to_vec())?;
    output.set_item("time_spectrum_p10", result.time_spectrum_p10.to_vec())?;
    output.set_item("time_spectrum_median", result.time_spectrum_median.to_vec())?;
    output.set_item("time_spectrum_p90", result.time_spectrum_p90.to_vec())?;
    output.set_item("successful_repetitions", result.successful_repetitions)?;
    Ok(output.into())
}

fn data_handlers_for_result(result: &pyrth_core::EvaluationResult) -> Vec<String> {
    let mut handlers = vec!["impedance".to_string()];
    if result.time_spectrum.is_some() {
        handlers.push("time_spec".to_string());
    }
    if result.cauer.is_some() {
        handlers.push("structure".to_string());
    }
    handlers
}

fn evaluation_result_keys(result: &pyrth_core::EvaluationResult) -> Vec<String> {
    let mut keys = vec![
        "time".to_string(),
        "impedance".to_string(),
        "log_time".to_string(),
    ];
    if result.derivative.is_some() {
        keys.push("derivative".to_string());
        keys.push("imp_deriv_interp".to_string());
        keys.push("log_time_interp".to_string());
    }
    if result.time_spectrum.is_some() {
        keys.push("time_spec".to_string());
    }
    if result.foster.is_some() {
        keys.push("foster".to_string());
        keys.push("therm_resist_fost".to_string());
        keys.push("therm_capa_fost".to_string());
    }
    if result.cauer.is_some() {
        keys.push("cauer".to_string());
        keys.push("cau_res".to_string());
        keys.push("cau_cap".to_string());
        keys.push("int_cau_res".to_string());
        keys.push("int_cau_cap".to_string());
        keys.push("diff_struc".to_string());
    }
    keys
}

fn foster_network_to_dict(
    py: Python<'_>,
    foster: &pyrth_core::FosterNetwork,
) -> PyResult<PyObject> {
    let foster_dict = PyDict::new(py);
    foster_dict.set_item("resistance", foster.resistance.to_vec())?;
    foster_dict.set_item("capacitance", foster.capacitance.to_vec())?;
    foster_dict.set_item("tau", foster.tau.to_vec())?;
    Ok(foster_dict.into())
}

fn cauer_network_to_dict(py: Python<'_>, cauer: &pyrth_core::CauerNetwork) -> PyResult<PyObject> {
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
    Ok(cauer_dict.into())
}

fn exported_csv_files_to_dict(
    py: Python<'_>,
    files: pyrth_core::ExportedCsvFiles,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("impedance", path_to_string(&files.impedance))?;
    if let Some(path) = files.imp_deriv {
        output.set_item("imp_deriv", path_to_string(&path))?;
    }
    if let Some(path) = files.time_spec {
        output.set_item("time_spec", path_to_string(&path))?;
    }
    if let Some(path) = files.foster {
        output.set_item("foster", path_to_string(&path))?;
    }
    if let Some(path) = files.cauer {
        output.set_item("cauer", path_to_string(&path))?;
    }
    if let Some(path) = files.diff_struc {
        output.set_item("diff_struc", path_to_string(&path))?;
    }
    Ok(output.into())
}

fn exported_figure_files_to_dict(
    py: Python<'_>,
    files: pyrth_core::ExportedFigureFiles,
) -> PyResult<PyObject> {
    let output = PyDict::new(py);
    output.set_item("impedance", path_to_string(&files.impedance))?;
    if let Some(path) = files.imp_deriv {
        output.set_item("imp_deriv", path_to_string(&path))?;
    }
    if let Some(path) = files.time_spec {
        output.set_item("time_spec", path_to_string(&path))?;
    }
    if let Some(path) = files.foster {
        output.set_item("foster", path_to_string(&path))?;
    }
    if let Some(path) = files.cauer {
        output.set_item("cauer", path_to_string(&path))?;
    }
    if let Some(path) = files.diff_struc {
        output.set_item("diff_struc", path_to_string(&path))?;
    }
    Ok(output.into())
}

fn path_to_string(path: &Path) -> String {
    path.to_string_lossy().into_owned()
}

#[pymodule]
fn pyrth_py(m: &Bound<'_, PyModule>) -> PyResult<()> {
    m.add_class::<Evaluation>()?;
    m.add_class::<PyStructureFunction>()?;
    m.add_function(wrap_pyfunction!(evaluate_impedance, m)?)?;
    m.add_function(wrap_pyfunction!(foster_step_response, m)?)?;
    m.add_function(wrap_pyfunction!(theoretical_impedance, m)?)?;
    m.add_function(wrap_pyfunction!(bootstrap_theoretical, m)?)?;
    m.add_function(wrap_pyfunction!(optimize_rc, m)?)?;
    m.add_function(wrap_pyfunction!(predict_temperature_response, m)?)?;
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

fn extract_u64(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<u64>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<u64>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a positive integer: {err}")))
}

fn extract_string(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<String>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<String>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a string: {err}")))
}

fn extract_vec_f64(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<f64>>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<Vec<f64>>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a sequence of numbers: {err}")))
}

fn require_string_list(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<String>> {
    parameters
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{key} is required")))?
        .extract::<Vec<String>>()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a sequence of strings: {err}")))
}

fn require_object_list(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<PyObject>> {
    parameters
        .get_item(key)?
        .ok_or_else(|| PyValueError::new_err(format!("{key} is required")))?
        .extract::<Vec<PyObject>>()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a sequence: {err}")))
}

fn require_first_item<'py>(
    parameters: &Bound<'py, PyDict>,
    keys: &[&str],
    missing_message: &str,
) -> PyResult<Bound<'py, PyAny>> {
    for key in keys {
        if let Some(value) = parameters.get_item(key)? {
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(missing_message.to_string()))
}

fn clone_dict<'py>(
    py: Python<'py>,
    parameters: &Bound<'py, PyDict>,
) -> PyResult<Bound<'py, PyDict>> {
    let output = PyDict::new(py);
    for (key, value) in parameters.iter() {
        output.set_item(key, value)?;
    }
    Ok(output)
}

fn extract_pairs(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Option<Vec<(f64, f64)>>> {
    parameters
        .get_item(key)?
        .map(|value| value.extract::<Vec<(f64, f64)>>())
        .transpose()
        .map_err(|err| PyValueError::new_err(format!("{key} must be a sequence of pairs: {err}")))
}

fn extract_first_vec_f64(
    parameters: &Bound<'_, PyDict>,
    keys: &[&str],
) -> PyResult<Option<Vec<f64>>> {
    for key in keys {
        if let Some(value) = extract_vec_f64(parameters, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn extract_first_pairs(
    parameters: &Bound<'_, PyDict>,
    keys: &[&str],
) -> PyResult<Option<Vec<(f64, f64)>>> {
    for key in keys {
        if let Some(value) = extract_pairs(parameters, key)? {
            return Ok(Some(value));
        }
    }
    Ok(None)
}

fn require_first_vec_f64(parameters: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<Vec<f64>> {
    extract_first_vec_f64(parameters, keys)?
        .ok_or_else(|| PyValueError::new_err(format!("{} is required", keys.join(" or "))))
}

fn require_first_f64(parameters: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<f64> {
    for key in keys {
        if let Some(value) = extract_f64(parameters, key)? {
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(format!(
        "{} is required",
        keys.join(" or ")
    )))
}

fn require_first_usize(parameters: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<usize> {
    for key in keys {
        if let Some(value) = extract_usize(parameters, key)? {
            return Ok(value);
        }
    }
    Err(PyValueError::new_err(format!(
        "{} is required",
        keys.join(" or ")
    )))
}

fn require_pairs(parameters: &Bound<'_, PyDict>, key: &str) -> PyResult<Vec<(f64, f64)>> {
    extract_pairs(parameters, key)?
        .ok_or_else(|| PyValueError::new_err(format!("{key} is required")))
}

fn require_first_pairs(parameters: &Bound<'_, PyDict>, keys: &[&str]) -> PyResult<Vec<(f64, f64)>> {
    extract_first_pairs(parameters, keys)?
        .ok_or_else(|| PyValueError::new_err(format!("{} is required", keys.join(" or "))))
}

fn extract_calibration(parameters: &Bound<'_, PyDict>) -> PyResult<Option<Vec<[f64; 2]>>> {
    for key in ["calibration", "calib"] {
        let Some(value) = parameters.get_item(key)? else {
            continue;
        };
        return value
            .extract::<Vec<(f64, f64)>>()
            .map(|pairs| {
                pairs
                    .into_iter()
                    .map(|(temperature, voltage)| [temperature, voltage])
                    .collect()
            })
            .map(Some)
            .map_err(|err| {
                PyValueError::new_err(format!(
                    "{key} must be a sequence of (temperature, voltage) pairs: {err}"
                ))
            });
    }

    Ok(None)
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
