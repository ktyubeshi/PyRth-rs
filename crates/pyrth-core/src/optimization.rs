use ndarray::{Array1, Zip};

use crate::{
    data::TransientInput,
    error::{PyrthError, Result},
    evaluation::{EvaluationResult, FosterNetwork},
    theoretical::TheoreticalModel,
};

#[derive(Clone, Debug, PartialEq)]
pub struct RcParameters {
    pub resistance: Array1<f64>,
    pub capacitance: Array1<f64>,
}

impl RcParameters {
    pub fn new(resistance: Array1<f64>, capacitance: Array1<f64>) -> Result<Self> {
        let parameters = Self {
            resistance,
            capacitance,
        };
        parameters.validate()?;
        Ok(parameters)
    }

    pub fn from_slices(resistance: &[f64], capacitance: &[f64]) -> Result<Self> {
        Self::new(
            Array1::from(resistance.to_vec()),
            Array1::from(capacitance.to_vec()),
        )
    }

    pub fn from_flattened(values: &Array1<f64>) -> Result<Self> {
        if values.is_empty() || values.len() % 2 != 0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "values",
                expected: "non-empty even-length [resistance..., capacitance...] vector",
                actual: values.len().to_string(),
            });
        }

        let split = values.len() / 2;
        Self::new(
            values.slice(ndarray::s![..split]).to_owned(),
            values.slice(ndarray::s![split..]).to_owned(),
        )
    }

    pub fn from_foster_network(foster: &FosterNetwork) -> Result<Self> {
        Self::new(foster.resistance.clone(), foster.capacitance.clone())
    }

    pub fn from_evaluation(result: &EvaluationResult) -> Result<Self> {
        let foster = result
            .foster
            .as_ref()
            .ok_or_else(|| PyrthError::InvalidParameter {
                parameter: "result.foster",
                expected: "present",
                actual: "missing".to_string(),
            })?;
        Self::from_foster_network(foster)
    }

    pub fn len(&self) -> usize {
        self.resistance.len()
    }

    pub fn is_empty(&self) -> bool {
        self.resistance.is_empty()
    }

    pub fn validate(&self) -> Result<()> {
        validate_rc_arrays(&self.resistance, &self.capacitance)
    }

    pub fn to_flattened(&self) -> Result<Array1<f64>> {
        self.validate()?;

        let mut values = Vec::with_capacity(self.len() * 2);
        values.extend(self.resistance.iter().copied());
        values.extend(self.capacitance.iter().copied());
        Ok(Array1::from(values))
    }

    pub fn to_theoretical_model(&self) -> Result<TheoreticalModel> {
        self.validate()?;
        TheoreticalModel::new(self.resistance.clone(), self.capacitance.clone())
    }

    pub fn impedance_at(&self, time: f64) -> Result<f64> {
        if !time.is_finite() || time <= 0.0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "time",
                expected: "finite and positive",
                actual: time.to_string(),
            });
        }

        Ok(self.to_theoretical_model()?.impedance_at(time))
    }

    pub fn impedance_on(&self, time: &Array1<f64>) -> Result<Array1<f64>> {
        if time.is_empty() {
            return Err(PyrthError::EmptyInput);
        }
        if !time.iter().all(|value| value.is_finite() && *value > 0.0) {
            return Err(PyrthError::InvalidParameter {
                parameter: "time",
                expected: "finite and positive",
                actual: format!("{time:?}"),
            });
        }

        let model = self.to_theoretical_model()?;
        Ok(time.mapv(|sample_time| model.impedance_at(sample_time)))
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct RcParameterBounds {
    pub lower: RcParameters,
    pub upper: RcParameters,
}

impl RcParameterBounds {
    pub fn new(lower: RcParameters, upper: RcParameters) -> Result<Self> {
        lower.validate()?;
        upper.validate()?;
        validate_same_len(lower.len(), upper.len(), "bounds")?;
        if !all_le(&lower.resistance, &upper.resistance)
            || !all_le(&lower.capacitance, &upper.capacitance)
        {
            return Err(PyrthError::InvalidParameter {
                parameter: "bounds",
                expected: "lower values less than or equal to upper values",
                actual: format!("lower={lower:?}, upper={upper:?}"),
            });
        }

        Ok(Self { lower, upper })
    }

    pub fn contains(&self, parameters: &RcParameters) -> Result<bool> {
        self.lower.validate()?;
        self.upper.validate()?;
        validate_same_len(self.lower.len(), self.upper.len(), "bounds")?;
        parameters.validate()?;
        validate_same_len(self.lower.len(), parameters.len(), "parameters")?;

        Ok(all_le(&self.lower.resistance, &parameters.resistance)
            && all_le(&parameters.resistance, &self.upper.resistance)
            && all_le(&self.lower.capacitance, &parameters.capacitance)
            && all_le(&parameters.capacitance, &self.upper.capacitance))
    }

    pub fn clamp(&self, parameters: &RcParameters) -> Result<RcParameters> {
        self.lower.validate()?;
        self.upper.validate()?;
        validate_same_len(self.lower.len(), self.upper.len(), "bounds")?;
        validate_same_len(
            parameters.resistance.len(),
            parameters.capacitance.len(),
            "capacitance",
        )?;
        validate_same_len(self.lower.len(), parameters.len(), "parameters")?;
        validate_finite(&parameters.resistance, "resistance")?;
        validate_finite(&parameters.capacitance, "capacitance")?;

        Ok(RcParameters {
            resistance: clamp_array(
                &parameters.resistance,
                &self.lower.resistance,
                &self.upper.resistance,
            ),
            capacitance: clamp_array(
                &parameters.capacitance,
                &self.lower.capacitance,
                &self.upper.capacitance,
            ),
        })
    }
}

#[derive(Clone, Copy, Debug, PartialEq)]
pub struct OptimizationConfig {
    pub max_iter: usize,
    pub initial_step: f64,
    pub min_step: f64,
    pub shrink_factor: f64,
}

impl Default for OptimizationConfig {
    fn default() -> Self {
        Self {
            max_iter: 128,
            initial_step: 0.1,
            min_step: 1e-6,
            shrink_factor: 0.5,
        }
    }
}

impl OptimizationConfig {
    pub fn validate(&self) -> Result<()> {
        if self.max_iter == 0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "max_iter",
                expected: "greater than 0",
                actual: self.max_iter.to_string(),
            });
        }
        if !self.initial_step.is_finite() || self.initial_step <= 0.0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "initial_step",
                expected: "finite and positive",
                actual: self.initial_step.to_string(),
            });
        }
        if !self.min_step.is_finite() || self.min_step <= 0.0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "min_step",
                expected: "finite and positive",
                actual: self.min_step.to_string(),
            });
        }
        if self.min_step > self.initial_step {
            return Err(PyrthError::InvalidParameter {
                parameter: "min_step",
                expected: "less than or equal to initial_step",
                actual: self.min_step.to_string(),
            });
        }
        if !self.shrink_factor.is_finite() || self.shrink_factor <= 0.0 || self.shrink_factor >= 1.0
        {
            return Err(PyrthError::InvalidParameter {
                parameter: "shrink_factor",
                expected: "finite and in the open interval (0, 1)",
                actual: self.shrink_factor.to_string(),
            });
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct OptimizationResult {
    pub parameters: RcParameters,
    pub residual_norm: f64,
    pub iterations: usize,
}

pub fn optimize_rc_parameters(
    input: &TransientInput,
    initial: &RcParameters,
    bounds: &RcParameterBounds,
    config: OptimizationConfig,
) -> Result<OptimizationResult> {
    input.validate()?;
    initial.validate()?;
    bounds.lower.validate()?;
    bounds.upper.validate()?;
    validate_same_len(initial.len(), bounds.lower.len(), "bounds")?;
    validate_same_len(bounds.lower.len(), bounds.upper.len(), "bounds")?;
    config.validate()?;

    let mut current = bounds.clamp(initial)?;
    let mut current_residual = residual_for_parameters(input, &current)?;
    let mut step = config.initial_step;
    let mut iterations = 0;

    while iterations < config.max_iter && step >= config.min_step {
        iterations += 1;
        let mut improved = false;

        for index in 0..current.len() {
            for direction in [1.0, -1.0] {
                let mut candidate = current.clone();
                candidate.resistance[index] += direction * step;
                candidate = bounds.clamp(&candidate)?;
                if candidate == current {
                    continue;
                }

                let candidate_residual = residual_for_parameters(input, &candidate)?;
                if candidate_residual < current_residual {
                    current = candidate;
                    current_residual = candidate_residual;
                    improved = true;
                }
            }

            for direction in [1.0, -1.0] {
                let mut candidate = current.clone();
                candidate.capacitance[index] += direction * step;
                candidate = bounds.clamp(&candidate)?;
                if candidate == current {
                    continue;
                }

                let candidate_residual = residual_for_parameters(input, &candidate)?;
                if candidate_residual < current_residual {
                    current = candidate;
                    current_residual = candidate_residual;
                    improved = true;
                }
            }
        }

        if !improved {
            step *= config.shrink_factor;
        }
    }

    Ok(OptimizationResult {
        parameters: current,
        residual_norm: current_residual,
        iterations,
    })
}

pub fn relative_l2_norm(reference: &Array1<f64>, candidate: &Array1<f64>) -> Result<f64> {
    validate_same_len(reference.len(), candidate.len(), "candidate")?;
    if reference.is_empty() {
        return Err(PyrthError::EmptyInput);
    }
    if !reference.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidParameter {
            parameter: "reference",
            expected: "finite values",
            actual: format!("{reference:?}"),
        });
    }
    if !candidate.iter().all(|value| value.is_finite()) {
        return Err(PyrthError::InvalidParameter {
            parameter: "candidate",
            expected: "finite values",
            actual: format!("{candidate:?}"),
        });
    }

    let squared_diff = reference
        .iter()
        .zip(candidate.iter())
        .map(|(reference_value, candidate_value)| (candidate_value - reference_value).powi(2))
        .sum::<f64>();
    let squared_reference = reference
        .iter()
        .map(|reference_value| reference_value.powi(2))
        .sum::<f64>();

    if squared_reference == 0.0 {
        if squared_diff == 0.0 {
            return Ok(0.0);
        }
        return Err(PyrthError::InvalidParameter {
            parameter: "reference",
            expected: "non-zero L2 norm",
            actual: "0".to_string(),
        });
    }

    Ok(squared_diff.sqrt() / squared_reference.sqrt())
}

pub fn impedance_residual_norm(input: &TransientInput, model: &TheoreticalModel) -> Result<f64> {
    input.validate()?;
    let parameters = RcParameters::new(model.resistances.clone(), model.capacitances.clone())?;

    let model_value = parameters.impedance_on(&input.time)?;
    relative_l2_norm(&input.value, &model_value)
}

fn validate_rc_arrays(resistance: &Array1<f64>, capacitance: &Array1<f64>) -> Result<()> {
    if resistance.is_empty() {
        return Err(PyrthError::EmptyInput);
    }
    validate_same_len(resistance.len(), capacitance.len(), "capacitance")?;
    validate_positive_finite(resistance, "resistance")?;
    validate_positive_finite(capacitance, "capacitance")?;
    Ok(())
}

fn validate_positive_finite(values: &Array1<f64>, parameter: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite() && *value > 0.0) {
        return Ok(());
    }

    Err(PyrthError::InvalidParameter {
        parameter,
        expected: "finite and positive",
        actual: format!("{values:?}"),
    })
}

fn validate_finite(values: &Array1<f64>, parameter: &'static str) -> Result<()> {
    if values.iter().all(|value| value.is_finite()) {
        return Ok(());
    }

    Err(PyrthError::InvalidParameter {
        parameter,
        expected: "finite",
        actual: format!("{values:?}"),
    })
}

fn validate_same_len(left_len: usize, right_len: usize, parameter: &'static str) -> Result<()> {
    if left_len == right_len {
        return Ok(());
    }

    Err(PyrthError::InvalidParameter {
        parameter,
        expected: "same length as paired array",
        actual: format!("{left_len} != {right_len}"),
    })
}

fn all_le(left: &Array1<f64>, right: &Array1<f64>) -> bool {
    Zip::from(left)
        .and(right)
        .all(|left_value, right_value| left_value <= right_value)
}

fn clamp_array(values: &Array1<f64>, lower: &Array1<f64>, upper: &Array1<f64>) -> Array1<f64> {
    Zip::from(values)
        .and(lower)
        .and(upper)
        .map_collect(|value, lower_value, upper_value| value.clamp(*lower_value, *upper_value))
}

fn residual_for_parameters(input: &TransientInput, parameters: &RcParameters) -> Result<f64> {
    let model = parameters.to_theoretical_model()?;
    impedance_residual_norm(input, &model)
}
