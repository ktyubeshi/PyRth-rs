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
