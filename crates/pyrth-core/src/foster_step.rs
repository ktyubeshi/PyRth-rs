use ndarray::Array1;

use crate::{
    data::TransientInput,
    error::{PyrthError, Result},
};

#[derive(Clone, Debug, PartialEq)]
pub struct FosterStepResponseModel {
    pub resistances: Array1<f64>,
    pub capacitances: Array1<f64>,
}

impl FosterStepResponseModel {
    pub fn new(resistances: Array1<f64>, capacitances: Array1<f64>) -> Result<Self> {
        let model = Self {
            resistances,
            capacitances,
        };
        model.validate()?;
        Ok(model)
    }

    pub fn from_slices(resistances: &[f64], capacitances: &[f64]) -> Result<Self> {
        Self::new(
            Array1::from(resistances.to_vec()),
            Array1::from(capacitances.to_vec()),
        )
    }

    pub fn to_transient_input(
        &self,
        time_start: f64,
        time_end: f64,
        time_size: usize,
    ) -> Result<TransientInput> {
        validate_time_range(time_start, time_end, time_size)?;

        let log_start = time_start.ln();
        let log_step = (time_end.ln() - log_start) / (time_size - 1) as f64;
        let time = Array1::from_iter((0..time_size).map(|index| {
            if index == time_size - 1 {
                time_end
            } else {
                (log_start + log_step * index as f64).exp()
            }
        }));
        let value = self.step_response_on(&time)?;

        TransientInput::new(time, value)
    }

    pub fn step_response_at(&self, time: f64) -> Result<f64> {
        if !time.is_finite() || time <= 0.0 {
            return Err(PyrthError::InvalidParameter {
                parameter: "time",
                expected: "finite and positive",
                actual: time.to_string(),
            });
        }

        Ok(self
            .resistances
            .iter()
            .zip(self.capacitances.iter())
            .map(|(resistance, capacitance)| {
                resistance * (1.0 - (-time / (resistance * capacitance)).exp())
            })
            .sum())
    }

    pub fn step_response_on(&self, time: &Array1<f64>) -> Result<Array1<f64>> {
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

        time.iter()
            .map(|sample_time| self.step_response_at(*sample_time))
            .collect()
    }

    #[deprecated(note = "Use step_response_at for the Foster lumped RC step response.")]
    pub fn impedance_at(&self, time: f64) -> f64 {
        self.step_response_at(time).unwrap_or(f64::NAN)
    }

    fn validate(&self) -> Result<()> {
        if self.resistances.is_empty() {
            return Err(PyrthError::EmptyInput);
        }
        if self.resistances.len() != self.capacitances.len() {
            return Err(PyrthError::LengthMismatch {
                time_len: self.resistances.len(),
                value_len: self.capacitances.len(),
            });
        }
        if !self
            .resistances
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(PyrthError::InvalidParameter {
                parameter: "resistances",
                expected: "finite and positive",
                actual: format!("{:?}", self.resistances),
            });
        }
        if !self
            .capacitances
            .iter()
            .all(|value| value.is_finite() && *value > 0.0)
        {
            return Err(PyrthError::InvalidParameter {
                parameter: "capacitances",
                expected: "finite and positive",
                actual: format!("{:?}", self.capacitances),
            });
        }
        Ok(())
    }
}

pub fn foster_step_response_input(
    resistances: &[f64],
    capacitances: &[f64],
    time_start: f64,
    time_end: f64,
    time_size: usize,
) -> Result<TransientInput> {
    FosterStepResponseModel::from_slices(resistances, capacitances)?
        .to_transient_input(time_start, time_end, time_size)
}

fn validate_time_range(time_start: f64, time_end: f64, time_size: usize) -> Result<()> {
    if !(time_start.is_finite()
        && time_end.is_finite()
        && time_start > 0.0
        && time_end > time_start)
    {
        return Err(PyrthError::InvalidParameter {
            parameter: "time_range",
            expected: "finite, positive, and increasing",
            actual: format!("[{time_start}, {time_end}]"),
        });
    }
    if time_size < 2 {
        return Err(PyrthError::InvalidParameter {
            parameter: "time_size",
            expected: "at least 2",
            actual: time_size.to_string(),
        });
    }
    Ok(())
}
