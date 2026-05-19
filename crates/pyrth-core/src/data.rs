use ndarray::Array1;

use crate::error::{PyrthError, Result};

#[derive(Clone, Debug, PartialEq)]
pub struct TransientInput {
    pub time: Array1<f64>,
    pub value: Array1<f64>,
}

impl TransientInput {
    pub fn new(time: Array1<f64>, value: Array1<f64>) -> Result<Self> {
        let input = Self { time, value };
        input.validate()?;
        Ok(input)
    }

    pub fn from_pairs<I>(pairs: I) -> Result<Self>
    where
        I: IntoIterator<Item = (f64, f64)>,
    {
        let (time, value): (Vec<_>, Vec<_>) = pairs.into_iter().unzip();
        Self::new(Array1::from(time), Array1::from(value))
    }

    pub fn validate(&self) -> Result<()> {
        if self.time.is_empty() {
            return Err(PyrthError::EmptyInput);
        }
        if self.time.len() != self.value.len() {
            return Err(PyrthError::LengthMismatch {
                time_len: self.time.len(),
                value_len: self.value.len(),
            });
        }
        if !self.value.iter().all(|value| value.is_finite()) {
            return Err(PyrthError::InvalidValues);
        }
        if !self.time.iter().all(|time| time.is_finite() && *time > 0.0) {
            return Err(PyrthError::InvalidTimeAxis);
        }
        if self
            .time
            .windows(2)
            .into_iter()
            .any(|pair| pair[0] >= pair[1])
        {
            return Err(PyrthError::InvalidTimeAxis);
        }
        Ok(())
    }
}

#[derive(Clone, Debug, PartialEq)]
pub struct ImpedanceData {
    pub time: Array1<f64>,
    pub impedance: Array1<f64>,
    pub log_time: Array1<f64>,
}
