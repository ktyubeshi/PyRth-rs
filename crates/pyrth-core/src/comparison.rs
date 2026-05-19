use ndarray::Array1;

use crate::{
    error::{PyrthError, Result},
    evaluation::EvaluationResult,
};

#[derive(Clone, Debug, PartialEq)]
pub struct ComparisonResult {
    pub time_const_norm: f64,
    pub structure_norm: f64,
    pub total_resistance_diff: f64,
}

pub fn compare_evaluations(
    reference: &EvaluationResult,
    candidate: &EvaluationResult,
) -> Result<ComparisonResult> {
    let reference_spectrum =
        reference
            .time_spectrum
            .as_ref()
            .ok_or_else(|| PyrthError::InvalidParameter {
                parameter: "reference.time_spectrum",
                expected: "present",
                actual: "missing".to_string(),
            })?;
    let candidate_spectrum =
        candidate
            .time_spectrum
            .as_ref()
            .ok_or_else(|| PyrthError::InvalidParameter {
                parameter: "candidate.time_spectrum",
                expected: "present",
                actual: "missing".to_string(),
            })?;

    Ok(ComparisonResult {
        time_const_norm: relative_l2_norm(reference_spectrum, candidate_spectrum, "time_spectrum")?,
        structure_norm: structure_norm(reference, candidate)?,
        total_resistance_diff: (total_resistance(reference)? - total_resistance(candidate)?).abs(),
    })
}

fn structure_norm(reference: &EvaluationResult, candidate: &EvaluationResult) -> Result<f64> {
    let Some(reference_cauer) = reference.cauer.as_ref() else {
        return Ok(0.0);
    };
    let Some(candidate_cauer) = candidate.cauer.as_ref() else {
        return Ok(0.0);
    };
    if reference_cauer.differential_structure.is_empty()
        || candidate_cauer.differential_structure.is_empty()
    {
        return Ok(0.0);
    }

    relative_l2_norm(
        &reference_cauer.differential_structure,
        &candidate_cauer.differential_structure,
        "cauer.differential_structure",
    )
}

fn total_resistance(result: &EvaluationResult) -> Result<f64> {
    if let Some(cauer) = result.cauer.as_ref() {
        if let Some(value) = cauer.cumulative_resistance.last() {
            return Ok(*value);
        }
    }
    if let Some(foster) = result.foster.as_ref() {
        if !foster.resistance.is_empty() {
            return Ok(foster.resistance.iter().sum());
        }
    }

    Err(PyrthError::InvalidParameter {
        parameter: "total_resistance",
        expected: "non-empty Cauer cumulative resistance or Foster resistance",
        actual: "missing".to_string(),
    })
}

fn relative_l2_norm(
    reference: &Array1<f64>,
    candidate: &Array1<f64>,
    parameter: &'static str,
) -> Result<f64> {
    let len = reference.len().min(candidate.len());
    if len == 0 {
        return Err(PyrthError::InvalidParameter {
            parameter,
            expected: "at least one overlapping sample",
            actual: "0".to_string(),
        });
    }

    let mut squared_diff = 0.0;
    let mut squared_reference = 0.0;
    for (reference_value, candidate_value) in reference.iter().zip(candidate.iter()).take(len) {
        squared_diff += (candidate_value - reference_value).powi(2);
        squared_reference += reference_value.powi(2);
    }

    if squared_reference == 0.0 {
        if squared_diff == 0.0 {
            return Ok(0.0);
        }
        return Err(PyrthError::InvalidParameter {
            parameter,
            expected: "non-zero reference norm",
            actual: "0".to_string(),
        });
    }

    Ok(squared_diff.sqrt() / squared_reference.sqrt())
}
