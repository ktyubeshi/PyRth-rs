use ndarray::Array1;

use crate::{
    error::{PyrthError, Result},
    evaluation::FosterNetwork,
};

pub fn foster_from_time_spectrum(
    log_time: &Array1<f64>,
    time_spectrum: &Array1<f64>,
    threshold: f64,
) -> Result<FosterNetwork> {
    let mut resistance = Vec::new();
    let mut capacitance = Vec::new();
    let mut tau = Vec::new();

    for (log_tau, spectrum_value) in log_time.iter().zip(time_spectrum) {
        if *spectrum_value >= threshold {
            let tau_value = log_tau.exp();
            resistance.push(*spectrum_value);
            capacitance.push(tau_value / spectrum_value);
            tau.push(tau_value);
        }
    }

    if resistance.is_empty() {
        return Err(PyrthError::EmptySpectrum);
    }

    Ok(FosterNetwork {
        resistance: Array1::from(resistance),
        capacitance: Array1::from(capacitance),
        tau: Array1::from(tau),
    })
}
