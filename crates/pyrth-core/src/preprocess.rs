use ndarray::Array1;

use crate::{
    data::{ImpedanceData, TransientInput},
    error::Result,
};

pub fn make_impedance_data(input: TransientInput) -> Result<ImpedanceData> {
    input.validate()?;
    let log_time: Array1<f64> = input.time.mapv(f64::ln);
    Ok(ImpedanceData {
        time: input.time,
        impedance: input.value,
        log_time,
    })
}
