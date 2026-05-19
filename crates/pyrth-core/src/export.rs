use std::{
    fs,
    path::{Path, PathBuf},
};

use csv::Writer;

use crate::{error::Result, EvaluationResult};

#[derive(Clone, Debug, Default, Eq, PartialEq)]
pub struct ExportedCsvFiles {
    pub impedance: PathBuf,
    pub imp_deriv: Option<PathBuf>,
    pub time_spec: Option<PathBuf>,
    pub foster: Option<PathBuf>,
    pub cauer: Option<PathBuf>,
    pub diff_struc: Option<PathBuf>,
}

pub fn export_csv(
    result: &EvaluationResult,
    output_dir: impl AsRef<Path>,
) -> Result<ExportedCsvFiles> {
    let output_dir = output_dir.as_ref();
    fs::create_dir_all(output_dir)?;

    let impedance_path = output_dir.join("impedance.csv");
    write_pairs(
        &impedance_path,
        ["time", "impedance"],
        result.impedance.time.iter().copied(),
        result.impedance.impedance.iter().copied(),
    )?;

    let mut files = ExportedCsvFiles {
        impedance: impedance_path,
        ..ExportedCsvFiles::default()
    };

    if let Some(derivative) = result.derivative.as_ref() {
        let imp_deriv_path = output_dir.join("imp_deriv.csv");
        write_pairs(
            &imp_deriv_path,
            ["time", "imp_deriv"],
            derivative.log_time_pad.iter().map(|value| value.exp()),
            derivative.imp_deriv_interp.iter().copied(),
        )?;
        files.imp_deriv = Some(imp_deriv_path);

        if let Some(time_spectrum) = result.time_spectrum.as_ref() {
            let time_spec_path = output_dir.join("time_spec.csv");
            write_pairs(
                &time_spec_path,
                ["time", "time_spec"],
                derivative.log_time_pad.iter().map(|value| value.exp()),
                time_spectrum.iter().copied(),
            )?;
            files.time_spec = Some(time_spec_path);
        }
    }

    if let Some(foster) = result.foster.as_ref() {
        let foster_path = output_dir.join("foster.csv");
        write_pairs(
            &foster_path,
            ["resistance", "capacitance"],
            foster.resistance.iter().copied(),
            foster.capacitance.iter().copied(),
        )?;
        files.foster = Some(foster_path);
    }

    if let Some(cauer) = result.cauer.as_ref() {
        let cauer_path = output_dir.join("cauer.csv");
        write_pairs(
            &cauer_path,
            ["cumulative_resistance", "cumulative_capacitance"],
            cauer.cumulative_resistance.iter().copied(),
            cauer.cumulative_capacitance.iter().copied(),
        )?;
        files.cauer = Some(cauer_path);

        let diff_struc_path = output_dir.join("diff_struc.csv");
        write_pairs(
            &diff_struc_path,
            ["cumulative_resistance", "differential_structure"],
            cauer.cumulative_resistance.iter().copied(),
            cauer.differential_structure.iter().copied(),
        )?;
        files.diff_struc = Some(diff_struc_path);
    }

    Ok(files)
}

fn write_pairs(
    path: &Path,
    header: [&str; 2],
    left: impl Iterator<Item = f64>,
    right: impl Iterator<Item = f64>,
) -> Result<()> {
    let mut writer = Writer::from_path(path)?;
    writer.write_record(header)?;
    for (left, right) in left.zip(right) {
        writer.write_record([format!("{left:.17e}"), format!("{right:.17e}")])?;
    }
    writer.flush()?;
    Ok(())
}
