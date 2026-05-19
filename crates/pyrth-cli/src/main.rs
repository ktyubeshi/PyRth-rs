use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader, Write},
    path::{Path, PathBuf},
};

use pyrth_core::{evaluate, EvaluationParams, TransientInput};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = CliArgs::parse(env::args().skip(1))?;
    let input = read_two_column_data(&args.input)?;

    let mut params = EvaluationParams::default();
    params.only_make_z = args.only_make_z;
    params.calc_struc = !args.no_structure;

    let result = evaluate(input, &params)?;
    fs::create_dir_all(&args.output_dir)?;

    write_pairs(
        &args.output_dir.join("impedance.csv"),
        "time,impedance",
        result.impedance.time.iter().copied(),
        result.impedance.impedance.iter().copied(),
    )?;

    if let (Some(derivative), Some(time_spectrum)) =
        (result.derivative.as_ref(), result.time_spectrum.as_ref())
    {
        write_pairs(
            &args.output_dir.join("imp_deriv.csv"),
            "time,imp_deriv",
            derivative.log_time_pad.iter().map(|value| value.exp()),
            derivative.imp_deriv_interp.iter().copied(),
        )?;
        write_pairs(
            &args.output_dir.join("time_spec.csv"),
            "time,time_spec",
            derivative.log_time_pad.iter().map(|value| value.exp()),
            time_spectrum.iter().copied(),
        )?;
    }

    if let Some(foster) = result.foster.as_ref() {
        write_pairs(
            &args.output_dir.join("foster.csv"),
            "resistance,capacitance",
            foster.resistance.iter().copied(),
            foster.capacitance.iter().copied(),
        )?;
    }

    if let Some(cauer) = result.cauer.as_ref() {
        write_pairs(
            &args.output_dir.join("cauer.csv"),
            "cumulative_resistance,cumulative_capacitance",
            cauer.cumulative_resistance.iter().copied(),
            cauer.cumulative_capacitance.iter().copied(),
        )?;
        write_pairs(
            &args.output_dir.join("diff_struc.csv"),
            "cumulative_resistance,differential_structure",
            cauer.cumulative_resistance.iter().copied(),
            cauer.differential_structure.iter().copied(),
        )?;
    }

    Ok(())
}

struct CliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    only_make_z: bool,
    no_structure: bool,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut input = None;
        let mut output_dir = None;
        let mut only_make_z = false;
        let mut no_structure = false;

        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--input" | "-i" => input = args.next().map(PathBuf::from),
                "--output" | "-o" => output_dir = args.next().map(PathBuf::from),
                "--only-make-z" => only_make_z = true,
                "--no-structure" => no_structure = true,
                _ if input.is_none() => input = Some(PathBuf::from(arg)),
                _ if output_dir.is_none() => output_dir = Some(PathBuf::from(arg)),
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }

        Ok(Self {
            input: input.ok_or("missing input path")?,
            output_dir: output_dir.unwrap_or_else(|| PathBuf::from("output/rust-cli")),
            only_make_z,
            no_structure,
        })
    }
}

fn print_usage() {
    println!("Usage: pyrth-cli --input <path> --output <dir> [--only-make-z] [--no-structure]");
}

fn read_two_column_data(path: &Path) -> Result<TransientInput, Box<dyn Error>> {
    let file = File::open(path)?;
    let reader = BufReader::new(file);
    let mut pairs = Vec::new();

    for (line_number, line) in reader.lines().enumerate() {
        let line = line?;
        let trimmed = line.trim();
        if trimmed.is_empty() || trimmed.starts_with('#') {
            continue;
        }

        let parts = trimmed
            .split(|ch: char| ch == ',' || ch == ';' || ch.is_ascii_whitespace())
            .filter(|part| !part.is_empty())
            .collect::<Vec<_>>();

        if parts.len() < 2 {
            continue;
        }

        match (parts[0].parse::<f64>(), parts[1].parse::<f64>()) {
            (Ok(time), Ok(value)) => pairs.push((time, value)),
            _ if line_number == 0 => continue,
            _ => {
                return Err(format!(
                    "failed to parse numeric data at {}:{}",
                    path.display(),
                    line_number + 1
                )
                .into())
            }
        }
    }

    TransientInput::from_pairs(pairs).map_err(Into::into)
}

fn write_pairs(
    path: &Path,
    header: &str,
    left: impl Iterator<Item = f64>,
    right: impl Iterator<Item = f64>,
) -> Result<(), Box<dyn Error>> {
    let mut file = File::create(path)?;
    writeln!(file, "{header}")?;
    for (left, right) in left.zip(right) {
        writeln!(file, "{left:.17e},{right:.17e}")?;
    }
    Ok(())
}
