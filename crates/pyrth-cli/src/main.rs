use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use pyrth_core::{evaluate, export_csv, EvaluationParams, TransientInput};

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
    if let Some(log_time_size) = args.log_time_size {
        params.log_time_size = log_time_size;
    }
    if let Some(bay_steps) = args.bay_steps {
        params.bay_steps = bay_steps;
    }
    if let Some(blockwise_sum_width) = args.blockwise_sum_width {
        params.blockwise_sum_width = blockwise_sum_width;
    }
    if let Some(min_index) = args.min_index {
        params.min_index = min_index;
    }
    if let Some(minimum_window_size) = args.minimum_window_size {
        params.minimum_window_size = minimum_window_size;
    }

    let result = evaluate(input, &params)?;
    export_csv(&result, &args.output_dir)?;

    Ok(())
}

struct CliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    only_make_z: bool,
    no_structure: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    blockwise_sum_width: Option<usize>,
    min_index: Option<usize>,
    minimum_window_size: Option<usize>,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut input = None;
        let mut output_dir = None;
        let mut only_make_z = false;
        let mut no_structure = false;
        let mut log_time_size = None;
        let mut bay_steps = None;
        let mut blockwise_sum_width = None;
        let mut min_index = None;
        let mut minimum_window_size = None;

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
                "--log-time-size" => {
                    log_time_size = Some(parse_next_usize(&mut args, "--log-time-size")?)
                }
                "--bay-steps" => bay_steps = Some(parse_next_usize(&mut args, "--bay-steps")?),
                "--blockwise-sum-width" => {
                    blockwise_sum_width =
                        Some(parse_next_usize(&mut args, "--blockwise-sum-width")?)
                }
                "--min-index" => min_index = Some(parse_next_usize(&mut args, "--min-index")?),
                "--minimum-window-size" => {
                    minimum_window_size =
                        Some(parse_next_usize(&mut args, "--minimum-window-size")?)
                }
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
            log_time_size,
            bay_steps,
            blockwise_sum_width,
            min_index,
            minimum_window_size,
        })
    }
}

fn print_usage() {
    println!(
        "Usage: pyrth-cli --input <path> --output <dir> [--only-make-z] [--no-structure] [--log-time-size <n>] [--bay-steps <n>] [--blockwise-sum-width <n>] [--min-index <n>] [--minimum-window-size <n>]"
    );
}

fn parse_next_usize(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<usize, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse::<usize>()
        .map_err(|err| format!("invalid value for {flag}: {value} ({err})").into())
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
