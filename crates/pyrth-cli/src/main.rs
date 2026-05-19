use std::{
    env,
    error::Error,
    fs::File,
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use pyrth_core::{evaluate, export_csv, EvaluationParams, InputMode, TransientInput};

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
    params.input_mode = args.input_mode;
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
    if let Some(power_step) = args.power_step {
        params.power_step = power_step;
    }
    if let Some(power_scale_factor) = args.power_scale_factor {
        params.power_scale_factor = power_scale_factor;
    }
    if let Some(optical_power) = args.optical_power {
        params.optical_power = optical_power;
    }
    params.is_heating = args.is_heating;
    if let Some(kfac_fit_deg) = args.kfac_fit_deg {
        params.kfac_fit_deg = kfac_fit_deg;
    }
    if let Some(calibration_path) = args.calibration.as_ref() {
        params.calibration = Some(read_calibration_data(calibration_path)?);
    }
    if let Some(data_cut_lower) = args.data_cut_lower {
        params.data_cut_lower = data_cut_lower;
    }
    if let Some(data_cut_upper) = args.data_cut_upper {
        params.data_cut_upper = Some(data_cut_upper);
    }
    if let Some(temp_0_avg_range) = args.temp_0_avg_range {
        params.temp_0_avg_range = temp_0_avg_range;
    }

    let result = evaluate(input, &params)?;
    export_csv(&result, &args.output_dir)?;

    Ok(())
}

struct CliArgs {
    input: PathBuf,
    output_dir: PathBuf,
    input_mode: InputMode,
    only_make_z: bool,
    no_structure: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    blockwise_sum_width: Option<usize>,
    min_index: Option<usize>,
    minimum_window_size: Option<usize>,
    power_step: Option<f64>,
    power_scale_factor: Option<f64>,
    optical_power: Option<f64>,
    is_heating: bool,
    kfac_fit_deg: Option<usize>,
    calibration: Option<PathBuf>,
    data_cut_lower: Option<usize>,
    data_cut_upper: Option<usize>,
    temp_0_avg_range: Option<(usize, usize)>,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut input = None;
        let mut output_dir = None;
        let mut input_mode = InputMode::Impedance;
        let mut only_make_z = false;
        let mut no_structure = false;
        let mut log_time_size = None;
        let mut bay_steps = None;
        let mut blockwise_sum_width = None;
        let mut min_index = None;
        let mut minimum_window_size = None;
        let mut power_step = None;
        let mut power_scale_factor = None;
        let mut optical_power = None;
        let mut is_heating = false;
        let mut kfac_fit_deg = None;
        let mut calibration = None;
        let mut data_cut_lower = None;
        let mut data_cut_upper = None;
        let mut temp_0_avg_range = None;

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
                "--input-mode" => {
                    let value = args.next().ok_or("missing value for --input-mode")?;
                    input_mode = InputMode::from_label(&value)?;
                }
                "--power-step" => power_step = Some(parse_next_f64(&mut args, "--power-step")?),
                "--power-scale-factor" => {
                    power_scale_factor = Some(parse_next_f64(&mut args, "--power-scale-factor")?)
                }
                "--optical-power" => {
                    optical_power = Some(parse_next_f64(&mut args, "--optical-power")?)
                }
                "--is-heating" => is_heating = true,
                "--kfac-fit-deg" => {
                    kfac_fit_deg = Some(parse_next_usize(&mut args, "--kfac-fit-deg")?)
                }
                "--calibration" => calibration = args.next().map(PathBuf::from),
                "--data-cut-lower" => {
                    data_cut_lower = Some(parse_next_usize(&mut args, "--data-cut-lower")?)
                }
                "--data-cut-upper" => {
                    data_cut_upper = Some(parse_next_usize(&mut args, "--data-cut-upper")?)
                }
                "--temp-zero-range" => {
                    temp_0_avg_range = Some(parse_next_range(&mut args, "--temp-zero-range")?)
                }
                _ if input.is_none() => input = Some(PathBuf::from(arg)),
                _ if output_dir.is_none() => output_dir = Some(PathBuf::from(arg)),
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }

        Ok(Self {
            input: input.ok_or("missing input path")?,
            output_dir: output_dir.unwrap_or_else(|| PathBuf::from("output/rust-cli")),
            input_mode,
            only_make_z,
            no_structure,
            log_time_size,
            bay_steps,
            blockwise_sum_width,
            min_index,
            minimum_window_size,
            power_step,
            power_scale_factor,
            optical_power,
            is_heating,
            kfac_fit_deg,
            calibration,
            data_cut_lower,
            data_cut_upper,
            temp_0_avg_range,
        })
    }
}

fn print_usage() {
    println!(
        "Usage: pyrth-cli --input <path> --output <dir> [--input-mode impedance|temp|volt] [--power-step <w>] [--power-scale-factor <x>] [--optical-power <w>] [--is-heating] [--calibration <path>] [--kfac-fit-deg <n>] [--data-cut-lower <n>] [--data-cut-upper <n>] [--temp-zero-range <start:end>] [--only-make-z] [--no-structure] [--log-time-size <n>] [--bay-steps <n>] [--blockwise-sum-width <n>] [--min-index <n>] [--minimum-window-size <n>]"
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

fn parse_next_f64(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<f64, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    value
        .parse::<f64>()
        .map_err(|err| format!("invalid value for {flag}: {value} ({err})").into())
}

fn parse_next_range(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<(usize, usize), Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    let (start, end) = value
        .split_once(':')
        .ok_or_else(|| format!("invalid value for {flag}: expected start:end"))?;
    Ok((start.parse::<usize>()?, end.parse::<usize>()?))
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

fn read_calibration_data(path: &Path) -> Result<Vec<[f64; 2]>, Box<dyn Error>> {
    let input = read_two_column_data(path)?;
    Ok(input
        .time
        .iter()
        .zip(input.value.iter())
        .map(|(temperature, voltage)| [*temperature, *voltage])
        .collect())
}
