use std::{
    env,
    error::Error,
    fs::{self, File},
    io::{BufRead, BufReader},
    path::{Path, PathBuf},
};

use pyrth_core::{
    evaluate, export_csv, export_svg_figures, parse_t3ster_calibration_text,
    parse_t3ster_power_step, parse_t3ster_raw_text, t3ster_raw_to_temperature_input,
    theoretical_impedance_input, DeconvMode, EvaluationParams, FourierFilter, InputMode,
    StructureMethod, TransientInput,
};

fn main() {
    if let Err(err) = run() {
        eprintln!("error: {err}");
        std::process::exit(1);
    }
}

fn run() -> Result<(), Box<dyn Error>> {
    let args = CliArgs::parse(env::args().skip(1))?;

    let mut params = EvaluationParams::default();
    if let Some(kfac_fit_deg) = args.kfac_fit_deg {
        params.kfac_fit_deg = kfac_fit_deg;
    }
    let input = if args.has_theoretical_model() {
        params.input_mode = InputMode::Impedance;
        build_theoretical_input(&args)?
    } else if args.input_mode == InputMode::T3ster {
        params.input_mode = InputMode::Temperature;
        read_t3ster_input(&args, &mut params)?
    } else {
        params.input_mode = args.input_mode;
        let input = args.input.as_ref().ok_or("missing input path")?;
        read_two_column_data(input)?
    };
    params.deconv_mode = args.deconv_mode;
    params.structure_method = args.structure_method;
    if let Some(precision) = args.precision {
        params.precision = precision;
    }
    params.filter_name = args.filter_name;
    if let Some(filter_range) = args.filter_range {
        params.filter_range = filter_range;
    }
    if let Some(filter_parameter) = args.filter_parameter {
        params.filter_parameter = filter_parameter;
    }
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
    if let Some(lasso_alpha) = args.lasso_alpha {
        params.lasso_alpha = lasso_alpha;
    }
    if let Some(lasso_max_iter) = args.lasso_max_iter {
        params.lasso_max_iter = lasso_max_iter;
    }
    if let Some(lasso_tol) = args.lasso_tol {
        params.lasso_tol = lasso_tol;
    }
    if let Some(timespec_interpolate_factor) = args.timespec_interpolate_factor {
        params.timespec_interpolate_factor = timespec_interpolate_factor;
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
    params.extrapolate = args.extrapolate;
    params.lower_fit_limit = args.lower_fit_limit;
    params.upper_fit_limit = args.upper_fit_limit;

    let result = evaluate(input, &params)?;
    export_csv(&result, &args.output_dir)?;
    if let Some(figures_output_dir) = args.figures_output_dir.as_ref() {
        export_svg_figures(&result, figures_output_dir)?;
    }

    Ok(())
}

struct CliArgs {
    input: Option<PathBuf>,
    output_dir: PathBuf,
    figures_output_dir: Option<PathBuf>,
    input_mode: InputMode,
    deconv_mode: DeconvMode,
    structure_method: StructureMethod,
    precision: Option<usize>,
    filter_name: FourierFilter,
    filter_range: Option<f64>,
    filter_parameter: Option<f64>,
    only_make_z: bool,
    no_structure: bool,
    log_time_size: Option<usize>,
    bay_steps: Option<usize>,
    blockwise_sum_width: Option<usize>,
    min_index: Option<usize>,
    minimum_window_size: Option<usize>,
    lasso_alpha: Option<f64>,
    lasso_max_iter: Option<usize>,
    lasso_tol: Option<f64>,
    timespec_interpolate_factor: Option<f64>,
    power_step: Option<f64>,
    power_scale_factor: Option<f64>,
    optical_power: Option<f64>,
    is_heating: bool,
    kfac_fit_deg: Option<usize>,
    calibration: Option<PathBuf>,
    t3ster_power: Option<PathBuf>,
    t3ster_calibration: Option<PathBuf>,
    data_cut_lower: Option<usize>,
    data_cut_upper: Option<usize>,
    temp_0_avg_range: Option<(usize, usize)>,
    extrapolate: bool,
    lower_fit_limit: Option<f64>,
    upper_fit_limit: Option<f64>,
    theoretical_resistance: Option<Vec<f64>>,
    theoretical_capacitance: Option<Vec<f64>>,
    time_start: Option<f64>,
    time_end: Option<f64>,
    time_size: Option<usize>,
}

impl CliArgs {
    fn parse(args: impl Iterator<Item = String>) -> Result<Self, Box<dyn Error>> {
        let mut input = None;
        let mut output_dir = None;
        let mut figures_output_dir = None;
        let mut input_mode = InputMode::Impedance;
        let mut deconv_mode = DeconvMode::Bayesian;
        let mut structure_method = StructureMethod::Lanczos;
        let mut precision = None;
        let mut filter_name = FourierFilter::Hann;
        let mut filter_range = None;
        let mut filter_parameter = None;
        let mut only_make_z = false;
        let mut no_structure = false;
        let mut log_time_size = None;
        let mut bay_steps = None;
        let mut blockwise_sum_width = None;
        let mut min_index = None;
        let mut minimum_window_size = None;
        let mut lasso_alpha = None;
        let mut lasso_max_iter = None;
        let mut lasso_tol = None;
        let mut timespec_interpolate_factor = None;
        let mut power_step = None;
        let mut power_scale_factor = None;
        let mut optical_power = None;
        let mut is_heating = false;
        let mut kfac_fit_deg = None;
        let mut calibration = None;
        let mut t3ster_power = None;
        let mut t3ster_calibration = None;
        let mut data_cut_lower = None;
        let mut data_cut_upper = None;
        let mut temp_0_avg_range = None;
        let mut extrapolate = false;
        let mut lower_fit_limit = None;
        let mut upper_fit_limit = None;
        let mut theoretical_resistance = None;
        let mut theoretical_capacitance = None;
        let mut time_start = None;
        let mut time_end = None;
        let mut time_size = None;

        let mut args = args.peekable();
        while let Some(arg) = args.next() {
            match arg.as_str() {
                "--help" | "-h" => {
                    print_usage();
                    std::process::exit(0);
                }
                "--input" | "-i" => input = args.next().map(PathBuf::from),
                "--output" | "-o" => output_dir = args.next().map(PathBuf::from),
                "--figures-output" => figures_output_dir = args.next().map(PathBuf::from),
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
                "--lasso-alpha" => lasso_alpha = Some(parse_next_f64(&mut args, "--lasso-alpha")?),
                "--lasso-max-iter" => {
                    lasso_max_iter = Some(parse_next_usize(&mut args, "--lasso-max-iter")?)
                }
                "--lasso-tol" => lasso_tol = Some(parse_next_f64(&mut args, "--lasso-tol")?),
                "--timespec-interpolate-factor" => {
                    timespec_interpolate_factor =
                        Some(parse_next_f64(&mut args, "--timespec-interpolate-factor")?)
                }
                "--input-mode" => {
                    let value = args.next().ok_or("missing value for --input-mode")?;
                    input_mode = InputMode::from_label(&value)?;
                }
                "--deconv" | "--deconv-mode" => {
                    let value = args.next().ok_or("missing value for --deconv")?;
                    deconv_mode = DeconvMode::from_label(&value)?;
                }
                "--structure-method" | "--struc-method" => {
                    let value = args.next().ok_or("missing value for --structure-method")?;
                    structure_method = StructureMethod::from_label(&value)?;
                }
                "--precision" => precision = Some(parse_next_usize(&mut args, "--precision")?),
                "--filter-name" | "--filter" => {
                    let value = args.next().ok_or("missing value for --filter-name")?;
                    filter_name = FourierFilter::from_label(&value)?;
                }
                "--filter-range" => {
                    filter_range = Some(parse_next_f64(&mut args, "--filter-range")?)
                }
                "--filter-parameter" => {
                    filter_parameter = Some(parse_next_f64(&mut args, "--filter-parameter")?)
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
                "--t3ster-power" => t3ster_power = args.next().map(PathBuf::from),
                "--t3ster-calibration" => t3ster_calibration = args.next().map(PathBuf::from),
                "--data-cut-lower" => {
                    data_cut_lower = Some(parse_next_usize(&mut args, "--data-cut-lower")?)
                }
                "--data-cut-upper" => {
                    data_cut_upper = Some(parse_next_usize(&mut args, "--data-cut-upper")?)
                }
                "--temp-zero-range" => {
                    temp_0_avg_range = Some(parse_next_range(&mut args, "--temp-zero-range")?)
                }
                "--extrapolate" => extrapolate = true,
                "--lower-fit-limit" => {
                    lower_fit_limit = Some(parse_next_f64(&mut args, "--lower-fit-limit")?)
                }
                "--upper-fit-limit" => {
                    upper_fit_limit = Some(parse_next_f64(&mut args, "--upper-fit-limit")?)
                }
                "--theoretical-resistance" => {
                    theoretical_resistance =
                        Some(parse_next_f64_list(&mut args, "--theoretical-resistance")?)
                }
                "--theoretical-capacitance" => {
                    theoretical_capacitance =
                        Some(parse_next_f64_list(&mut args, "--theoretical-capacitance")?)
                }
                "--time-start" => time_start = Some(parse_next_f64(&mut args, "--time-start")?),
                "--time-end" => time_end = Some(parse_next_f64(&mut args, "--time-end")?),
                "--time-size" => time_size = Some(parse_next_usize(&mut args, "--time-size")?),
                _ if input.is_none() => input = Some(PathBuf::from(arg)),
                _ if output_dir.is_none() => output_dir = Some(PathBuf::from(arg)),
                _ => return Err(format!("unknown argument: {arg}").into()),
            }
        }

        validate_theoretical_args(
            theoretical_resistance.as_deref(),
            theoretical_capacitance.as_deref(),
            time_start,
            time_end,
            time_size,
        )?;

        if theoretical_resistance.is_none() && input.is_none() {
            return Err("missing input path".into());
        }

        Ok(Self {
            input,
            output_dir: output_dir.unwrap_or_else(|| PathBuf::from("output/rust-cli")),
            figures_output_dir,
            input_mode,
            deconv_mode,
            structure_method,
            precision,
            filter_name,
            filter_range,
            filter_parameter,
            only_make_z,
            no_structure,
            log_time_size,
            bay_steps,
            blockwise_sum_width,
            min_index,
            minimum_window_size,
            lasso_alpha,
            lasso_max_iter,
            lasso_tol,
            timespec_interpolate_factor,
            power_step,
            power_scale_factor,
            optical_power,
            is_heating,
            kfac_fit_deg,
            calibration,
            t3ster_power,
            t3ster_calibration,
            data_cut_lower,
            data_cut_upper,
            temp_0_avg_range,
            extrapolate,
            lower_fit_limit,
            upper_fit_limit,
            theoretical_resistance,
            theoretical_capacitance,
            time_start,
            time_end,
            time_size,
        })
    }

    fn has_theoretical_model(&self) -> bool {
        self.theoretical_resistance.is_some()
    }
}

fn print_usage() {
    println!(
        "Usage: pyrth-cli [--input <path>] --output <dir> [--figures-output <dir>] [--theoretical-resistance <r1,r2>] [--theoretical-capacitance <c1,c2>] [--time-start <t>] [--time-end <t>] [--time-size <n>] [--input-mode impedance|temp|volt|t3ster] [--deconv bayesian|fourier|lasso|adaptive] [--structure-method lanczos|sobhy|boor_golub|khatwani|polylong] [--precision <bits>] [--filter-name hann|rectangular|gauss|fermi|nuttall|blackman_nuttall|blackman_harris] [--filter-range <x>] [--filter-parameter <x>] [--power-step <w>] [--power-scale-factor <x>] [--optical-power <w>] [--is-heating] [--calibration <path>] [--t3ster-power <path>] [--t3ster-calibration <path>] [--kfac-fit-deg <n>] [--data-cut-lower <n>] [--data-cut-upper <n>] [--temp-zero-range <start:end>] [--extrapolate --lower-fit-limit <t> --upper-fit-limit <t>] [--only-make-z] [--no-structure] [--log-time-size <n>] [--bay-steps <n>] [--blockwise-sum-width <n>] [--min-index <n>] [--minimum-window-size <n>] [--lasso-alpha <x>] [--lasso-max-iter <n>] [--lasso-tol <x>] [--timespec-interpolate-factor <x>]"
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

fn parse_next_f64_list(
    args: &mut std::iter::Peekable<impl Iterator<Item = String>>,
    flag: &str,
) -> Result<Vec<f64>, Box<dyn Error>> {
    let value = args
        .next()
        .ok_or_else(|| format!("missing value for {flag}"))?;
    let values = value
        .split(',')
        .map(str::trim)
        .map(|part| {
            if part.is_empty() {
                Err(format!("invalid value for {flag}: empty list item").into())
            } else {
                part.parse::<f64>()
                    .map_err(|err| format!("invalid value for {flag}: {part} ({err})").into())
            }
        })
        .collect::<Result<Vec<_>, Box<dyn Error>>>()?;

    if values.is_empty() {
        return Err(format!("invalid value for {flag}: expected at least one number").into());
    }

    Ok(values)
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

fn validate_theoretical_args(
    theoretical_resistance: Option<&[f64]>,
    theoretical_capacitance: Option<&[f64]>,
    time_start: Option<f64>,
    time_end: Option<f64>,
    time_size: Option<usize>,
) -> Result<(), Box<dyn Error>> {
    let has_resistance = theoretical_resistance.is_some();
    let has_capacitance = theoretical_capacitance.is_some();
    let has_time = time_start.is_some() || time_end.is_some() || time_size.is_some();

    if !has_resistance && !has_capacitance && !has_time {
        return Ok(());
    }
    if !has_resistance || !has_capacitance {
        return Err(
            "--theoretical-resistance and --theoretical-capacitance must be specified together"
                .into(),
        );
    }
    if time_start.is_none() || time_end.is_none() || time_size.is_none() {
        return Err(
            "--time-start, --time-end, and --time-size are required for theoretical input".into(),
        );
    }

    let resistance_len = theoretical_resistance.expect("checked above").len();
    let capacitance_len = theoretical_capacitance.expect("checked above").len();
    if resistance_len != capacitance_len {
        return Err(format!(
            "theoretical resistance/capacitance length mismatch: {resistance_len} != {capacitance_len}"
        )
        .into());
    }

    Ok(())
}

fn build_theoretical_input(args: &CliArgs) -> Result<TransientInput, Box<dyn Error>> {
    theoretical_impedance_input(
        args.theoretical_resistance
            .as_deref()
            .ok_or("missing --theoretical-resistance")?,
        args.theoretical_capacitance
            .as_deref()
            .ok_or("missing --theoretical-capacitance")?,
        args.time_start.ok_or("missing --time-start")?,
        args.time_end.ok_or("missing --time-end")?,
        args.time_size.ok_or("missing --time-size")?,
    )
    .map_err(Into::into)
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

fn read_t3ster_input(
    args: &CliArgs,
    params: &mut EvaluationParams,
) -> Result<TransientInput, Box<dyn Error>> {
    let input = args.input.as_ref().ok_or("missing input path")?;
    let raw = parse_t3ster_raw_text(&fs::read_to_string(input)?)?;
    let calibration_path = args
        .t3ster_calibration
        .as_ref()
        .or(args.calibration.as_ref())
        .ok_or("--input-mode t3ster requires --t3ster-calibration or --calibration")?;
    let calibration = parse_t3ster_calibration_text(&fs::read_to_string(calibration_path)?)?;

    if let Some(power_path) = args.t3ster_power.as_ref() {
        params.power_step = parse_t3ster_power_step(&fs::read_to_string(power_path)?)?;
    }

    t3ster_raw_to_temperature_input(&raw, &calibration, params.kfac_fit_deg).map_err(Into::into)
}
