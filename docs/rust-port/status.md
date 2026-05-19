# Rust Port Status

This document tracks the current Rust-port scope and known gaps.  The Python
implementation remains the reference implementation.

## Implemented

- Workspace scaffold:
  - `crates/pyrth-core`
  - `crates/pyrth-py`
  - `crates/pyrth-cli`
- Python golden fixture generation:
  - `tests/golden/generate_golden.py`
  - Bayesian + Lanczos reference fixtures for MOSFET TIM, MOSFET dry, and LED
- Rust core pipeline for direct two-column inputs:
  - `input_mode="impedance"` ingest
  - `input_mode="temp"` to impedance conversion, including optional
    early-time square-root extrapolation
  - `input_mode="volt"` to impedance conversion with polynomial calibration,
    including optional early-time square-root extrapolation after voltage to
    temperature conversion
  - log-time derivative preprocessing
  - Bayesian deconvolution
  - Fourier deconvolution with `hann`, `rectangular`, `gauss`, `fermi`,
    `nuttall`, `blackman_nuttall`, and `blackman_harris` filters
  - Lasso deconvolution with deterministic non-negative coordinate descent
  - Adaptive deconvolution as deterministic scaled-alpha sparse Lasso
  - Foster network conversion
  - Lanczos Cauer conversion
- Theoretical helpers:
  - Foster RC arrays to theoretical impedance input
  - validated Foster RC optimization helper parameters, flattening, bounds
    checks, and impedance residual norms
  - deterministic bounded coordinate-search solver for Foster RC parameters
  - standard temperature prediction by interpolating power and impulse response
  - comparison metrics for two evaluated results
  - deterministic bootstrap means from theoretical RC models
- T3Ster text helpers:
  - legacy `.raw` header/data parser
  - `.pwr` power-step parser
  - `.tco` calibration parser
  - raw ADC to temperature input conversion
- Core CSV export:
  - `impedance.csv`
  - `imp_deriv.csv`
  - `time_spec.csv`
  - `foster.csv`
  - `cauer.csv`
  - `diff_struc.csv`
- Evaluation gates:
  - `only_make_z`
  - `calc_struc`
- Minimal CLI:
  - two-column CSV/whitespace input
  - core CSV output
  - `--only-make-z`
  - `--no-structure`
  - `--log-time-size`
  - `--bay-steps`
  - `--blockwise-sum-width`
  - `--min-index`
  - `--minimum-window-size`
  - `--input-mode`
  - `--deconv`
  - `--filter-name`
  - `--filter-range`
  - `--filter-parameter`
  - `--power-step`
  - `--power-scale-factor`
  - `--optical-power`
  - `--is-heating`
  - `--calibration`
  - `--kfac-fit-deg`
  - `--data-cut-lower`
  - `--data-cut-upper`
  - `--temp-zero-range`
  - `--extrapolate`
  - `--lower-fit-limit`
  - `--upper-fit-limit`
  - `--t3ster-power`
  - `--t3ster-calibration`
  - `--theoretical-resistance`
  - `--theoretical-capacitance`
  - `--time-start`
  - `--time-end`
  - `--time-size`
- Minimal PyO3 entrypoint:
  - `evaluate_impedance(data, only_make_z=False, calc_struc=True)`
  - `theoretical_impedance(resistance, capacitance, time_start, time_end,
    time_size)`
  - `bootstrap_theoretical(resistance, capacitance, time_start, time_end,
    time_size, repetitions, noise_std, seed=0)`
  - `Evaluation().standard_module({"data": ...})`
  - `Evaluation().standard_module({...})` with `input_mode="t3ster"` and
    `infile`/`infile_pwr`/`infile_tco` or `input`/`t3ster_power`/
    `t3ster_calibration`
  - `standard_module` accepts `input_mode`, `only_make_z`, `calc_struc`,
    `deconv_mode`, `filter_name`, `filter_range`, `filter_parameter`,
    `log_time_size`, `bay_steps`, `min_index`, `minimum_window_size`,
    `timespec_interpolate_factor`, `blockwise_sum_width`, `struc_method`,
    `structure_method`, `precision`, `lasso_alpha`, `lasso_max_iter`,
    `lasso_tol`, `pad_factor_pre`, `pad_factor_after`,
    `minimum_window_length`, `maximum_window_length`, `window_increment`,
    `expected_var`, `power_step`, `power_scale_factor`, `optical_power`,
    `is_heating`, `calibration`, `kfac_fit_deg`, `data_cut_lower`,
    `data_cut_upper`, `temp_0_avg_range`, `extrapolate`,
    `lower_fit_limit`, and `upper_fit_limit`

## Golden Coverage

Strict golden comparisons currently cover:

- impedance ingest for MOSFET TIM, MOSFET dry, and LED
- derivative preprocessing for MOSFET TIM and MOSFET dry
- Bayesian `time_spec` for MOSFET TIM and MOSFET dry
- Foster network resistance/capacitance for MOSFET TIM and MOSFET dry
- Lasso smoke coverage for finite, non-negative sparse spectrum
- Adaptive smoke coverage for finite, non-negative sparse spectrum
- theoretical single/multiple RC impedance generation
- standard temperature prediction finite output
- comparison metric behavior for spectra, structure functions, and resistance
- optimization helper validation, flatten/unflatten, bounds, and theoretical
  impedance residuals, including bounded coordinate-search improvement checks
- deterministic theoretical bootstrap means

Lanczos Cauer coverage currently checks:

- golden Foster input to Rust Cauer conversion
- first cumulative Cauer block against Python golden for MOSFET TIM and MOSFET dry
- non-empty, finite, non-negative Cauer branches
- monotonic cumulative resistance

## Known Gaps

- Lanczos Cauer full-array golden equality is not yet achieved.  Initial
  Lanczos steps match Python closely, but long recurrence drift changes the
  stopping point and therefore the blockwise output length.
- LED derivative golden equality is not enabled.  Its small-window settings
  hit near-ties in the adaptive estimator and currently diverge by window
  selection in a few positions.
- PyO3 returns plain Python dictionaries rather than existing Python
  `StructureFunction` objects.
- Adaptive deconvolution is a minimal deterministic sparse implementation, not
  full Python adaptive parity.
- MPFR structure methods and full Python optimization parity are not ported.
  Rust now has a small f64 Foster rational assembly and poly-long helper under
  `network` for follow-on MPFR work, but `evaluate` still reports `sobhy`,
  `khatwani`, `boor_golub`, and `polylong` as unsupported rather than
  presenting finite-precision scaffolding as Python MPFR parity.
- Bootstrap and comparison currently cover core numerical helpers only; the
  wider Python module orchestration and exporter parity are not ported.
- Temperature prediction currently supports standard-evaluation impulse
  responses only; optimization-based prediction is not ported.
- The next implementation work is split into non-overlapping `jj` slices in
  `docs/rust-port/parallel-plan.md`.

## Useful Commands

```powershell
jj status
jj diff --stat
uv run --python 3.12 --with-editable . python tests/golden/generate_golden.py
uv run --python 3.12 --with-editable . --with pytest pytest tests/cases/test_standard_module.py -k MOSFET_tim_basic_lanczos
cargo test
cargo run -p pyrth-cli -- --input target\tmp\cli-input.csv --output target\tmp\cli-smoke --only-make-z
cargo run -p pyrth-cli -- --input target\tmp\cli-input.csv --output target\tmp\cli-full-small --log-time-size 10 --bay-steps 2 --min-index 1 --minimum-window-size 2 --no-structure
cargo run -p pyrth-cli -- --input target\tmp\temp-input.csv --output target\tmp\cli-temp --input-mode temp --power-step 2 --temp-zero-range 0:1 --only-make-z
cargo run -p pyrth-cli -- --input target\tmp\volt-input.csv --output target\tmp\cli-volt --input-mode volt --calibration target\tmp\calib.csv --kfac-fit-deg 1 --only-make-z
cargo run -p pyrth-cli -- --input target\tmp\temp-extrapolate.csv --output target\tmp\cli-temp-extrapolate --input-mode temp --extrapolate --lower-fit-limit 4 --upper-fit-limit 16 --only-make-z
cargo run -p pyrth-cli -- --input tests\data\MOSFET_tim.txt --output target\tmp\cli-fourier-filter --deconv fourier --filter-name rectangular --filter-range 0.6 --log-time-size 12 --min-index 1 --minimum-window-size 2 --no-structure
cargo run -p pyrth-cli -- --input tests\data\t3ster\T25_I-m5m-I-h600m_100s.raw --input-mode t3ster --t3ster-power tests\data\t3ster\T25_I-m5m-I-h600m_100s.pwr --t3ster-calibration tests\data\t3ster\calib.tco --output target\tmp\cli-t3ster --only-make-z
cargo run -p pyrth-cli -- --theoretical-resistance 1,2 --theoretical-capacitance 0.5,1.5 --time-start 1e-6 --time-end 1e-2 --time-size 32 --output target\tmp\cli-theoretical --only-make-z
uvx maturin develop --manifest-path crates/pyrth-py/Cargo.toml
.\.venv\Scripts\python.exe crates\pyrth-py\tests\smoke.py
cargo test -p pyrth-py
cargo test -p pyrth-core --test theoretical
cargo test -p pyrth-core --test prediction
```
