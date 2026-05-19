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
  - `input_mode="temp"` to impedance conversion without extrapolation
  - `input_mode="volt"` to impedance conversion with polynomial calibration,
    without extrapolation
  - log-time derivative preprocessing
  - Bayesian deconvolution
  - Foster network conversion
  - Lanczos Cauer conversion
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
  - `--power-step`
  - `--power-scale-factor`
  - `--optical-power`
  - `--is-heating`
  - `--calibration`
  - `--kfac-fit-deg`
  - `--data-cut-lower`
  - `--data-cut-upper`
  - `--temp-zero-range`
- Minimal PyO3 entrypoint:
  - `evaluate_impedance(data, only_make_z=False, calc_struc=True)`
  - `Evaluation().standard_module({"data": ...})`
  - `standard_module` accepts `input_mode`, `only_make_z`, `calc_struc`,
    `log_time_size`, `bay_steps`, `blockwise_sum_width`, `power_step`,
    `power_scale_factor`, `optical_power`, `is_heating`, `calibration`,
    `kfac_fit_deg`, `data_cut_lower`, `data_cut_upper`, and
    `temp_0_avg_range`

## Golden Coverage

Strict golden comparisons currently cover:

- impedance ingest for MOSFET TIM, MOSFET dry, and LED
- derivative preprocessing for MOSFET TIM and MOSFET dry
- Bayesian `time_spec` for MOSFET TIM and MOSFET dry
- Foster network resistance/capacitance for MOSFET TIM and MOSFET dry

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
- CLI/PyO3 only support direct two-column inputs. T3Ster files and
  extrapolated temp/volt preprocessing are not ported.
- PyO3 returns plain Python dictionaries rather than existing Python
  `StructureFunction` objects.
- Fourier, Lasso, adaptive, MPFR structure methods, optimization, bootstrap,
  comparison, and temperature prediction are not ported.

## Useful Commands

```powershell
uv run --python 3.12 --with-editable . python tests/golden/generate_golden.py
uv run --python 3.12 --with-editable . --with pytest pytest tests/cases/test_standard_module.py -k MOSFET_tim_basic_lanczos
cargo test
cargo run -p pyrth-cli -- --input target\tmp\cli-input.csv --output target\tmp\cli-smoke --only-make-z
cargo run -p pyrth-cli -- --input target\tmp\cli-input.csv --output target\tmp\cli-full-small --log-time-size 10 --bay-steps 2 --min-index 1 --minimum-window-size 2 --no-structure
cargo run -p pyrth-cli -- --input target\tmp\temp-input.csv --output target\tmp\cli-temp --input-mode temp --power-step 2 --temp-zero-range 0:1 --only-make-z
cargo run -p pyrth-cli -- --input target\tmp\volt-input.csv --output target\tmp\cli-volt --input-mode volt --calibration target\tmp\calib.csv --kfac-fit-deg 1 --only-make-z
uvx maturin develop --manifest-path crates/pyrth-py/Cargo.toml
.\.venv\Scripts\python.exe crates\pyrth-py\tests\smoke.py
```
