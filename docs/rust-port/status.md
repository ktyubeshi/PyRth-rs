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
- Rust core pipeline for `input_mode="impedance"`:
  - impedance ingest
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
- Minimal PyO3 entrypoint:
  - `evaluate_impedance(data, only_make_z=False, calc_struc=True)`

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
- PyO3 does not yet expose the existing Python `Evaluation().standard_module`
  compatibility facade.
- CLI only supports direct two-column impedance input.
- Fourier, Lasso, adaptive, MPFR structure methods, optimization, bootstrap,
  comparison, and temperature prediction are not ported.

## Useful Commands

```powershell
uv run --python 3.12 --with-editable . python tests/golden/generate_golden.py
uv run --python 3.12 --with-editable . --with pytest pytest tests/cases/test_standard_module.py -k MOSFET_tim_basic_lanczos
cargo test
cargo run -p pyrth-cli -- --input target\tmp\cli-input.csv --output target\tmp\cli-smoke --only-make-z
```
