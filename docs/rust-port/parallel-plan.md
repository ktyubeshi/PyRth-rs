# Parallel Rust Port Plan

This note tracks the current parallel implementation slices.  The repository is
managed with `jj` in colocated mode, so each slice should stay small enough to
review, test, and land independently.

## Current Baseline

- Base Git commit before `jj` setup: `d8873445` (`Update status for preprocessing parameters`)
- Local `jj` parent change: `Ignore local tmp workspace files`
- Working-copy rule: use `jj status` as the source of truth while implementing;
  `git status` may report fixture line-ending noise in the colocated checkout.

## Active Slices

### A. Fourier Deconvolution

Status: implemented with Python-compatible filter options.

Owned paths:

- `crates/pyrth-core/src/deconvolution.rs`
- `crates/pyrth-core/src/evaluation.rs`
- `crates/pyrth-core/tests/golden.rs` or a focused new core test
- `Cargo.toml` / `Cargo.lock` only if an FFT dependency is required

Goal:

- Make `DeconvMode::Fourier` produce a `TimeSpectrum` through the existing
  evaluation pipeline.
- Keep Bayesian behavior and golden coverage unchanged.

Suggested verification:

```powershell
cargo test -p pyrth-core
```

### B. Temperature Extrapolation

Status: implemented in core and wired through CLI/PyO3.  Temperature and voltage
preprocessing now reject non-positive effective power and the evaluation
pipeline validates the post-preprocessing sample count before derivative work.

Owned paths:

- `crates/pyrth-core/src/config.rs`
- `crates/pyrth-core/src/preprocess.rs`
- `crates/pyrth-core/tests/preprocess.rs`

Goal:

- Add a default-off `extrapolate` parameter for temperature preprocessing.
- Implement an early-time square-root fit path based on the Python
  `extrapolate_temperature` reference.
- Leave CLI and PyO3 wiring for a later integration slice.

Suggested verification:

```powershell
cargo test -p pyrth-core
```

### C. Coordination And Status

Status: active.

Owned paths:

- `docs/rust-port/status.md`
- `docs/rust-port/parallel-plan.md`

Goal:

- Keep the state of the parallel work understandable.
- Record which gaps remain after each slice lands.

### D. T3Ster Input

Status: core text parsers, CLI `--input-mode t3ster`, and PyO3 file-path
ingestion are implemented.

Owned paths:

- `crates/pyrth-core/src/t3ster.rs`
- `crates/pyrth-core/tests/preprocess.rs`
- `crates/pyrth-cli/src/main.rs`
- later: `crates/pyrth-py/src/lib.rs`

Goal:

- Convert legacy `.raw` ADC records plus `.pwr` and `.tco` companion files
  into the existing temperature preprocessing path.
- Keep file I/O at the CLI/PyO3 boundary and keep core parser functions
  testable from strings.

### E. Lasso And Adaptive Deconvolution

Status: Lasso is implemented in core with deterministic non-negative coordinate
descent.  Adaptive mode reuses the deterministic sparse solver with Python-style
Bayesian prior column weights.  The Lasso design matrix now uses an
impedance-domain step-response basis when the smoothed impedance grid is
available, with a derivative-response fallback.

Owned paths:

- `crates/pyrth-core/src/config.rs`
- `crates/pyrth-core/src/deconvolution.rs`
- `crates/pyrth-core/src/evaluation.rs`
- `crates/pyrth-core/tests/golden.rs`

Goal:

- Provide deterministic sparse deconvolution paths without pulling in a
  scikit-learn equivalent.
- Continue by moving Lasso/adaptive from the smoothed derivative grid toward
  Python's raw input-time design matrix and optional cross-validation behavior.

### F. Theoretical And Prediction

Status: theoretical RC impedance generation, standard temperature prediction,
reusable RC/optimization-result temperature prediction helpers, and the matching
PyO3 facades are implemented.  The PyO3 temperature prediction facade can also
run the current RC optimization helper internally before prediction, and labeled
temperature-prediction calls are registered in the module bookkeeping layer and
exported as `temperature_prediction.csv` / `.svg`.

Owned paths:

- `crates/pyrth-core/src/theoretical.rs`
- `crates/pyrth-core/src/prediction.rs`
- `crates/pyrth-core/tests/theoretical.rs`
- `crates/pyrth-core/tests/prediction.rs`

Goal:

- Build the foundation for comparison, bootstrap-from-theoretical, and
  temperature prediction workflows.  The remaining work is full Python-style
  orchestration plus richer power/temperature prediction export parity.

### G. Comparison Metrics

Status: core result comparison metrics are implemented.

Owned paths:

- `crates/pyrth-core/src/comparison.rs`
- `crates/pyrth-core/tests/comparison.rs`

Goal:

- Compare evaluated spectra and structure functions with relative L2 norms.
- Report total resistance differences using Cauer output when available and
  Foster output as a fallback.
- The PyO3 `standard_module_set` facade now supports standard-evaluation
  sweeps, which is the next building block for `comparison_module` parity.
- PyO3 `comparison_module` now covers standard-evaluation sweeps against
  generated theoretical impedance inputs, bootstrap sweeps over mean spectra,
  and optimization sweeps over the current RC optimizer helper.
- PyO3 `save_as_csv` exports every registered standard-evaluation module into
  per-label output directories.
- PyO3 `save_figures` exports simple SVG figures for every registered
  standard-evaluation module into per-label output directories, and `save_all`
  now returns both CSV and figure path dictionaries.
- PyO3 `save_as_csv`, `save_figures`, and `save_all` also export labeled
  temperature-prediction modules.
- PyO3 voltage input accepts both `calibration` and the Python-compatible
  `calib` alias.
- CLI `--figures-output` writes the same simple SVG figures next to the
  existing CSV output path when requested.
- Core SVG export now records and applies simple linear/log axis scales for the
  matching Python figure families.
- PyO3 standard-evaluation methods now return minimal `StructureFunction`
  objects with dict-like read access rather than bare dictionaries.
- PyO3 `StructureFunction` also exposes common Python-style array aliases for
  derivative, Foster, and Cauer outputs when present.

### H. Theoretical Bootstrap

Status: deterministic bootstrap from theoretical RC models and existing
impedance inputs is implemented in core, including mean and 10/50/90 percentile
bands.  The PyO3 bootstrap facade also accepts Python-style theoretical aliases
and signal-to-noise input, and can run seeded bootstrap from existing
impedance `data`.

Owned paths:

- `crates/pyrth-core/src/bootstrap.rs`
- `crates/pyrth-core/tests/bootstrap.rs`
- `Cargo.toml`
- `Cargo.lock`

Goal:

- Generate noisy theoretical impedance traces with a fixed seed.
- Re-run the existing evaluation pipeline and average successful impedance and
  time-spectrum outputs.

### I. MPFR Structure Methods

Status: feature-gated `polylong`, `sobhy`, `khatwani`, and `boor_golub` paths
are implemented behind the non-default `mpfr` feature.  Boor-Golub adapts the
Python-shaped raw output by dropping the trailing zero-resistance sentinel
before constructing Rust's `CauerNetwork`.  `rug` is optional, and Windows MSVC
cannot currently build `gmp-mpfr-sys`, so MPFR verification needs a supported
GNU/Linux or Windows GNU toolchain.  Lanczos Cauer now stops before non-finite
tails and has finite positive single/two-branch guards, but full-array Python
golden equality remains unresolved.
Short Lanczos networks are preserved when `blockwise_sum_width` is larger than
the branch count.

Owned paths:

- `crates/pyrth-core/src/network/*`
- `crates/pyrth-core/src/evaluation.rs` structure-method branch only
- `crates/pyrth-core/tests/structure_methods.rs`
- `docs/rust-port/status.md`
- `docs/rust-port/parallel-plan.md`

Goal:

- Port Python `transient_mpfr_utils.py` methods (`polylong`, `sobhy`,
  `khatwani`, `boor_golub`) without mislabeling Lanczos or f64 scaffolding as
  MPFR-equivalent output.
- Continue by adding stronger MPFR parity coverage on a toolchain supported by
  `gmp-mpfr-sys`.

## Landing Rules

- Do not edit another slice's owned paths unless explicitly coordinating.
- Do not regenerate golden fixtures as part of infrastructure-only changes.
- Prefer focused `cargo test -p pyrth-core` runs while working, then run the
  wider suite after integrating a completed slice.
- Before landing a slice, check:

```powershell
jj status
jj diff --stat
```
