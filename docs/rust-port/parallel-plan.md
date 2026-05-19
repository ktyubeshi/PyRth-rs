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

Status: implemented in core and wired through CLI/PyO3.

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

### E. Lasso Deconvolution

Status: implemented in core with deterministic non-negative coordinate descent.

Owned paths:

- `crates/pyrth-core/src/config.rs`
- `crates/pyrth-core/src/deconvolution.rs`
- `crates/pyrth-core/src/evaluation.rs`
- `crates/pyrth-core/tests/golden.rs`

Goal:

- Provide a deterministic sparse deconvolution path without pulling in a
  scikit-learn equivalent.
- Keep adaptive mode explicitly unsupported until a separate implementation is
  available.

### F. Theoretical And Prediction

Status: theoretical RC impedance generation, standard temperature prediction,
reusable RC/optimization-result temperature prediction helpers, and the matching
PyO3 facades are implemented.

Owned paths:

- `crates/pyrth-core/src/theoretical.rs`
- `crates/pyrth-core/src/prediction.rs`
- `crates/pyrth-core/tests/theoretical.rs`
- `crates/pyrth-core/tests/prediction.rs`

Goal:

- Build the foundation for comparison, bootstrap-from-theoretical, and
  temperature prediction workflows.  The remaining work is Python-style
  orchestration that runs a full optimization module internally before
  prediction.

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
  generated theoretical impedance inputs.  Bootstrap and optimization
  comparison modes remain separate work.
- PyO3 `save_as_csv` exports every registered standard-evaluation module into
  per-label output directories.

### H. Theoretical Bootstrap

Status: deterministic bootstrap from theoretical RC models and existing
impedance inputs is implemented in core, including mean and 10/50/90 percentile
bands.  The PyO3 bootstrap facade also accepts Python-style theoretical aliases
and signal-to-noise input.

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

Status: feature-gated `polylong`, `sobhy`, and `khatwani` paths are
implemented behind the non-default `mpfr` feature.  A raw feature-gated
Boor-Golub helper is available for parity investigation, but it is not connected
to `evaluate` yet.  `rug` is optional, and Windows MSVC cannot currently build
`gmp-mpfr-sys`, so MPFR verification needs a supported GNU/Linux or Windows GNU
toolchain.

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
- Continue by reconciling Boor-Golub's raw Python-shaped output with Rust's
  `CauerNetwork` contract and adding stronger parity coverage on a toolchain
  supported by `gmp-mpfr-sys`.

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
