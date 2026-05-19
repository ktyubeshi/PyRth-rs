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

Status: implemented as a minimal unity-filter FFT path.

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
