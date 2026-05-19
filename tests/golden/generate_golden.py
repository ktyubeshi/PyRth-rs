"""Generate Python reference fixtures for the Rust port.

The Rust implementation uses these JSON files as the contract for each
pipeline stage.  Fixtures intentionally store intermediate arrays instead of
only end results so deviations can be isolated to ingest, differentiation,
deconvolution, Foster conversion, or Cauer conversion.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from typing import Any

import numpy as np


ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

from PyRth import Evaluation  # noqa: E402
from PyRth.utils import transient_utils as utl  # noqa: E402


FIXTURE_DIR = ROOT / "tests" / "golden" / "fixtures"
DATA_DIR = ROOT / "tests" / "data"

MOSFET_CALIB_DATA = np.array(
    [
        [23.4, 0.55843],
        [37.625, 0.52536],
        [51.85, 0.49232],
        [66.075, 0.45927],
        [80.3, 0.42621],
    ]
)

LED_CALIB_DATA = np.array(
    [
        [20.0, 2.53473992],
        [30.0, 2.55473992],
        [40.0, 2.57473992],
        [50.0, 2.59473992],
        [60.0, 2.61473992],
    ]
)


def read_voltage_data(filepath: Path) -> np.ndarray:
    """Read legacy measurement text files used by the existing tests."""

    lines = filepath.read_text(encoding="utf-8").splitlines()
    data_start = next(i for i, line in enumerate(lines) if line.strip() == "DATA") + 2
    rows = [line.split() for line in lines[data_start:] if line.strip()]
    return np.array([[float(value) for value in row] for row in rows])


def to_jsonable(value: Any) -> Any:
    """Convert NumPy values and paths into stable JSON-compatible objects."""

    if isinstance(value, np.ndarray):
        return value.tolist()
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, Path):
        return value.as_posix()
    if isinstance(value, dict):
        return {str(key): to_jsonable(val) for key, val in value.items()}
    if isinstance(value, (list, tuple)):
        return [to_jsonable(item) for item in value]
    return value


def collect_fixture(case: dict[str, Any]) -> dict[str, Any]:
    """Run a Python evaluation and collect the arrays Rust should match."""

    params = dict(case["params"])
    params["data"] = read_voltage_data(DATA_DIR / case["source_file"])
    params["output_dir"] = str(ROOT / "tests" / "output" / "golden")

    evaluation = Evaluation()
    module = evaluation.standard_module(params)

    impedance_input = np.column_stack((module.time, module.impedance))

    fixture = {
        "schema_version": 1,
        "case": case["name"],
        "source_file": case["source_file"],
        "input": {
            "mode": "impedance",
            "data": impedance_input,
        },
        "parameters": {
            "input_mode": "impedance",
            "deconv_mode": module.deconv_mode,
            "struc_method": module.struc_method,
            "precision": module.precision,
            "log_time_size": module.log_time_size,
            "bay_steps": module.bay_steps,
            "pad_factor_pre": module.pad_factor_pre,
            "pad_factor_after": module.pad_factor_after,
            "minimum_window_length": module.minimum_window_length,
            "maximum_window_length": module.maximum_window_length,
            "minimum_window_size": module.minimum_window_size,
            "window_increment": module.window_increment,
            "expected_var": module.expected_var,
            "min_index": module.min_index,
            "timespec_interpolate_factor": module.timespec_interpolate_factor,
            "blockwise_sum_width": module.blockwise_sum_width,
        },
        "reference": {
            "time": module.time,
            "impedance": module.impedance,
            "log_time": module.log_time,
            "log_time_interp": module.log_time_interp,
            "log_time_pad": module.log_time_pad,
            "log_time_delta": module.log_time_delta,
            "imp_smooth": module.imp_smooth,
            "imp_smooth_full": module.imp_smooth_full,
            "imp_deriv_interp": module.imp_deriv_interp,
            "weight": utl.weight_z(module.log_time_pad),
            "time_spec": module.time_spec,
            "sum_time_spec": module.sum_time_spec,
            "crop_log_time": module.crop_log_time,
            "crop_time_spec": module.crop_time_spec,
            "therm_resist_fost": module.therm_resist_fost,
            "therm_capa_fost": module.therm_capa_fost,
            "cau_res": module.cau_res,
            "cau_cap": module.cau_cap,
            "int_cau_res": module.int_cau_res,
            "int_cau_cap": module.int_cau_cap,
            "diff_struc": module.diff_struc,
        },
    }

    return to_jsonable(fixture)


def golden_cases() -> list[dict[str, Any]]:
    """Return the MVP cases for Bayesian deconvolution plus Lanczos Cauer."""

    common = {
        "input_mode": "volt",
        "deconv_mode": "bayesian",
        "bay_steps": 1000,
        "struc_method": "lanczos",
    }

    return [
        {
            "name": "mosfet_tim_bayesian_lanczos",
            "source_file": "MOSFET_tim.txt",
            "params": {
                **common,
                "label": "mosfet_tim_bayesian_lanczos",
                "calib": MOSFET_CALIB_DATA,
                "lower_fit_limit": 5e-4,
                "upper_fit_limit": 1e-3,
            },
        },
        {
            "name": "mosfet_dry_bayesian_lanczos",
            "source_file": "MOSFET_dry.txt",
            "params": {
                **common,
                "label": "mosfet_dry_bayesian_lanczos",
                "calib": MOSFET_CALIB_DATA,
                "lower_fit_limit": 5e-4,
                "upper_fit_limit": 1e-3,
            },
        },
        {
            "name": "led_bayesian_lanczos",
            "source_file": "LED_data.txt",
            "params": {
                **common,
                "label": "led_bayesian_lanczos",
                "calib": LED_CALIB_DATA,
                "lower_fit_limit": 1e-7,
                "upper_fit_limit": 2e-7,
                "log_time_size": 100,
                "minimum_window_length": 0.2,
                "maximum_window_length": 3.0,
                "minimum_window_size": 5,
                "window_increment": 0.05,
                "expected_var": 0.01,
                "min_index": 3,
            },
        },
    ]


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--case",
        action="append",
        choices=[case["name"] for case in golden_cases()],
        help="Generate only the named case. Can be supplied multiple times.",
    )
    args = parser.parse_args()

    selected = {
        case["name"]: case
        for case in golden_cases()
        if args.case is None or case["name"] in args.case
    }

    FIXTURE_DIR.mkdir(parents=True, exist_ok=True)

    for name, case in selected.items():
        fixture = collect_fixture(case)
        output_file = FIXTURE_DIR / f"{name}.json"
        output_file.write_text(
            json.dumps(fixture, indent=2, sort_keys=True, allow_nan=False) + "\n",
            encoding="utf-8",
        )
        print(f"Wrote {output_file.relative_to(ROOT)}")


if __name__ == "__main__":
    main()
