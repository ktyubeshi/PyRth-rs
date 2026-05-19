"""Smoke checks for the experimental PyO3 module.

Run after installing the extension into the active Python environment:

    maturin develop --manifest-path crates/pyrth-py/Cargo.toml
    python crates/pyrth-py/tests/smoke.py
"""

from __future__ import annotations

import math
from pathlib import Path

import pyrth_py


REPO_ROOT = Path(__file__).resolve().parents[3]
T3STER_DIR = REPO_ROOT / "tests" / "data" / "t3ster"
DATA = [(1e-6, 0.1), (1e-5, 0.2), (1e-4, 0.3)]
LASSO_DATA = [
    (1e-6, 0.10),
    (3e-6, 0.14),
    (1e-5, 0.20),
    (3e-5, 0.29),
    (1e-4, 0.40),
    (3e-4, 0.54),
]
TEMP_DATA = [(1.0, 20.0), (2.0, 19.0), (3.0, 18.0)]
EXTRAP_TEMP_DATA = [(1.0, 12.0), (4.0, 14.0), (9.0, 16.0), (16.0, 18.0)]
VOLT_DATA = [(1.0, 0.5), (2.0, 0.4), (3.0, 0.3)]
CALIBRATION = [(20.0, 0.5), (30.0, 0.4), (40.0, 0.3)]


def assert_impedance_only(result: dict) -> None:
    assert sorted(result) == ["impedance", "log_time", "time"]
    assert result["time"] == [point[0] for point in DATA]
    assert result["impedance"] == [point[1] for point in DATA]
    assert all(math.isfinite(value) for value in result["log_time"])


def assert_close_list(actual: list[float], expected: list[float], tol: float = 1e-9) -> None:
    assert len(actual) == len(expected)
    for left, right in zip(actual, expected):
        assert abs(left - right) <= tol


def main() -> None:
    result = pyrth_py.evaluate_impedance(DATA, only_make_z=True)
    assert_impedance_only(result)

    evaluation = pyrth_py.Evaluation()
    module = evaluation.standard_module(
        {"data": DATA, "only_make_z": True, "structure_method": "lanczos"}
    )
    assert_impedance_only(module)

    lasso_module = evaluation.standard_module(
        {
            "data": LASSO_DATA,
            "deconv_mode": "lasso",
            "struc_method": "lanczos",
            "calc_struc": False,
            "log_time_size": 4,
            "min_index": 1,
            "minimum_window_size": 2,
            "minimum_window_length": 0.1,
            "maximum_window_length": 0.5,
            "window_increment": 0.2,
            "pad_factor_pre": 0.0,
            "pad_factor_after": 0.0,
            "expected_var": 0.01,
            "timespec_interpolate_factor": 1.0,
            "lasso_alpha": 1e-4,
            "lasso_max_iter": 2,
            "lasso_tol": 1e-3,
            "precision": 64,
        }
    )
    assert "time_spec" in lasso_module
    assert len(lasso_module["time_spec"]) == 4
    assert all(math.isfinite(value) for value in lasso_module["time_spec"])

    temp_module = evaluation.standard_module(
        {
            "data": TEMP_DATA,
            "input_mode": "temp",
            "deconv_mode": "fourier",
            "filter_name": "rectangular",
            "filter_range": 0.6,
            "only_make_z": True,
            "power_step": 2.0,
            "temp_0_avg_range": (0, 1),
        }
    )
    assert temp_module["impedance"] == [0.0, 0.5, 1.0]

    extrapolated_temp_module = evaluation.standard_module(
        {
            "data": EXTRAP_TEMP_DATA,
            "input_mode": "temp",
            "only_make_z": True,
            "extrapolate": True,
            "lower_fit_limit": 4.0,
            "upper_fit_limit": 16.0,
        }
    )
    assert len(extrapolated_temp_module["time"]) == 18
    assert_close_list(extrapolated_temp_module["impedance"][-3:], [-4.0, -6.0, -8.0])

    volt_module = evaluation.standard_module(
        {
            "data": VOLT_DATA,
            "input_mode": "volt",
            "only_make_z": True,
            "calibration": CALIBRATION,
            "kfac_fit_deg": 1,
        }
    )
    assert_close_list(volt_module["impedance"], [0.0, -10.0, -20.0])

    t3ster_module = evaluation.standard_module(
        {
            "input_mode": "t3ster",
            "input": str(T3STER_DIR / "T25_I-m5m-I-h600m_100s.raw"),
            "t3ster_power": str(T3STER_DIR / "T25_I-m5m-I-h600m_100s.pwr"),
            "t3ster_calibration": str(T3STER_DIR / "calib.tco"),
            "only_make_z": True,
        }
    )
    assert sorted(t3ster_module) == ["impedance", "log_time", "time"]
    assert len(t3ster_module["time"]) > 100
    assert t3ster_module["time"][0] == 1e-6
    assert all(math.isfinite(value) for value in t3ster_module["impedance"])

    try:
        evaluation.standard_module(
            {
                "data": DATA,
                "only_make_z": False,
                "log_time_size": 0,
            }
        )
    except ValueError as exc:
        assert "log_time_size" in str(exc)
    else:
        raise AssertionError("invalid log_time_size should raise ValueError")


if __name__ == "__main__":
    main()
