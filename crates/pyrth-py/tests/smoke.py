"""Smoke checks for the experimental PyO3 module.

Run after installing the extension into the active Python environment:

    maturin develop --manifest-path crates/pyrth-py/Cargo.toml
    python crates/pyrth-py/tests/smoke.py
"""

from __future__ import annotations

import math

import pyrth_py


DATA = [(1e-6, 0.1), (1e-5, 0.2), (1e-4, 0.3)]
TEMP_DATA = [(1.0, 20.0), (2.0, 19.0), (3.0, 18.0)]
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
    module = evaluation.standard_module({"data": DATA, "only_make_z": True})
    assert_impedance_only(module)

    temp_module = evaluation.standard_module(
        {
            "data": TEMP_DATA,
            "input_mode": "temp",
            "only_make_z": True,
            "power_step": 2.0,
            "temp_0_avg_range": (0, 1),
        }
    )
    assert temp_module["impedance"] == [0.0, 0.5, 1.0]

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
