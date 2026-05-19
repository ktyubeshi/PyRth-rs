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

    standard = evaluation.standard(
        {"data": DATA, "only_make_z": True, "structure_method": "lanczos"}
    )
    assert_impedance_only(standard)

    module_set = evaluation.standard_module_set(
        {"data": DATA, "only_make_z": True, "structure_method": "lanczos"}
    )
    assert_impedance_only(module_set)

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

    theoretical = pyrth_py.theoretical_impedance([1.0, 2.0], [0.5, 1.5], 1e-6, 1e-2, 8)
    assert sorted(theoretical) == ["impedance", "time"]
    assert len(theoretical["time"]) == 8
    assert all(math.isfinite(value) for value in theoretical["impedance"])

    theoretical_facade = evaluation.theoretical(
        {
            "resistance": [1.0, 2.0],
            "capacitance": [0.5, 1.5],
            "time_start": 1e-6,
            "time_end": 1e-2,
            "time_size": 8,
        }
    )
    assert sorted(theoretical_facade) == ["impedance", "time"]
    assert len(theoretical_facade["time"]) == 8
    assert all(math.isfinite(value) for value in theoretical_facade["impedance"])

    bootstrap = pyrth_py.bootstrap_theoretical(
        [1.0, 2.0],
        [0.5, 1.5],
        1e-6,
        1e-1,
        80,
        2,
        0.0,
        seed=7,
    )
    assert bootstrap["successful_repetitions"] == 2
    assert len(bootstrap["impedance_mean"]) == 80
    assert len(bootstrap["time_spectrum_mean"]) > 0
    assert all(math.isfinite(value) for value in bootstrap["time_spectrum_mean"])

    bootstrap_facade = evaluation.bootstrap(
        {
            "resistance": [1.0, 2.0],
            "capacitance": [0.5, 1.5],
            "time_start": 1e-6,
            "time_end": 1e-1,
            "time_size": 80,
            "repetitions": 2,
            "noise_std": 0.0,
            "seed": 7,
        }
    )
    assert bootstrap_facade["successful_repetitions"] == 2
    assert len(bootstrap_facade["impedance_mean"]) == 80
    assert len(bootstrap_facade["time_spectrum_mean"]) > 0
    assert all(math.isfinite(value) for value in bootstrap_facade["time_spectrum_mean"])

    target = pyrth_py.theoretical_impedance([1.0, 3.0], [0.4, 2.0], 1e-3, 1e2, 32)
    optimized = pyrth_py.optimize_rc(
        list(zip(target["time"], target["impedance"])),
        [0.75, 3.5],
        [0.65, 1.5],
        [0.5, 2.0],
        [0.2, 1.0],
        [1.5, 4.0],
        [1.0, 3.0],
        max_iter=8,
        initial_step=0.25,
        min_step=1e-3,
    )
    assert sorted(optimized) == ["capacitance", "iterations", "residual_norm", "resistance"]
    assert len(optimized["resistance"]) == 2
    assert len(optimized["capacitance"]) == 2
    assert optimized["iterations"] > 0
    assert math.isfinite(optimized["residual_norm"])

    optimized_facade = evaluation.optimization(
        {
            "data": list(zip(target["time"], target["impedance"])),
            "initial_resistance": [0.75, 3.5],
            "initial_capacitance": [0.65, 1.5],
            "lower_resistance": [0.5, 2.0],
            "lower_capacitance": [0.2, 1.0],
            "upper_resistance": [1.5, 4.0],
            "upper_capacitance": [1.0, 3.0],
            "max_iter": 8,
            "initial_step": 0.25,
            "min_step": 1e-3,
        }
    )
    assert sorted(optimized_facade) == [
        "capacitance",
        "iterations",
        "residual_norm",
        "resistance",
    ]
    assert len(optimized_facade["resistance"]) == 2
    assert len(optimized_facade["capacitance"]) == 2
    assert optimized_facade["iterations"] > 0
    assert math.isfinite(optimized_facade["residual_norm"])

    prediction_input = pyrth_py.theoretical_impedance([1.0], [0.5], 1e-6, 1e-2, 80)
    predicted = pyrth_py.predict_temperature_response(
        list(zip(prediction_input["time"], prediction_input["impedance"])),
        [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
        lin_sampling_period=1e-3,
    )
    assert sorted(predicted) == ["temperature", "time"]
    assert len(predicted["time"]) == len(predicted["temperature"])
    assert len(predicted["time"]) > 0
    assert all(math.isfinite(value) for value in predicted["temperature"])

    predicted_facade = evaluation.temperature_prediction(
        {
            "impulse_response": list(
                zip(prediction_input["time"], prediction_input["impedance"])
            ),
            "power_data": [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
            "lin_sampling_period": 1e-3,
        }
    )
    assert sorted(predicted_facade) == ["temperature", "time"]
    assert len(predicted_facade["time"]) == len(predicted_facade["temperature"])
    assert len(predicted_facade["time"]) > 0
    assert all(math.isfinite(value) for value in predicted_facade["temperature"])

    comparison = evaluation.comparison(lasso_module, lasso_module)
    assert sorted(comparison) == [
        "structure_norm",
        "time_const_norm",
        "total_resistance_diff",
    ]
    assert comparison["time_const_norm"] == 0.0
    assert comparison["structure_norm"] == 0.0
    assert comparison["total_resistance_diff"] == 0.0

    comparison_facade = evaluation.comparison(
        {"reference": lasso_module, "candidate": lasso_module}
    )
    assert comparison_facade == comparison

    try:
        evaluation.comparison({"reference": lasso_module})
    except ValueError as exc:
        assert "reference and candidate" in str(exc)
        assert "candidate" in str(exc)
    else:
        raise AssertionError("comparison without candidate should raise ValueError")

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
