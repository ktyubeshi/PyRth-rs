"""Smoke checks for the experimental PyO3 module.

Run after installing the extension into the active Python environment:

    maturin develop --manifest-path crates/pyrth-py/Cargo.toml
    python crates/pyrth-py/tests/smoke.py
"""

from __future__ import annotations

import math
import shutil
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

    swept_modules = evaluation.standard_module_set(
        {
            "data": DATA,
            "only_make_z": True,
            "structure_method": "lanczos",
            "label": "sweep",
            "evaluation_type": "standard",
            "iterable_keywords": ["power_step"],
            "power_step": [1.0, 2.0],
        }
    )
    assert len(swept_modules) == 2
    assert_impedance_only(swept_modules[0])
    assert_impedance_only(swept_modules[1])
    labels_after_sweep = evaluation.module_labels()
    assert "sweep_power_step_0" in labels_after_sweep
    assert "sweep_power_step_1" in labels_after_sweep

    comparison_sweep = evaluation.comparison_module(
        {
            "resistance": [1.0],
            "capacitance": [0.5],
            "time_start": 1e-6,
            "time_end": 1e-2,
            "time_size": 32,
            "label": "comparison_sweep",
            "evaluation_type": "standard",
            "iterable_keywords": ["bay_steps"],
            "bay_steps": [2, 3],
            "log_time_size": 8,
            "min_index": 1,
            "minimum_window_size": 2,
            "calc_struc": False,
        }
    )
    assert sorted(comparison_sweep) == [
        "mod_key_display_name",
        "mod_value_list",
        "structure_comparison",
        "time_const_comparison",
        "total_resist_diff",
    ]
    assert comparison_sweep["mod_key_display_name"] == "bay_steps"
    assert comparison_sweep["mod_value_list"] == [2, 3]
    assert len(comparison_sweep["time_const_comparison"]) == 2
    assert comparison_sweep["time_const_comparison"][0] == 0.0
    assert all(math.isfinite(value) for value in comparison_sweep["time_const_comparison"])

    labeled = evaluation.standard_module(
        {
            "data": DATA,
            "only_make_z": True,
            "structure_method": "lanczos",
            "label": "smoke",
        }
    )
    assert_impedance_only(labeled)
    labeled_again = evaluation.standard_module(
        {
            "data": DATA,
            "only_make_z": True,
            "structure_method": "lanczos",
            "label": "smoke",
        }
    )
    assert_impedance_only(labeled_again)
    labels = evaluation.module_labels()
    assert "smoke" in labels
    assert "smoke_1" in labels
    assert evaluation.module_count() == len(labels)
    smoke_object = evaluation.module("smoke")
    assert smoke_object.label == "smoke"
    assert smoke_object.time == [point[0] for point in DATA]
    assert smoke_object.impedance == [point[1] for point in DATA]
    assert "impedance" in smoke_object.data_handlers
    assert sorted(smoke_object.keys()) == ["impedance", "log_time", "time"]
    assert smoke_object["impedance"] == smoke_object.impedance
    assert smoke_object.get("missing", "fallback") == "fallback"
    assert "time" in smoke_object
    assert smoke_object.to_dict()["impedance"] == smoke_object.impedance
    try:
        smoke_object["missing"]
    except KeyError:
        pass
    else:
        raise AssertionError("missing StructureFunction key should raise KeyError")

    csv_dir = REPO_ROOT / "target" / "tmp" / "pyrth-py-smoke-csv"
    shutil.rmtree(csv_dir, ignore_errors=True)
    saved = evaluation.save_as_csv(str(csv_dir))
    assert "no_label" in saved
    assert "sweep_power_step_0" in saved
    assert "sweep_power_step_1" in saved
    assert sorted(saved["no_label"]) == ["impedance"]
    assert Path(saved["no_label"]["impedance"]).exists()
    assert (
        Path(saved["no_label"]["impedance"]).read_text().splitlines()[0]
        == "time,impedance"
    )
    assert Path(saved["sweep_power_step_0"]["impedance"]).exists()
    assert Path(saved["sweep_power_step_1"]["impedance"]).exists()

    figure_dir = REPO_ROOT / "target" / "tmp" / "pyrth-py-smoke-figures"
    shutil.rmtree(figure_dir, ignore_errors=True)
    figures = evaluation.save_figures(str(figure_dir))
    assert sorted(figures["no_label"]) == ["impedance"]
    impedance_svg = Path(figures["no_label"]["impedance"])
    assert impedance_svg.exists()
    assert impedance_svg.read_text().startswith("<svg ")

    all_dir = REPO_ROOT / "target" / "tmp" / "pyrth-py-smoke-all"
    shutil.rmtree(all_dir, ignore_errors=True)
    saved_all = evaluation.save_all(str(all_dir))
    assert Path(saved_all["csv"]["no_label"]["impedance"]).exists()
    assert Path(saved_all["figures"]["no_label"]["impedance"]).exists()

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
            "label": "lasso_object",
        }
    )
    assert "time_spec" in lasso_module
    assert len(lasso_module["time_spec"]) == 4
    assert all(math.isfinite(value) for value in lasso_module["time_spec"])
    lasso_object = evaluation.module("lasso_object")
    assert "time_spec" in lasso_object
    assert lasso_object["time_spec"] == lasso_module["time_spec"]
    assert "time_spec" in lasso_object.keys()
    assert lasso_object.imp_deriv_interp == lasso_module["imp_deriv_interp"]
    assert lasso_object.log_time_interp == lasso_module["log_time_interp"]
    assert lasso_object.therm_resist_fost == lasso_module["therm_resist_fost"]
    assert lasso_object.therm_capa_fost == lasso_module["therm_capa_fost"]
    assert lasso_object.cau_res is None
    assert "therm_resist_fost" in lasso_object.keys()

    foster = pyrth_py.foster_step_response([1.0, 2.0], [0.5, 1.5], 1e-6, 1e-2, 8)
    assert sorted(foster) == ["impedance", "time"]
    assert len(foster["time"]) == 8
    assert all(math.isfinite(value) for value in foster["impedance"])

    theoretical_facade = evaluation.theoretical(
        {
            "theo_resistances": [1.0, 2.0],
            "theo_capacitances": [0.5, 1.5],
            "theo_time": [1e-6, 1e-2],
            "theo_time_size": 8,
        }
    )
    assert "theo_time_const" in theoretical_facade
    assert "theo_imp_deriv" in theoretical_facade
    assert "theo_impedance" in theoretical_facade
    assert len(theoretical_facade["time"]) == 8
    assert len(theoretical_facade["theo_time_const"]) == 8
    assert len(theoretical_facade["theo_imp_deriv"]) == 8
    assert len(theoretical_facade["theo_impedance"]) == 8
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
    assert len(bootstrap["impedance_p10"]) == 80
    assert len(bootstrap["impedance_median"]) == 80
    assert len(bootstrap["impedance_p90"]) == 80
    assert len(bootstrap["time_spectrum_mean"]) > 0
    assert len(bootstrap["time_spectrum_p10"]) == len(bootstrap["time_spectrum_mean"])
    assert len(bootstrap["time_spectrum_median"]) == len(bootstrap["time_spectrum_mean"])
    assert len(bootstrap["time_spectrum_p90"]) == len(bootstrap["time_spectrum_mean"])
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
    assert len(bootstrap_facade["impedance_median"]) == 80
    assert len(bootstrap_facade["time_spectrum_mean"]) > 0
    assert len(bootstrap_facade["time_spectrum_median"]) == len(
        bootstrap_facade["time_spectrum_mean"]
    )
    assert all(math.isfinite(value) for value in bootstrap_facade["time_spectrum_mean"])

    bootstrap_python_aliases = evaluation.bootstrap(
        {
            "theo_resistances": [1.0, 2.0],
            "theo_capacitances": [0.5, 1.5],
            "theo_time": [1e-6, 1e-1],
            "theo_time_size": 80,
            "repetitions": 2,
            "signal_to_noise_ratio": 1e12,
            "random_seed": 7,
        }
    )
    assert bootstrap_python_aliases["successful_repetitions"] == 2
    assert len(bootstrap_python_aliases["impedance_median"]) == 80
    assert len(bootstrap_python_aliases["time_spectrum_median"]) > 0

    bootstrap_from_data = evaluation.bootstrap(
        {
            "data": list(zip(theoretical["time"], theoretical["impedance"])),
            "repetitions": 2,
            "noise_std": 0.0,
            "random_seed": 7,
            "deconv_mode": "fourier",
            "log_time_size": 8,
            "min_index": 1,
            "minimum_window_size": 2,
            "calc_struc": False,
        }
    )
    assert bootstrap_from_data["successful_repetitions"] == 2
    assert len(bootstrap_from_data["impedance_median"]) == len(theoretical["time"])
    assert len(bootstrap_from_data["time_spectrum_median"]) == 8

    bootstrap_comparison = evaluation.comparison_module(
        {
            "data": list(zip(theoretical["time"], theoretical["impedance"])),
            "evaluation_type": "bootstrap",
            "label": "bootstrap_comparison",
            "iterable_keywords": ["noise_std"],
            "noise_std": [0.0, 1e-12],
            "repetitions": 2,
            "random_seed": 7,
            "deconv_mode": "fourier",
            "log_time_size": 8,
            "min_index": 1,
            "minimum_window_size": 2,
            "calc_struc": False,
        }
    )
    assert sorted(bootstrap_comparison) == [
        "mod_key_display_name",
        "mod_value_list",
        "structure_comparison",
        "time_const_comparison",
        "total_resist_diff",
    ]
    assert bootstrap_comparison["mod_key_display_name"] == "noise_std"
    assert bootstrap_comparison["mod_value_list"] == [0.0, 1e-12]
    assert bootstrap_comparison["time_const_comparison"][0] == 0.0
    assert bootstrap_comparison["structure_comparison"] == [0.0, 0.0]
    assert all(
        math.isfinite(value) for value in bootstrap_comparison["time_const_comparison"]
    )
    assert all(math.isfinite(value) for value in bootstrap_comparison["total_resist_diff"])

    target = pyrth_py.foster_step_response([1.0, 3.0], [0.4, 2.0], 1e-3, 1e2, 32)
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

    optimization_comparison = evaluation.comparison_module(
        {
            "data": list(zip(target["time"], target["impedance"])),
            "evaluation_type": "optimization",
            "label": "optimization_comparison",
            "iterable_keywords": ["initial_step"],
            "initial_step": [0.25, 0.125],
            "initial_resistance": [0.75, 3.5],
            "initial_capacitance": [0.65, 1.5],
            "lower_resistance": [0.5, 2.0],
            "lower_capacitance": [0.2, 1.0],
            "upper_resistance": [1.5, 4.0],
            "upper_capacitance": [1.0, 3.0],
            "max_iter": 4,
            "min_step": 1e-3,
        }
    )
    assert sorted(optimization_comparison) == [
        "mod_key_display_name",
        "mod_value_list",
        "structure_comparison",
        "time_const_comparison",
        "total_resist_diff",
    ]
    assert optimization_comparison["mod_key_display_name"] == "initial_step"
    assert optimization_comparison["mod_value_list"] == [0.25, 0.125]
    assert optimization_comparison["time_const_comparison"][0] == 0.0
    assert optimization_comparison["structure_comparison"] == [0.0, 0.0]
    assert all(
        math.isfinite(value)
        for value in optimization_comparison["time_const_comparison"]
    )
    assert all(
        math.isfinite(value) for value in optimization_comparison["total_resist_diff"]
    )

    prediction_input = pyrth_py.foster_step_response([1.0], [0.5], 1e-6, 1e-2, 80)
    predicted = pyrth_py.predict_temperature_response(
        list(zip(prediction_input["time"], prediction_input["impedance"])),
        [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
        lin_sampling_period=1e-3,
    )
    assert sorted(predicted) == ["temperature", "time"]
    assert len(predicted["time"]) == len(predicted["temperature"])
    assert len(predicted["time"]) > 0
    assert all(math.isfinite(value) for value in predicted["temperature"])

    labels_before_prediction = evaluation.module_labels()
    count_before_prediction = evaluation.module_count()
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
    assert evaluation.module_count() == count_before_prediction
    assert evaluation.module_labels() == labels_before_prediction

    predicted_labeled = evaluation.temperature_prediction(
        {
            "impulse_response": list(
                zip(prediction_input["time"], prediction_input["impedance"])
            ),
            "power_data": [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
            "lin_sampling_period": 1e-3,
            "label": "temperature_prediction_smoke",
        }
    )
    assert sorted(predicted_labeled) == ["temperature", "time"]
    assert_close_list(predicted_labeled["time"], predicted_facade["time"])
    assert_close_list(predicted_labeled["temperature"], predicted_facade["temperature"])
    labels_after_prediction = evaluation.module_labels()
    assert "temperature_prediction_smoke" in labels_after_prediction
    assert evaluation.module_count() == count_before_prediction + 1
    assert evaluation.module_count() == len(labels_after_prediction)

    prediction_csv_dir = REPO_ROOT / "target" / "tmp" / "pyrth-py-smoke-prediction-csv"
    shutil.rmtree(prediction_csv_dir, ignore_errors=True)
    prediction_csv = evaluation.save_as_csv(str(prediction_csv_dir))
    prediction_csv_path = Path(
        prediction_csv["temperature_prediction_smoke"]["temperature_prediction"]
    )
    assert prediction_csv_path.exists()
    assert prediction_csv_path.read_text().splitlines()[0] == "time,temperature"

    prediction_figure_dir = (
        REPO_ROOT / "target" / "tmp" / "pyrth-py-smoke-prediction-figures"
    )
    shutil.rmtree(prediction_figure_dir, ignore_errors=True)
    prediction_figures = evaluation.save_figures(str(prediction_figure_dir))
    prediction_svg_path = Path(
        prediction_figures["temperature_prediction_smoke"]["temperature_prediction"]
    )
    assert prediction_svg_path.exists()
    prediction_svg = prediction_svg_path.read_text()
    assert prediction_svg.startswith("<svg ")
    assert "Temperature prediction" in prediction_svg

    predicted_from_rc = evaluation.temperature_prediction(
        {
            "resistance": [1.0],
            "capacitance": [0.5],
            "reference_time": prediction_input["time"],
            "power_data": [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
            "lin_sampling_period": 1e-3,
        }
    )
    assert sorted(predicted_from_rc) == ["temperature", "time"]
    assert len(predicted_from_rc["time"]) == len(predicted_from_rc["temperature"])
    assert len(predicted_from_rc["time"]) > 0
    assert all(math.isfinite(value) for value in predicted_from_rc["temperature"])

    predicted_from_optimization = pyrth_py.predict_temperature_response(
        {
            "optimization_result": {"resistance": [1.0], "capacitance": [0.5]},
            "reference_time": prediction_input["time"],
            "power_data": [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
            "lin_sampling_period": 1e-3,
        }
    )
    assert sorted(predicted_from_optimization) == ["temperature", "time"]
    assert_close_list(predicted_from_optimization["time"], predicted_from_rc["time"])
    assert_close_list(
        predicted_from_optimization["temperature"], predicted_from_rc["temperature"]
    )

    predicted_from_internal_optimization = evaluation.temperature_prediction(
        {
            "data": list(zip(prediction_input["time"], prediction_input["impedance"])),
            "initial_resistance": [1.0],
            "initial_capacitance": [0.5],
            "lower_resistance": [1.0],
            "lower_capacitance": [0.5],
            "upper_resistance": [1.0],
            "upper_capacitance": [0.5],
            "max_iter": 1,
            "initial_step": 0.1,
            "min_step": 0.1,
            "reference_time": prediction_input["time"],
            "power_data": [(0.0, 0.0), (0.02, 1.0), (0.04, 0.5)],
            "lin_sampling_period": 1e-3,
        }
    )
    assert_close_list(
        predicted_from_internal_optimization["time"], predicted_from_rc["time"]
    )
    assert_close_list(
        predicted_from_internal_optimization["temperature"],
        predicted_from_rc["temperature"],
    )

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

    comparison_object = evaluation.comparison(lasso_object, lasso_module)
    assert comparison_object == comparison

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

    volt_module_with_calib_alias = evaluation.standard_module(
        {
            "data": VOLT_DATA,
            "input_mode": "volt",
            "only_make_z": True,
            "calib": CALIBRATION,
            "kfac_fit_deg": 1,
        }
    )
    assert_close_list(volt_module_with_calib_alias["impedance"], volt_module["impedance"])

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
