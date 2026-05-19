"""Smoke checks for the experimental PyO3 module.

Run after installing the extension into the active Python environment:

    maturin develop --manifest-path crates/pyrth-py/Cargo.toml
    python crates/pyrth-py/tests/smoke.py
"""

from __future__ import annotations

import math

import pyrth_py


DATA = [(1e-6, 0.1), (1e-5, 0.2), (1e-4, 0.3)]


def assert_impedance_only(result: dict) -> None:
    assert sorted(result) == ["impedance", "log_time", "time"]
    assert result["time"] == [point[0] for point in DATA]
    assert result["impedance"] == [point[1] for point in DATA]
    assert all(math.isfinite(value) for value in result["log_time"])


def main() -> None:
    result = pyrth_py.evaluate_impedance(DATA, only_make_z=True)
    assert_impedance_only(result)

    evaluation = pyrth_py.Evaluation()
    module = evaluation.standard_module({"data": DATA, "only_make_z": True})
    assert_impedance_only(module)

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
