from __future__ import annotations

import argparse
import importlib.util
import json
import math
from pathlib import Path


BLOCKED_SOURCE_SNIPPETS = (
    "math.sin",
    "cmath.sin",
    "numpy.sin",
    "np.sin",
    "from math import sin",
    "import sin",
)


def _load_program(program_path: str):
    spec = importlib.util.spec_from_file_location("program", program_path)
    if spec is None or spec.loader is None:
        raise ImportError(f"Could not load program: {program_path}")
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def _sample_points() -> list[float]:
    points = []
    for index in range(161):
        base = -math.pi + (2.0 * math.pi * index / 160.0)
        wobble = 0.007 * math.sin(index * 1.61803398875)
        points.append(max(-math.pi, min(math.pi, base + wobble)))
    return points


def _source_violation(source: str) -> str | None:
    lowered = source.lower()
    for blocked in BLOCKED_SOURCE_SNIPPETS:
        if blocked in lowered:
            return f"blocked direct sine implementation: {blocked}"
    return None


def _evaluate(program_path: str) -> tuple[dict, bool, str]:
    source = Path(program_path).read_text(encoding="utf-8")
    violation = _source_violation(source)
    if violation:
        return _failure_metrics(violation), False, violation

    module = _load_program(program_path)
    approximate = getattr(module, "approximate", None)
    if not callable(approximate):
        return (
            _failure_metrics("missing callable approximate(x)"),
            False,
            "missing callable approximate(x)",
        )

    abs_errors = []
    for x in _sample_points():
        try:
            estimate = float(approximate(x))
        except Exception as exc:  # noqa: BLE001
            return _failure_metrics(str(exc)), False, str(exc)
        if not math.isfinite(estimate) or abs(estimate) > 10.0:
            return (
                _failure_metrics("non-finite or out-of-range output"),
                False,
                "non-finite or out-of-range output",
            )
        abs_errors.append(abs(estimate - math.sin(x)))

    mae = sum(abs_errors) / len(abs_errors)
    rmse = math.sqrt(sum(error * error for error in abs_errors) / len(abs_errors))
    max_error = max(abs_errors)
    score = 1.0 / (1.0 + 4.0 * rmse + max_error)
    return (
        {
            "combined_score": score,
            "public": {
                "score": score,
                "mean_abs_error": mae,
                "rmse": rmse,
                "max_abs_error": max_error,
            },
            "private": {
                "num_points": len(abs_errors),
            },
        },
        True,
        "",
    )


def _failure_metrics(error: str) -> dict:
    return {
        "combined_score": 0.0,
        "public": {
            "score": 0.0,
            "mean_abs_error": 1e9,
            "rmse": 1e9,
            "max_abs_error": 1e9,
        },
        "private": {"error": error},
    }


def main(program_path: str, results_dir: str):
    metrics, correct, error = _evaluate(program_path)
    results_path = Path(results_dir)
    results_path.mkdir(parents=True, exist_ok=True)
    (results_path / "metrics.json").write_text(
        json.dumps(metrics, indent=2),
        encoding="utf-8",
    )
    (results_path / "correct.json").write_text(
        json.dumps({"correct": correct, "error": error}, indent=2),
        encoding="utf-8",
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--program_path", required=True)
    parser.add_argument("--results_dir", required=True)
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
