import argparse
import json
import math
import os
import subprocess
import tempfile
import time
from pathlib import Path
from typing import Any


CASES_PUBLIC = [
    (96, 120, 0.18),
    (128, 180, 0.21),
    (192, 220, 0.16),
]
CASES_PRIVATE = [
    (224, 280, 0.19),
    (320, 360, 0.17),
    (384, 440, 0.15),
]
TIMEOUT_SECONDS = 20
REL_TOLERANCE = 1.0e-9
ABS_TOLERANCE = 1.0e-7


def _reference_answer(n: int, steps: int, alpha: float) -> float:
    state = []
    for i in range(n):
        x = i / (n - 1)
        value = math.exp(-80.0 * (x - 0.35) * (x - 0.35))
        value += 0.5 * math.exp(-120.0 * (x - 0.72) * (x - 0.72))
        state.append(value)
    state[0] = 0.0
    state[-1] = 0.0

    for _ in range(steps):
        next_state = [0.0] * n
        for i in range(1, n - 1):
            next_state[i] = state[i] + alpha * (
                state[i - 1] - 2.0 * state[i] + state[i + 1]
            )
        state = next_state

    return sum(value * (idx + 1) for idx, value in enumerate(state))


def _expected_answers(cases: list[tuple[int, int, float]]) -> list[float]:
    return [_reference_answer(n, steps, alpha) for n, steps, alpha in cases]


def _parse_output(output_path: Path) -> list[float]:
    text = output_path.read_text(encoding="utf-8").strip()
    if not text:
        return []
    answers: list[float] = []
    for line in text.splitlines():
        stripped = line.strip()
        if stripped:
            answers.append(float(stripped))
    return answers


def _compile_program(program_path: str, tmpdir_path: Path) -> Path:
    executable_path = tmpdir_path / "candidate"
    completed = subprocess.run(
        [
            "gfortran",
            "-O2",
            str(Path(program_path).resolve()),
            "-o",
            str(executable_path),
        ],
        check=False,
        capture_output=True,
        text=True,
        timeout=TIMEOUT_SECONDS,
        cwd=tmpdir_path,
    )
    if completed.returncode != 0:
        stderr = completed.stderr.strip()
        stdout = completed.stdout.strip()
        error_parts = ["Fortran compilation failed"]
        if stderr:
            error_parts.append(f"stderr: {stderr}")
        if stdout:
            error_parts.append(f"stdout: {stdout}")
        raise RuntimeError(" | ".join(error_parts))
    return executable_path


def _run_program(
    program_path: str,
    cases: list[tuple[int, int, float]],
) -> tuple[list[float], float]:
    with tempfile.TemporaryDirectory() as tmpdir:
        tmpdir_path = Path(tmpdir)
        input_path = tmpdir_path / "cases.txt"
        output_path = tmpdir_path / "answers.txt"
        input_path.write_text(
            "\n".join(f"{n} {steps} {alpha:.17g}" for n, steps, alpha in cases),
            encoding="utf-8",
        )

        executable_path = _compile_program(program_path, tmpdir_path)
        start = time.perf_counter()
        completed = subprocess.run(
            [str(executable_path), str(input_path), str(output_path)],
            check=False,
            capture_output=True,
            text=True,
            timeout=TIMEOUT_SECONDS,
            cwd=tmpdir_path,
        )
        elapsed = time.perf_counter() - start

        if completed.returncode != 0:
            stderr = completed.stderr.strip()
            stdout = completed.stdout.strip()
            error_parts = ["Fortran program failed"]
            if stderr:
                error_parts.append(f"stderr: {stderr}")
            if stdout:
                error_parts.append(f"stdout: {stdout}")
            raise RuntimeError(" | ".join(error_parts))

        if not output_path.exists():
            raise RuntimeError("Fortran program did not produce output file")

        answers = _parse_output(output_path)
        return answers, elapsed


def _is_close(expected: float, predicted: float) -> bool:
    return math.isclose(
        predicted,
        expected,
        rel_tol=REL_TOLERANCE,
        abs_tol=ABS_TOLERANCE,
    )


def _build_metrics(
    cases: list[tuple[int, int, float]],
    expected: list[float],
    predicted: list[float],
    runtime_seconds: float,
) -> tuple[dict[str, Any], bool, str]:
    if len(predicted) != len(expected):
        msg = (
            f"Output length mismatch. Expected {len(expected)} answers, "
            f"got {len(predicted)}."
        )
        metrics = {
            "combined_score": 0.0,
            "public": {
                "runtime_seconds": runtime_seconds,
                "num_cases": len(cases),
                "num_answers": len(predicted),
                "accuracy": 0.0,
            },
            "private": {},
        }
        return metrics, False, msg

    mismatches: list[dict[str, Any]] = []
    max_abs_error = 0.0
    for idx, (case, exp, pred) in enumerate(zip(cases, expected, predicted)):
        abs_error = abs(pred - exp)
        max_abs_error = max(max_abs_error, abs_error)
        if not _is_close(exp, pred):
            mismatches.append(
                {
                    "index": idx,
                    "case": {
                        "n": case[0],
                        "steps": case[1],
                        "alpha": case[2],
                    },
                    "expected": exp,
                    "predicted": pred,
                    "abs_error": abs_error,
                }
            )

    num_correct = len(cases) - len(mismatches)
    accuracy = num_correct / len(cases)
    all_correct = len(mismatches) == 0
    combined_score = max(0.0, accuracy * 100.0 - runtime_seconds)

    metrics = {
        "combined_score": combined_score,
        "public": {
            "runtime_seconds": runtime_seconds,
            "num_cases": len(cases),
            "num_correct": num_correct,
            "accuracy": accuracy,
            "max_abs_error": max_abs_error,
        },
        "private": {
            "mismatch_count": len(mismatches),
            "first_mismatch": mismatches[0] if mismatches else None,
        },
    }

    if all_correct:
        return metrics, True, ""
    msg = f"{len(mismatches)} mismatches found. First mismatch: {mismatches[0]}"
    return metrics, False, msg


def main(program_path: str, results_dir: str):
    os.makedirs(results_dir, exist_ok=True)
    all_cases = CASES_PUBLIC + CASES_PRIVATE
    expected = _expected_answers(all_cases)

    try:
        predicted, runtime_seconds = _run_program(program_path, all_cases)
        metrics, correct, error = _build_metrics(
            cases=all_cases,
            expected=expected,
            predicted=predicted,
            runtime_seconds=runtime_seconds,
        )
    except FileNotFoundError:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": 0.0, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = (
            "gfortran executable not found. Install gfortran and make it available "
            "in PATH."
        )
    except subprocess.TimeoutExpired:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": TIMEOUT_SECONDS, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = f"Fortran evaluation timed out after {TIMEOUT_SECONDS} seconds."
    except Exception as exc:
        metrics = {
            "combined_score": 0.0,
            "public": {"runtime_seconds": 0.0, "accuracy": 0.0},
            "private": {},
        }
        correct = False
        error = str(exc)

    correct_path = Path(results_dir) / "correct.json"
    metrics_path = Path(results_dir) / "metrics.json"
    correct_path.write_text(
        json.dumps({"correct": correct, "error": error}, indent=4), encoding="utf-8"
    )
    metrics_path.write_text(json.dumps(metrics, indent=4), encoding="utf-8")

    print(f"Evaluated program: {program_path}")
    print(f"Results saved to: {results_dir}")
    print(f"Correct: {correct}")
    if error:
        print(f"Error: {error}")
    print(f"Combined score: {metrics.get('combined_score', 0.0):.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Evaluate Fortran heat-diffusion stencil program"
    )
    parser.add_argument(
        "--program_path",
        type=str,
        default="initial.f90",
        help="Path to candidate Fortran program",
    )
    parser.add_argument(
        "--results_dir",
        type=str,
        default="results",
        help="Directory where metrics.json and correct.json are written",
    )
    args = parser.parse_args()
    main(args.program_path, args.results_dir)
