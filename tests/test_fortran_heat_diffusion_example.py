import importlib.util
import subprocess
from pathlib import Path


def _load_evaluator():
    repo_root = Path(__file__).resolve().parents[1]
    module_path = repo_root / "examples" / "fortran_heat_diffusion" / "evaluate.py"
    spec = importlib.util.spec_from_file_location("fortran_heat_eval", module_path)
    assert spec is not None
    module = importlib.util.module_from_spec(spec)
    assert spec.loader is not None
    spec.loader.exec_module(module)
    return module


def test_fortran_evaluator_runs_candidate_from_tempdir(monkeypatch, tmp_path):
    evaluator = _load_evaluator()
    run_calls = []

    def fake_compile_program(program_path, tmpdir_path):
        executable_path = tmpdir_path / "candidate"
        executable_path.write_text("", encoding="utf-8")
        return executable_path

    def fake_run(args, **kwargs):
        assert kwargs["cwd"] == Path(args[1]).parent
        run_calls.append((args, kwargs))
        output_path = Path(args[2])
        output_path.write_text("1.0\n", encoding="utf-8")
        return subprocess.CompletedProcess(args=args, returncode=0, stdout="", stderr="")

    monkeypatch.setattr(evaluator, "_compile_program", fake_compile_program)
    monkeypatch.setattr(evaluator.subprocess, "run", fake_run)

    answers, _ = evaluator._run_program(str(tmp_path / "candidate.f90"), [(8, 1, 0.1)])

    assert answers == [1.0]
    assert run_calls
