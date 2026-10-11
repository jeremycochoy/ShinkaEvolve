# Fortran Heat Diffusion

Runnable non-Python candidate example using a Fortran seed program and Python
evaluator.

## Requirements

- Python 3.10+
- `gfortran` on `PATH`
- ShinkaEvolve installed from the repository root

## Files

- `initial.f90`: seed Fortran implementation with `! EVOLVE-BLOCK` markers.
- `evaluate.py`: compiles candidates with `gfortran`, runs fixed heat-diffusion
  cases, and writes `metrics.json` plus `correct.json`.
- `run_evo.py`: small async Shinka run config with `language="fortran"`.

## Manual Evaluation

```bash
cd examples/fortran_heat_diffusion
python evaluate.py --program_path initial.f90 --results_dir results/manual_eval
```

## Run Evolution

```bash
cd examples/fortran_heat_diffusion
python run_evo.py
```

The evaluator checks one-dimensional heat diffusion checksums against a Python
reference implementation with strict floating-point tolerances. The combined
score prioritizes correctness, then subtracts runtime so faster correct programs
rank higher.
