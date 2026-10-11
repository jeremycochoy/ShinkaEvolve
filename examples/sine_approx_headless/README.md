# Headless Math Approximation Example

This task uses a subscription-backed Headless CLI agent for Shinka mutations.
The evolved program approximates `sin(x)` on `[-pi, pi]` without calling a standard sine implementation. The evaluator reports bounded approximation metrics and disables embeddings, so no provider API key is needed for mutation calls.

```bash
shinka_run \
  --task-dir examples/sine_approx_headless \
  --results_dir results/sine_approx_headless \
  --num_generations 5 \
  --max-evaluation-jobs 1 \
  --max-proposal-jobs 1 \
  --set evo.llm_models='["headless/codex@gpt-5.5?effort=high"]' \
  --set evo.embedding_model=null \
  --set evo.patch_types='["full", "diff"]' \
  --set evo.patch_type_probs='[0.5, 0.5]'
```

Before the run starts, Shinka executes the configured Headless command with
`--check`. Each mutation prompt is saved under the run directory in
`headless_prompts/` and copied into the matching `gen_*/attempts/...` folder.

The Python runner variant uses both Codex and Claude through Headless, runs for
10 target generations, and uses concurrency of 2:

```bash
python examples/sine_approx_headless/run_evo.py
```
