# Autoprobe experiment program

You are running a bounded scientific baseline experiment.

1. Read `policy.json`, `case.yaml`, the baseline report JSON, and the experiment log.
2. Edit only `candidate.yaml`. Never edit the case, evaluator, metrics, split manifests, holdout policy, or provenance code.
3. Run the baseline before alternatives. Do not install dependencies during a run.
4. Evaluate at most three candidates and stop after two consecutive non-improving candidates.
5. Run `uv run bio-ml-preflight autoprobe evaluate RUN_DIR/candidate.yaml` once per candidate.
6. Preserve every result in `experiments.jsonl`, including errors and negative outcomes.
7. Never access final holdout labels. Do not weaken leakage or validity guardrails.
8. Compare the full vector: primary metric, worst scenario, dispersion, top-k stability, runtime, complexity, and guardrail failures. Do not invent a universal score.
9. Stop when the fixed count or non-improvement limit is reached. Never enter an infinite loop.

