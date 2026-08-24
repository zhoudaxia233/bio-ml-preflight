# Constrained Autoprobe

Autoprobe borrows the small mutable surface and experiment log idea from autoresearch, but scientific evaluation cannot be optimized by one scalar. A candidate emits primary development performance, worst-scenario performance, split dispersion, ranking stability, runtime, complexity, and guardrail failures.

Selection is lexicographic: validity guardrails pass first; the primary metric clears a delta; worst-scenario and ranking evidence do not materially regress; runtime stays within budget; simplicity resolves ties. The evaluator and manifests are invariant, negative experiments remain in JSONL, and the final holdout is inaccessible.

This is a scaffold for two or three bounded alternatives, not an autonomous open-ended search.

