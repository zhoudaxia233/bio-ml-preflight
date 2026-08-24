---
name: bio-ml-preflight
description: Use when asked to audit whether biological or scientific tabular data supports a proposed prediction, generalization, or ranking claim; discover testable tasks; interpret capability reports; or recommend the cheapest next evidence.
---

# Bio ML preflight

1. Locate the supplied YAML case. If no claim exists, inspect the table with `discover`; do not invent scientific semantics.
2. Validate the case and call out every unconfirmed role.
3. Run deterministic inspection before modeling. Read `references/evidence-rules.md`.
4. Identify missing high-value metadata, especially independent units, replicates, batches, time, treatment assignment, and deployment groups.
5. Run the requested `smoke` or `standard` budget. Do not substitute a larger search.
6. Read `report.json` and the Parquet tables before explaining `report.md`. Python artifacts are the source of numerical truth. The LLM may organize and explain evidence but may not manufacture it.
7. Explain scenario-specific capability boundaries conservatively. Separate exploratory development findings from confirmatory holdout evidence.
8. State `NOT_ASSESSABLE` when required evidence is absent. Missing biological metadata must not be replaced by assumptions.
9. Recommend the cheapest next data or validation that addresses the limiting boundary; do not automatically recommend a more complex model.

Never claim a causal effect from a predictive result alone. Never describe random-split success as proof of unseen-entity, future-time, or deployment generalization.

