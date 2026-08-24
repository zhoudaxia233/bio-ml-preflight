# Limitations

- Retrospective finite data cannot prove future success.
- Public benchmarks are pseudo-sealed; community knowledge weakens blindness.
- Character hashes, Morgan fingerprints, and amino-acid composition are inexpensive probes, not state-of-the-art chemical or protein representations.
- Near-duplicate audits require an explicit scientific similarity function; v0.1 never guesses one.
- Replicate dispersion is descriptive and cannot become a defensible noise ceiling without a measurement model.
- Smoke runs confound training and split seeds, so initialization sensitivity is `NOT_ASSESSABLE`; split, family, preprocessing, and scenario sensitivity are reported separately.
- Development ranking stability uses development predictions and must be confirmed once on a locked external set.
- Capability rules are transparent diagnostics, not universal scientific standards.
- Treatment/control claim cards are associations unless assignment and causal identification assumptions are supplied.
- Nine permutation draws make the smoke null auditable but only resolve empirical p-values down to 0.10; increase `evaluation.permutation_draws` for stronger inference.
- BBB_Martins contains repeated compound identifiers, conflicting labels, and inconsistent structures for some names. The adapter preserves them; the example case explicitly excludes all affected identifiers before splitting and records the narrowed analysis population.
- Representation sensitivity is retrospective and based on two smoke manifests per scenario. Fixed manifests and matched model families isolate the comparison better, but do not replace locked external validation.
- Invalid SMILES produce zero Morgan vectors; v0.1 does not guess a chemical correction, so callers must validate input chemistry before confirmatory use.
