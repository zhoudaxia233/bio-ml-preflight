# Repository guidance

- At the start of a task, read the local `Plan.md` when it exists and continue its
  first incomplete work item; keep that file local and never commit it.
- Treat generated JSON and Parquet artifacts as the numerical source of truth.
- Do not infer biological meaning from a column name.
- Preserve split manifests and never expose holdout labels to adaptive probing.
- Prefer a transparent baseline or rule over a speculative abstraction.
- A predictive association is not a causal effect.
- After verifying that a feature branch has been merged, delete it both locally
  and from its remote; never delete unmerged, default, or protected branches.
