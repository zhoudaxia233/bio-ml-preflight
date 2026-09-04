# Constrained Autoprobe

Autoprobe borrows the small mutable surface and experiment log idea from autoresearch, but scientific evaluation cannot be optimized by one scalar. A candidate emits primary development performance, worst-scenario performance, split dispersion, ranking stability, runtime, complexity, and guardrail failures.

Selection applies validity and runtime guardrails first; the primary metric must clear a delta,
and worst-scenario and ranking evidence must not materially regress. Complexity is recorded
for review rather than used as an automatic tie-breaker. Negative experiments remain in JSONL,
and the final holdout is inaccessible to Autoprobe.

Both preparation and evaluation reject holdout-enabled cases before data access or report
writes. Evaluation checks the prepared case fingerprint, and a run's case and budget must
remain unchanged across candidates. Older run directories without that protocol record must
be replaced with a newly prepared directory; their historical decisions are not reinterpreted.

Candidate selection minimizes MAE, RMSE and log loss, and maximizes score metrics, using the
same direction as capability evaluation. Vectors retain raw metric units. The worst scenario
is the largest error or smallest score, and a candidate must improve on the latest accepted
candidate without materially worsening that boundary. Every scenario summary must contain
finite primary metrics for every scheduled seed, including for the first candidate; a finite
median cannot hide an invalid or missing run. Tolerances must be finite and nonnegative.
`KEEP` means acceptance within this development comparison; it does not assert a supported
scientific claim or successful independent confirmation.

This is a scaffold for two or three bounded alternatives, not an autonomous open-ended search.

Decision regressions cover controlled candidate sequences and one real synthetic development
run. The four qualitative synthetic cases also run with data seed 73129, reserved before the
decision-reliability repairs, with unchanged thresholds. These are bounded acceptance checks,
not estimates of false-positive or false-negative rates across biological datasets.
