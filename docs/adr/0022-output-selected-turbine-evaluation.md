# ADR-0022: Output-Selected Turbine Evaluation

- Status: Accepted
- Date: 2026-11-11
- Supersedes: The output-evaluator and aggregate-evaluation scopes of
  [0017](0017-constructor-defined-output-contract.md) and
  [0019](0019-reduced-output-payloads.md), and the manual-registration entry
  point described in [0016](0016-state-turbine-performance-evaluation.md).

## Context

`TurbineEvaluation` already owns state-turbine performance formulas.
`PrepareForOutput` can apply it from the selected output variables, without
requiring callers to attach a turbine model. Separate output evaluators and
farm/cluster aggregation implementations duplicate that responsibility.

An unrestricted collector promises available fields, not every possible
derived metric. Treating it as a request for all metrics also requests
probability yields without their required uncertainty, breaking all-fields
collection and Iterative's internal convergence output.

## Decision

- Use output variable selection to request turbine metrics. Output preparation
  evaluates missing supported fields before filtering and farm reordering.
- Unrestricted selections retain available fields only. Explicit probability
  yields still require configured uncertainty and fail when it is unavailable.
- Remove the competing output evaluator, farm/cluster aggregation classes,
  evaluation-only helpers, exports and model-book registration, without aliases.
- Keep ordinary collectors, writers, farm/cluster membership metadata and the
  `farm_results_analysis` variable selector.
- Remove aggregate summaries from maintained callers until a new aggregation
  implementation is introduced. Do not reproduce the old formulas in scripts.
- Keep turbine-history frame generation with animation rendering; it performs
  no metric calculation or aggregation.

## Consequences And Verification

Metric arrays retain `(FC.STATE, FC.TURBINE)` dimensions and use the existing
`TurbineEvaluation` formulas, units, validation and read-only running lifecycle.
Examples, YAML and notebooks select fields instead of invoking post-processing
evaluators. Removed APIs are not retained as compatibility paths.

Tests cover automatic output-selected metrics without manual registration,
Downwind/Iterative, original turbine order, ambient runs, chunked serial/thread/
process execution, unrestricted collection and missing uncertainty. Output,
animation, YAML and notebook checks cover retained consumers.
