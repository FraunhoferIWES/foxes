# ADR-0031: Total Output Preset Family

- Status: Accepted
- Date: 2026-11-11
- Supersedes: All-turbine model-book naming in
  [ADR-0029](0029-grouped-output-preset-families.md)

## Context

The user accepted `total_results` as the brief name for aggregation across all
turbines. This follows the implicit grouping used by `farms_results` and
`cluster_results`.

## Decision

- Register every all-turbine farm collector and count/chunk/calendar writer under
  `total_results`, including the plain preset.
- Preserve optional ambient/waked filters, named field sets and independent
  state means. Examples are `total_results_analysis_mean` and
  `total_results_waked_P_mean_write_month`.
- Remove the previous `farm_results..._aggregate_all` selectors without aliases.
  Do not append a redundant `_aggregate_all` to `total_results`.
- Match writer filename stems to the selected family, filters, fields and mean.
- Preserve `aggregate_turbines="all"`, physical reductions, provenance and public
  dataset keys such as `farm_results_aggregate_all_mean`. A preset name is not
  a dataset schema name; `result_name` still customizes collector keys.
- Keep point aggregation under `point_results..._aggregate_all`. The new family
  combines turbines, not points.

## Consequences And Migration

Update Python/YAML selectors from `farm_results_analysis_aggregate_all_mean`
to `total_results_analysis_mean`, for example. All-turbine output file stems
change accordingly. Explicit output constructors and WindIO aggregation options
remain unchanged. No dependencies or compatibility shims are introduced.

## Verification

The full preset matrix checks actual classes, constructor options and writer
stems. Execution tests verify numerical results, raw/total composition, unchanged
dataset keys, YAML selection and every writer policy. Rejection tests cover
superseded names and redundant suffixes. The executed output tutorial demonstrates
single and composed model-book shortcuts.
