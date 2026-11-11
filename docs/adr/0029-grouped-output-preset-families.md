# ADR-0029: Grouped Output Preset Families

- Status: Accepted
- Date: 2026-11-11
- Supersedes: Grouped model-book naming in
  [ADR-0027](0027-domain-result-aggregation.md) and
  [ADR-0028](0028-aggregation-presets-and-provenance.md)

## Context

The user requested `farms_results` for wind-farm aggregation and `cluster_results`
for cluster aggregation. They explicitly selected grouping implied by the family,
without redundant aggregation suffixes, and limited style cleanup to files
changed in this conversation.

## Decision

- Register every wind-farm grouping preset under `farms_results` and every
  cluster grouping preset under `cluster_results`.
- Preserve ambient/waked filters, named fields, state means and all count,
  chunk and calendar writer combinations. Examples are `farms_results_all_mean`
  and `cluster_results_amb_analysis_mean_write_month`.
- Remove previous grouped model-book names without aliases. Do not retain
  `_aggregate_wind_farm` or `_aggregate_cluster` on the new families.
- Keep `farm_results`/`point_results` ungrouped families and their independent
  `_aggregate_all`/`_mean` choices.
- Match writer stems to the selected family. Constructor parameters, physical
  reductions, attributes and public dataset keys remain unchanged.

## Consequences And Migration

Update Python/YAML output selectors to the new families. An explicit
`FarmResultsOutput(aggregate_turbines="wind_farm")` remains valid; its dataset key
is still `farm_results_aggregate_wind_farm`. Model-book spelling is not a dataset
schema rename. No dependencies or compatibility shims are introduced.

## Verification

The complete matrix in
[configuration tests](../../tests/2_models/test_aggregation_configuration.py)
resolves actual classes/options, rejects superseded/redundant spellings and
executes collection plus every writer policy. Existing raw result-key,
YAML/WindIO, chunking and numerical tests preserve the underlying contract.

## Alternatives Considered

- Replacing only the prefix while retaining aggregation suffixes was rejected
  explicitly in favor of implicit grouping.
- Changing public dataset keys was not requested and is outside this naming
  change.
- Repository-wide style refactoring was excluded by the confirmed changed-file
  scope.
