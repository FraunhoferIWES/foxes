# ADR-0028: Aggregation Presets And Provenance

- Status: Accepted; grouped selector grammar superseded by
  [ADR-0029](0029-grouped-output-preset-families.md)
- Date: 2026-11-11
- Extends: [ADR-0027](0027-domain-result-aggregation.md)

## Context

The user requested model-book entries for every aggregation, dataset provenance,
removal of redundant helpers and equivalent YAML/WindIO behavior. They confirmed
that presets must include all collector/count/chunk/calendar, named-field and
ambient/waked combinations. Provenance is limited to original dimension counts,
not per-variable weights. WindIO flow files must remain gridded and support
state aggregation only.

## Decision

- Use the ordered selector grammar
  `<domain>_results[_amb|_waked][_<variables>][_aggregate_<group>][_mean]`,
  followed for writers by `_write_n<count>`, `_write_chunks` or
  `_write_<period>`. All supported spatial groups, field sets, filters and
  day/week/month/year policies are available. Register specific writer factories
  before broad field selectors rather than modifying global factory behavior.
- Include filter, named fields, grouping and state-mean components in writer
  filename stems. Preserve existing simple selector stems and default selections.
- Record scalar `n_states`, `n_turbines` and `n_points` attributes only for
  aggregated dimensions. Counts describe selected inputs before reduction and
  cover the complete result/file, not only its first worker. Chunk files count
  their own tile. Preserve existing unaggregated count/calendar metadata.
- Share dimension-aware concatenation and aggregation count bookkeeping.
  Remove duplicate chunk-join and farm-table assembly helpers. Keep distinct
  grid formatting and arbitrary configured-point classes; their contracts are
  not interchangeable with domain collectors.
- YAML accepts class constructors or copied model-book presets. Named presets
  reject constructor overrides; use model-book configuration or explicit classes.
  Resolve referenced state labels/positions and select inputs before reduction.
  Selection positions may be negative; WindIO retains its nonnegative subset
  contract. Preserve requested order for state-resolved grids and tables.
- WindIO turbine reports use farm reductions, external field names and optional
  wind-farm/cluster axes. Flow reports accept boolean `aggregate_states`, retain
  grids and reject point aggregation. Explicit normalized slice means share
  sufficient statistics and circular direction reductions; ordinary
  `reduction="mean"` retains its different global weighted-sum contract.
- Mean and state-resolved xarray grid formatting both honor field-name mappings,
  so aggregated WindIO flow files retain external names.

## Consequences And Compatibility

No separate evaluator or migration alias is introduced. Existing simple
selectors, unaggregated schemas and ordinary slice weighted sums remain valid.
Aggregation provenance adds only scalar attributes, not arrays proportional to
the number of fields/entities. YAML selection occurs before an axis disappears;
named-preset execution does not mutate cached model-book figure settings.
No new dependency is required.

## Verification

[Configuration tests](../../tests/2_models/test_aggregation_configuration.py)
resolve the complete selector matrix, execute every writer family, verify counts
across partial selections/uneven tiles and check YAML references/preset isolation.
[WindIO tests](../../tests/2_models/test_windio_outputs.py) verify actual summary
files, normalized/circular gridded means, field names and invalid options.
Existing output tests retain raw, table, slice and worker-payload coverage.

## Alternatives Considered

- Per-variable/per-entity weight metadata was rejected in favor of scalar counts.
- Spatial flow aggregation was rejected because reports must retain their grids.
- Separate YAML/WindIO numerical reducers were rejected in favor of shared
  sufficient statistics.
- Silently accepting named-preset constructor overrides was rejected because
  mutating arbitrary attributes bypasses constructor validation and dependencies.
