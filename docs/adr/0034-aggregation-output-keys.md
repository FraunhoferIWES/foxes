# ADR-0034: Aggregation Output Keys

- Status: Accepted
- Date: 2026-11-12
- Supersedes: Public collector key naming in
  [ADR-0027](0027-domain-result-aggregation.md),
  [ADR-0029](0029-grouped-output-preset-families.md) and
  [ADR-0031](0031-total-output-preset-family.md)

## Context

Farm and point collectors previously named aggregated datasets using suffixes
unrelated to the corresponding result constants and grouped farm preset names.
State means appended another suffix. The collector key is observable when
multiple outputs are composed.

## Decision

`FarmResultsOutput` selects `FC.FARM_RESULTS`, `FC.FARMS_RESULTS`,
`FC.CLUSTER_RESULTS` or `FC.TOTAL_RESULTS` for individual turbines, wind farms,
clusters or all turbines. State reduction selects the corresponding
`FC.MEAN_*` constant. `PointResultsOutput` uses `FC.POINT_RESULTS` for
individual points and `FC.TOTAL_RESULTS` for all points, or their corresponding
mean constants when states are reduced. Explicit `result_name` overrides the
default key. Private chunk keys and public dataset dimensions are unchanged.

## Compatibility And Migration

Composed Python and YAML consumers must use the new keys; single-result
unwrapping, model-book selectors, writer files and scientific reductions are
unchanged. No old-key aliases are retained. Farm and point totals share keys,
so callers composing both must give at least one an explicit `result_name`.
Downstream consumers, including `foxes-opt`, must update any old key lookups.

## Consequences

The public keys now identify aggregation without depending on selected fields
or filters. Sharing total keys across domains requires explicit naming in
mixed farm/point total compositions. Worker payloads, engine ownership and
optional dependencies are unaffected.

## Verification

`tests/2_models/test_result_aggregation.py` covers each farm and point key,
state means, custom names, point composition and numerical dimensions.
`tests/2_models/test_aggregation_configuration.py` verifies model-book
composition. The output tutorial demonstrates updated key lookups.

## Alternatives Considered

- Preserve suffix-based names: rejected because the requested constants are
  the public aggregation identifiers.
- Give point totals new domain-specific constants: rejected in favor of the
  confirmed shared total keys.

## References

- [Architecture](../architecture.md#chunk-based-farm-outputs)
- [Naming conventions](../naming-conventions.md#output-names-and-result-keys)
