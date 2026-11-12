# ADR-0036: Domain-Specific Total Keys

- Status: Accepted
- Date: 2026-11-12
- Supersedes: Shared farm and point total keys in
  [ADR-0034](0034-aggregation-output-keys.md) and
  [ADR-0035](0035-fixed-collector-keys.md)

## Context

The shared total keys prevented farm and point totals with matching state
reductions from appearing in the same output mapping. Collector keys remain
fixed by aggregation, and duplicate keys must not silently overwrite results.

## Decision

Farm collectors use `FC.TOTAL_FARM_RESULTS` and `FC.MEAN_TOTAL_FARM_RESULTS`
for all-turbine results, respectively without and with state reduction.
Point collectors use `FC.TOTAL_POINT_RESULTS` and `FC.MEAN_TOTAL_POINT_RESULTS`
for all-point results. Identical keys are still rejected.

## Compatibility And Migration

Replace previous shared total-key lookups with the domain-specific constants.
The old constants and values are removed without aliases. Farm and point totals
can now be composed in either order. Model-book selectors, file names, output
dimensions and numerical reductions remain unchanged. Downstream consumers,
including `foxes-opt`, must update their key lookups.

## Consequences

Total results identify both their domain and state reduction, without custom
names or ambiguity. Two collectors of the same domain and aggregation still
require separate calculations. Engine ownership and dependencies are unchanged.

## Verification

`tests/2_models/test_result_aggregation.py` verifies both combined totals in
either order, with and without state reduction, including numerical values and
duplicate-key rejection. Preset and turbine-evaluation tests verify farm keys.
The output notebook demonstrates the updated farm total keys.

## Alternatives Considered

- Shared total keys: rejected because they collide in composed outputs.
- Caller-selected keys: rejected because collector names are fixed.

## References

- [Architecture](../architecture.md#chunk-based-farm-outputs)
- [Naming conventions](../naming-conventions.md#output-names-and-result-keys)
