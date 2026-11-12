# ADR-0035: Fixed Collector Keys

- Status: Accepted
- Date: 2026-11-12
- Supersedes: Custom farm and point collector naming in
  [ADR-0034](0034-aggregation-output-keys.md)

## Context

Farm and point collector keys identify their spatial and state reductions.
The user requires those keys to be fixed rather than overridden by callers.
Farm and point totals share keys, as do multiple collectors with the same
aggregation.

## Decision

Remove `result_name` from farm and point collectors and their writers.
Keep their existing aggregation-specific `FC` keys and private chunk buffers.
Reject duplicate collector keys with `ValueError` rather than replacing or
misinterpreting a dataset. Other output families keep their own naming contracts.

## Compatibility And Migration

Python and YAML callers remove collector `result_name` arguments and address
results through the corresponding `FC` constants. Collectors with the same key,
including farm and point totals with matching state reductions, run separately.
Single-result unwrapping, output selectors, NetCDF file names and numerical
reductions remain unchanged. No compatibility alias is retained.

## Consequences

Result names reliably reflect aggregation. Two datasets with the same fixed key
cannot be returned in one mapping, but no dataset is silently overwritten.
Engine chunking, parallel ownership and dependencies remain unchanged.

## Verification

`tests/2_models/test_result_aggregation.py` checks fixed keys, rejected
constructor options, duplicate farm/point keys and cross-domain total
collisions. Farm and point output suites cover collection and writing; the
output notebook demonstrates aggregation-specific keys.

## Alternatives Considered

- Retain optional names: rejected because the user requires fixed keys.
- Silently keep the first or last dataset: rejected because it loses results.

## References

- [Architecture](../architecture.md#chunk-based-farm-outputs)
- [Naming conventions](../naming-conventions.md#output-names-and-result-keys)
