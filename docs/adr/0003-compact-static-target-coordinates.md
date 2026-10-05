# ADR-0003: Compact Static Target Coordinates

- Status: Accepted
- Date: 2026-10-05
- Supersedes: None

## Context

Point calculations represent target coordinates with dimensions
`(FC.STATE, FC.TARGET, FC.TPOINT, FC.XYH)`. Static coordinates do not vary by
state, but eagerly repeating them across every state makes memory use and worker
serialization scale with the full state-target product.

FOXES needs to preserve the established target dimension contract while avoiding
that unnecessary duplication. State-dependent target coordinates must continue
to support a distinct coordinate array for every state.

## Decision

Static `FC.TARGETS` retain the established dimension tuple with a singleton
`FC.STATE` axis and no state coordinate. State-dependent targets retain their
full state axis and state coordinate.

`TData` accepts either representation. Engine runners broadcast singleton-state
target coordinates to the active state chunk immediately before model
calculation. The broadcast array is a read-only NumPy view; calculation models
must treat target coordinates as immutable.

This decision does not alter engine chunk counts, chunk traversal, result
collection, or the public point-result schema.

## Consequences

- Static target storage and serialization scale with the number of targets, not
  with states multiplied by targets.
- All engine runners perform the same late target expansion before calculation.
- Internal code must distinguish the logical state count in `TData` metadata
  from the singleton leading axis of compact static coordinates.
- Code that needs writable target coordinates must make an explicit copy.

## Verification

- `tests/0_consistency/engines/test_process_engine.py` verifies compact static
  target slicing and worker-side expansion.
- `tests/2_models/test_binned_point_cloud_data.py` verifies that multi-state
  point calculations still expose full state-dependent target data to models.

## Alternatives Considered

- Eagerly repeat static targets across all states. Rejected because it
  materializes and serializes identical coordinates for every state.
- Remove `FC.STATE` from static target dimensions. Rejected because it would
  introduce a second target-coordinate dimension contract throughout models and
  data containers.

## References

- [Architecture](../architecture.md#data-containers)
- [Naming conventions](../naming-conventions.md#dimensions-and-structural-constants)
- [ADR-0001](0001-forward-only-development.md)
