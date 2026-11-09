# ADR-0013: Full-Shaped Broadcast Data

- Status: Accepted
- Date: 2026-10-15
- Supersedes: [ADR-0003](0003-compact-static-target-coordinates.md) and the internal target-storage decision in [ADR-0005](0005-static-output-grid-coordinates.md)

## Context

FOXES data containers declare dimensions independently of the shape of an
input array. Retaining singleton axes in stored arrays made callers and worker
code handle multiple shapes for the same dimension contract. `Model.get_data()`
also exposed separate `upcast` and `broadcast` modes, making callers choose
how to meet a target shape. Static target coordinates were the primary
exception: they remained compact until engine runners expanded them.

## Decision

Dimensioned data containers expose arrays at their full declared shape. If an
input axis is one and its known dimension size is larger, represent the
repeated values with a read-only `numpy.broadcast_to` view. Apply this after
dimension sizes are known, including when a later entry establishes a size.
Preserve actual size-one dimensions and temporary calculation axes.

`Model.get_data()` aligns named source dimensions and automatically returns a
full target-shaped value when the source shape differs. Remove the `upcast` and
`broadcast` options. Broadcasted results are read-only; code requiring writable
storage must make an explicit copy. Matching-shape results keep their existing
storage. Iterative wake-source reconstruction uses private writable storage
and does not mutate prior farm results.

Static target coordinates use the same full-shaped view contract as all other
dimensioned data. Engine runners do not perform a separate late expansion.
Static output grids remain `(n_points, 3)` at the public grid-data boundary as
specified in ADR-0005; `TData` exposes them with full target dimensions using
broadcast views. Serialization and copying preserve compact backing storage
for broadcast axes where possible.

## Compatibility And Migration

Maintained FOXES callers no longer pass `upcast` or `broadcast` to
`Model.get_data()`. Callers that need to mutate broadcast results must call
`copy()` explicitly. Data-container entries may now have a larger logical
shape than the array originally supplied, with read-only broadcast semantics.
The dimension tuples and public xarray result schemas remain unchanged.

## Consequences

- Consumers see one full-shape data contract rather than special singleton
  representations.
- Repeated values avoid materialized copies, but broadcast views are read-only.
- Serialization and copy operations must preserve or intentionally materialize
  the broadcasted shape.
- Engines no longer own a special static-target expansion step.
- Static output grid coordinates remain compact at their public input boundary;
  their internal `TData` representation follows this full-shaped contract.

## Verification

- `tests/0_consistency/core/test_data_shapes.py` covers shape normalization,
  read-only views, copying and serialization.
- `tests/0_consistency/core/test_model_get_data.py` covers target broadcasting,
  named dimension alignment, selection and error behavior.
- `tests/0_consistency/engines/test_process_engine.py` covers worker transport.
- `tests/3_examples/test_output_selection.py` exercises provided examples.

## Alternatives Considered

- Keep singleton storage and expand in every consumer. Rejected because it
  preserves competing shape contracts throughout models and engines.
- Materialize repeated values on insertion. Rejected because it wastes memory
  and increases worker serialization for values identical along an axis.
- Keep optional `upcast` and `broadcast` modes. Rejected because callers
  should not need to choose how target-shape normalization is performed.

## References

- [Architecture](../architecture.md#runtime-contracts)
- [Naming conventions](../naming-conventions.md#dimensions-and-structural-constants)
- [ADR-0003](0003-compact-static-target-coordinates.md)
- [ADR-0005](0005-static-output-grid-coordinates.md)
