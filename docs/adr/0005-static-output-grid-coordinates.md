# ADR-0005: Static Output Grid Coordinates

- Status: Accepted
- Date: 2026-10-08
- Supersedes: None

## Context

Output XY, XZ, and YZ grids define one plane shared by every selected farm
state. Repeating their coordinates over states makes a small spatial grid
require terabytes for long time series before engine chunking begins.
[ADR-0003](0003-compact-static-target-coordinates.md) already provides compact
static target transport and worker-side broadcasting.

## Decision

`foxes.output.grids` returns static grid coordinates shaped `(n_points, 3)`.
`SliceData` and `FlowPlots2D` pass them directly to `Algorithm.calc_points()`.
Grid-data tuples returned by mean, per-state, and direct chunk output contain
that same static representation. Selected states determine bounds and point
results, not copies of the spatial coordinates.

Internal target dimensions keep their singleton state axis until engine runners
broadcast it. Direct chunk precalculation uses a read-only broadcast view sized
only for its active state chunk.

## Compatibility And Migration

Grid coordinates no longer have a leading state axis. Maintained slice and
direct chunk consumers use the static-point contract; callers inspecting
returned grid-data tuples must treat their final item as `(n_points, 3)`.
No state-expanded alias or alternate grid format is retained.

Public point-result dimensions, spatial ordering, mean weights, plotting
parameters, YAML/WindIO inputs, and serialized result formats are unchanged.
FOXES-opt and iwopy do not directly consume these grid-coordinate arrays.

## Consequences

- Spatial grid-coordinate memory is independent of the number of states.
- All slice orientations and state selections use existing engine broadcasting.
- Mean slices still collect complete state-by-point results before reduction;
  this change does not bound the memory required by those result arrays.
- State-dependent farm bounds still include all selected turbine positions.

## Verification

`tests/0_consistency/test_output_grids.py` covers coordinate values and ordering,
conflicting selections, selected-state bounds, the million-state allocation
regression, serial state/point chunking, weighted means, reordered and singleton
state selections, and direct chunk-local broadcast views.

## Alternatives Considered

- Keep state-expanded grids. Rejected because identical coordinates are stored
  for every state before the engine can split the calculation.
- Return a full-state broadcast view. Rejected because output grids have no
  physical state dependence and the algorithm already accepts static points.

## References

- [Architecture](../architecture.md#data-containers)
- [Naming conventions](../naming-conventions.md#dimensions-and-structural-constants)
- [ADR-0003](0003-compact-static-target-coordinates.md)
