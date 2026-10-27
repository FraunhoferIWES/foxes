# ADR-0006: Micro Reference Disk Averages

- Status: Accepted
- Date: 2026-10-08
- Supersedes: None

## Context

`MesoMicroField.load_data()` evaluates micro-sector WS/WD and other fields at
reference points. Those values determine the sector directions and speedups
used to calibrate CFD fields against meso states. Exact-point calibration can
be sensitive to a local CFD feature and need not represent its neighborhood.

## Decision

Add optional `ref_point_radius`, a positive finite horizontal radius in metres.
During loading, select all distinct available native CFD support x/y points
inside each closed reference disk. Evaluate them at that reference point's
height and average with equal node weights. Convert WS/WD to U/V before
averaging; recover speed as the mean vector's norm and direction from that
vector. Average other micro variables arithmetically.

Extend the temporary loading farm's bounds to contain the disks. Store only
the aggregated reference data in the existing loading contract. Apply existing
reference validation and sector construction to these means. Raise a contextual
error if a disk contains no support points; do not substitute a nearest point.

`None` selects exact-point calibration. Turbine-target micro fields and public
result dimensions are unchanged. The model remains engine-independent.

## Compatibility And Migration

The optional constructor parameter is appended before `kwargs`, preserving the
positions of existing parameters. Python and YAML callers enable it explicitly;
maintained existing callers remain valid without changes. Reference data retain
their state/reference/variable ordering and shape. No serialized format, model
registry, dependency, or downstream optimizer contract changes.

## Consequences

- Calibration can represent a local CFD neighborhood rather than one point.
- Vector means handle direction wraparound and can have lower speed than the
  scalar mean speed when local directions differ.
- Equal node weights approximate area averaging on a regular uniform grid;
  they are not cell-area weights on an irregular grid.
- At CFD boundaries, averages use only available support points in the disk,
  without synthetic padding or extrapolated samples.
- Larger radii increase loading-time point evaluations and temporary memory.
- This does not smooth CFD gradients at turbine targets or certify improved
  optimizer convergence.

## Verification

[Reference-field tests](../../tests/2_models/test_ref_point_fields.py) cover
invalid radii, empty disks, native disk membership, multiple reference heights,
direction wraparound, scalar means, preserved input results, and loading-time
calibration of a synthetic multilevel CFD grid with and without averaging.

## Alternatives Considered

- Arithmetic WD means were rejected because directions are circular quantities.
- A separate synthetic probe grid was not selected because it would introduce
  another spacing or sample-count contract beyond the available CFD support.
- Averaging turbine-target fields was not selected because the requested owner
  is loading-time reference calibration, not pointwise CFD evaluation.

## References

- [Architecture](../architecture.md#states-input-and-static-data)
- [Implementation](../../foxes/input/states/meso_micro_field.py)
