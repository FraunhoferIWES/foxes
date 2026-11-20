# ADR-0032: Output Dataset Units

- Status: Accepted
- Date: 2026-11-12
- Supersedes: Native-unit output policy in
  [ADR-0027](0027-domain-result-aggregation.md)

## Context

Numerical output datasets did not consistently identify physical units.
Power units depend on `TurbineType.P_unit`, and farms may contain different
units. The user required one unit per variable, no unit-based groups, and
conversion to the smallest unit used by the turbine types.

Iterative uses a farm collector for private downwind-ordered convergence
history, which feeds native-unit turbine models and under-relaxation.
Sequential preallocates accumulated datasets and must retain variable metadata.

## Decision

- Own unit mapping and output-only power conversion in
  `foxes.models.outputs._units`.
- Attach xarray `units` attributes to known scientific fields and coordinates
  in returned datasets and written NetCDF files, including renamed fields.
  Unknown custom fields receive no invented unit.
- Resolve the smallest unit through `FC.P_UNITS` over initialized turbine types.
  Before any output reduction, convert power, ambient power, capacity and
  maximum power by original turbine identity. Do not split spatial groups by
  unit. Record the common unit on each power variable and in dataset attribute
  `power_unit`. Uniform farms keep their turbine model's `P_unit`.
- When reprocessing collected farm results, use each field's declared unit
  instead of assuming native values. Raise contextual errors for unavailable
  turbine types or invalid power units.
- Keep yields in GWh/a, ratios/weights in `1`, rose frequencies in `%` and
  normalized rotor coordinates in `1`. Explicit non-unit slice normalization
  makes the affected coordinates/fields dimensionless.
- Collect private Iterative convergence power directly in native units and
  preserve its incoming downwind order for solver reuse. Do not advertise one
  physical unit on mixed-native private history. Public results remain normalized.
- Preserve variable attributes when Sequential accumulates state results.
  Do not mutate scientific inputs or running output instances.

## Compatibility And Migration

Output model-book selectors, constructors, dimensions and variable names are
unchanged. Mixed-unit public power values now change numerically to share one
unit, and mixed-unit group sums become valid. Scientific model containers
continue using native values; evaluation yields and ratios are unchanged.
Consumers read each variable's `attrs["units"]`, not an assumed kW default.
No new dependency, compatibility shim or legacy option is introduced.

## Consequences

- Returned and serialized values have the same unit contract.
- Reductions sum commensurate values and repeated output processing is
  idempotent.
- Conversion allocates new arrays only when scaling is required. Payloads
  retain NumPy values, not worker-side xarray objects.
- Private solver history uses the raw field-collection boundary without
  public output conversion; metadata must not relabel native scientific fields.

## Verification

- [Output-unit tests](../../tests/2_models/test_output_units.py) check known
  variables, ambient units, errors, renaming, files, normalized slices, roses,
  rotor metadata, Sequential accumulation and NumPy/thread/process engines.
- [Evaluation tests](../../tests/2_models/turbine_models/test_evaluation.py)
  cover mixed W/kW/MW types, changing downwind order, both algorithms,
  idempotent output reuse, aggregation and native Iterative history.
- [Aggregation tests](../../tests/2_models/test_result_aggregation.py) check
  conversion before grouped sums.

## Alternatives Considered

- Per-turbine unit arrays or unit-based groups were rejected by the user.
- Rejecting mixed-unit datasets was rejected in favor of physical conversion.
- Assuming kW or converting solver-owned arrays in place would misrepresent
  turbine models and alter scientific calculations.

## References

- [Architecture](../architecture.md#chunk-based-farm-outputs)
- [Naming conventions](../naming-conventions.md#physical-variables)
- [Models API](../source/api_models.rst)
