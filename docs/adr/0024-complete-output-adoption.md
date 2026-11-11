# ADR-0024: Complete constructor-defined output adoption

## Status

Accepted.

## Supersedes

The remaining-consumer migration boundary in
[ADR-0017](0017-constructor-defined-output-contract.md).

## Context

The remaining writers, state/turbine tables and maps, arbitrary-point calculator,
rotor plots and turbine-type curves still captured results, algorithms or model
books. Their imperative methods bypassed the selected output lifecycle.
The pyplot proxy also conflicted with the common `show` display flag.

## Decision

All concrete outputs live in `foxes.models.outputs` and execute through
`calc_farm`, `calc_points` or `calc_outputs`. Remove the old output package and
its root export without compatibility aliases. Register the remaining classes
as lightweight model-book presets.

State/turbine maps, tables, curves and single-file writers collect only selected
fields into private worker payloads. Their results require state history, so
parent finalization assembles selected arrays once. Count/chunk/period farm
writers remain the bounded-file alternative to a monolithic NetCDF/CSV file.

`PointCalculator` owns coordinate configuration, not an algorithm or farm
dataset. Initialization loads static coordinates as shared extra data and
dynamic coordinates as state-sliced model data; raw coordinate configuration is
stashed while running. Scientific evaluation uses ordinary bounded point
blocks. State means send only weighted point/coordinate sums and weight mass,
merge on the parent and normalize once. Zero-weight NaNs are ignored.

`RotorPointPlot` reads the executing algorithm's initialized rotor metadata.
`TurbineTypeCurves` collects the executing algorithm's scan: callers explicitly
construct scan states and comparison turbines, including power-mask models.
Outputs never launch hidden curve simulations or mutate shared model books.

Keep `output_type: plt` as an explicit YAML-only exception: the YAML runner
calls Matplotlib directly, without an output class, model-book registration or
algorithm lifecycle. Stored Matplotlib figures/axes can receive method calls
through `object: $name`; output models cannot. Stored `$name` references still
compose constructor-defined outputs. Figure-less runs skip these plotting
calls and publish `None` placeholders for their labels. WindIO emits
constructor-defined turbine tables and slices.

## Consequences

All scientific Python, YAML, WindIO and notebook outputs use one contract;
YAML-only Matplotlib calls are the explicitly requested exception.
Plotting and complete-file writing stay parent-side; workers never retain
run-local buffers on output instances. Figure-less execution remains available.
There is no replacement farm/cluster aggregation in this decision.

## Rejected alternatives

- Moving dataset-bound classes without redesign would retain the contract defect.
- Hidden auxiliary curve simulations would conceal scientific inputs and engine
  work, and would retain shared model-book mutation.
- Parallel writes into one CSV/NetCDF file would require synchronization and
  format-specific append semantics without reducing the requested state history.

## Verification

Exercise real output entry points with uneven state/point chunks and
NumPy/thread/process engines. Assert state-free mean payloads and read-only
running models, exact coordinate/field order, renamed/transposed tables, CSV
layouts, singleton/datetime maps, metadata-only plots and composed YAML figures.
Run all quick examples once without display or remaining artifacts and execute
changed notebooks.
