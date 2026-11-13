# ADR-0040: Explicit Output Data Lifecycle

- Status: Accepted
- Date: 2026-11-13
- Supersedes: [ADR-0039](0039-parent-owned-output-rendering.md)

## Context

Rendering objects must stay off worker models, but the shared attribute registry
and generic rendering-data bundle introduce an unnecessary abstraction over the
existing initial-result and model lifecycle hooks.

## Decision

Each output explicitly implements its own `get_initial_output_results()`,
`set_running()` and `unset_running()` hooks when it owns data that must not travel
to workers. Initial results retain exact references under separate
`output.var(...)` keys. Named fields in the standard lifecycle stash preserve
the original attributes for restoration. Finalizers consume only their output's
private keys and pass the required options directly to their processing helpers.

`Output` has no attribute registry, generic ownership bundle or automatic
stashing implementation. Families may implement shared, concrete behavior in
their existing output base classes; subclasses extend the normal lifecycle hooks.
Scientific arrays and worker-required writing options remain available through
their established calculation contracts.

Metadata-only output processing uses `Engine.run_calculation(model=None)`, with
no scientific backend dispatch. It still obtains initial results, brackets
finalization with the output running hooks and restores configuration on failure.
Sequential animation adapters use normal output updates and finalization rather
than calling private renderers as a separate output execution path.

Native binned support roses are an output model, `SupportWindRosePlotOutput`,
registered as `support_rose_plot`. The output obtains the current algorithm's
native histogram through model-data inputs for each run. Its native bin edges,
support order and percentage weights are preserved without interpolation.
Binned states retain numerical accessors, not plotting or image-writing methods.

## Consequences

- Per-output ownership is explicit and auditable, with some deliberate lifecycle
  boilerplate instead of another shared abstraction.
- Caller-owned objects retain their identity and remain off serialized workers.
- Reusing a support-rose output does not retain another algorithm's histogram.
- Existing constructor choices, scientific outputs and chunk-writer execution
  remain unchanged; examples use public output selectors.

## Verification

Focused tests check exact native support histograms, current-algorithm reuse,
metadata-only execution, output composition, compact serialized models, supplied
figure identity, file writes and restoration after calculation/rendering errors.
Existing slice, animation, writer, YAML and Sequential suites cover dependent
consumers. Full tests and repository hooks remain the closure gates.

## Alternatives Considered

- Retaining the registry or renaming its generic bundle preserves the abstraction
  the user explicitly rejected.
- Restoring rendering objects onto running models breaks read-only execution.
- Disabling running hooks for metadata-only outputs creates a second lifecycle.
