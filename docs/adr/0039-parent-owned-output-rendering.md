# ADR-0039: Parent-Owned Output Rendering

- Status: Accepted
- Date: 2026-11-13
- Supersedes: None

## Context

Figure-producing outputs retain rendering dictionaries that can reference
Matplotlib figures, axes, artists and callbacks with large object graphs.
Parent-side rendering alone does not prevent these objects from being
serialized with output models submitted to engine workers.

## Decision

`Output._parent_attributes` declares constructor attributes needed only on the
parent. Subclasses extend the inherited tuple. `get_initial_output_results()`
retains their references under `output.var("parent_data")`, without recursively
copying configuration containers. `set_running()` removes those attributes and
stashes them for exact restoration by `unset_running()`.

Output finalizers consume their private initial data and pass `parent_data`
explicitly to dataset processors, slice formatters and animation renderers.
They do not restore rendering attributes onto a running model. Worker-side
scientific requirements, such as layout colour fields and rotor-overlay
selection, retain only their necessary scalar selectors.

Layouts, roses, rotor plots, state/turbine maps, turbine curves, static flow plots
and both animation variants follow this contract. Animation configuration no
longer has deep-copied rendering snapshots. Metadata-only output processing and
the Sequential animation adapter obtain initial results before finalization.
Standalone rendering helpers and `Animator` remain parent-side objects.

## Compatibility And Migration

Constructors, model-book names, YAML/WindIO options, scientific dimensions,
result keys and file formats remain unchanged. Caller-supplied figures and axes
retain their identity. Internal `process()`, `format_payload()` and animation
rendering hooks receive an explicit `parent_data` argument; maintained callers
and test doubles move directly to this contract.

## Consequences

- Plotting objects and large rendering configuration stay off worker models.
- Configuration is restored after successful runs and worker/rendering failures.
- Private ownership entries do not leak into public result mappings.
- Finalizers receive explicit data rather than mutating read-only running models.

## Verification

Focused output tests serialize all ten plotting families with non-serializable
figure sentinels and a 32 KiB model-size limit. NumPy/process tests verify actual
parent figure reuse, composed result keys and failure restoration. Existing
slice, animation, YAML and Sequential tests cover dependent callers and files.

## Alternatives Considered

- Stashing only raw options leaves prepared rendering copies on worker models.
- Restoring options temporarily for rendering violates running-model ownership.
- Removing only keys named `fig` or `ax` misses nested callbacks and artists.

## References

- [Architecture](../architecture.md#algorithms-and-engines)
- [Engine lifecycle](0038-engine-calculation-lifecycle.md)
