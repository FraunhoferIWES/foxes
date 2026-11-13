# ADR-0038: Engine Calculation Lifecycle

- Status: Accepted
- Date: 2026-11-13
- Supersedes: None

## Context

Pool execution accumulated nested cleanup boundaries for output running hooks,
worker resources and scientific chunk caches. These ownership guarantees belong
in the engine contract rather than a particular parallel backend.

## Decision

`Engine.run_calculation()` owns output running hooks, parent-side result
finalization, success-only cache merging and unconditional cache reattachment.
Three abstract hooks create per-call state, execute backend chunks and clean up
backend resources. Creation allocates no external resources; cleanup tolerates
partial preparation and finishes workers before releasing shared storage.

`PoolEngine` implements these hooks. `DefaultEngine` delegates the backend hooks
under its own single lifecycle boundary and retains delegated engine contexts
until output finalization and worker cleanup finish. Per-call contexts, not
engine attributes, retain futures, progress, shared handles and cache updates.
The base engine also resolves ordered result batches, merges output payloads
and cache updates, and calls output update hooks. Per-call contexts retain both
pending chunk mappings and collected output mappings. Backends provide the
per-chunk progress callback and determine when to submit or collect a batch.
An optional base helper selects and validates scientific datasets and prepares
previous-farm-result metadata. It is called by the resolved backend, without
requiring pool scheduling, futures, worker transport or progress machinery.
Only `PoolEngine` requires the abstract `new_runner()` factory. Worker runners
remain separate from engines to avoid serializing parent-side execution
resources; non-pool engines and the default dispatcher need no runner factory.
Iterative's explicit full-state cache restoration remains unchanged.
Initial output mappings are created before stashing and retained in the parent
through delegation, collection and finalization. Composed outputs merge child
initial mappings without copying values and reject overlapping initial keys.
Recursive model cleanup uses non-null stash membership to avoid stopping shared
submodels already running under another owner's lifecycle.

## Compatibility And Migration

Python callers retain `run_calculation()` and the existing calculation options.
Custom engine implementations must implement the abstract hooks instead of
overriding the entire lifecycle. Both in-repository engine families migrate in
this change, without a legacy execution path.
There are no changes to scientific dimensions, result keys, YAML/WindIO input,
model-book names or serialized output formats.

## Consequences

- Output configuration and caches are restored after backend cleanup on both
  success and failure.
- Default dispatch does not start output hooks or finalize results twice.
- Shared-cache mutations are not transactional and can survive a failed run.
- Small per-call context objects make partially initialized resources explicit.

## Verification

Focused engine, algorithm and farm-output tests cover Pool success,
setup/worker/finalization failures, Default single/process delegation, cache
identity and iterative subset passes. Seeded and composed outputs verify
parent-side data identity, duplicate-key rejection and configuration restoration;
point outputs verify shared submodel ownership. Default-engine test doubles
implement the new hooks rather than overriding the public lifecycle.
Real Ray/MPI runtime parity remains opt-in on configured external infrastructure.

## Alternatives Considered

- Keeping nested lifecycle ownership in PoolEngine duplicates the contract for
  future engine families.
- Non-abstract hooks leave missing backend implementations detectable only at
  runtime.
- Delegating `run_calculation()` from within the default lifecycle would nest
  output running hooks and finalization.

## References

- [Architecture](../architecture.md#algorithms-and-engines)
- [Engine output execution and progress](0011-engine-output-execution-and-progress.md)
