# ADR-0011: Engine Output Execution And Progress

- Status: Accepted
- Date: 2026-11-06
- Supersedes: The engine/algorithm scope restriction in ADR-0010

## Context

The user extended the farm-output work to all engines and directly affected
algorithm callers. Chunk collection must have one contract across synchronous,
threaded, process and distributed execution. Remaining plotting/evaluation
classes are not adopted as chunk-output implementations in this change.
The user reconfirmed corporate design and permitted adaptive Unicode terminal bars.

## Decision

- Every engine uses prepared-data runners and the core `Output` lifecycle.
  An absent output produces an empty result mapping, not implicit raw arrays.
- Output models exclusively own NetCDF writing. Remove engine `write_nc`
  options and chunk-file helpers; compose a collector and writer when callers
  need both an in-memory dataset and files.
- Parent-side output updates remain in state/target submission order, even when
  workers finish out of order. Writer boundaries therefore remain deterministic.
- `Engine.await_results()` resolves ordered batches; local Dask schedules the
  entire lazy batch together, using processes by default. It does not compute
  one chunk graph at a time. Distributed tasks use `pure=False` for file effects.
- Engine progress is parent-owned, never stored on a scientific model.
  `progress_bar=True` selects tqdm in terminals and notebooks, with automatic
  Unicode/ASCII selection. Redirected streams receive bounded newline-only
  snapshots. `False` forces snapshots; `None` and verbosity zero disable progress.
  Progress uses stderr and does not configure global application logging.
- Chunk completion is distinct from output finalization. Finalization failures
  are labelled failed, never finished. Outstanding workers are drained before
  shared storage is released; failed lazy graphs are not replayed for cleanup.
- Distributed client/cluster/configuration contexts unwind in reverse order,
  including partial startup failures. Slurm starts its own cluster exactly once.
- Point collection lives in `foxes.models.outputs.PointResultsOutput`, under
  `FC.POINT_RESULTS`, assembling both state and target axes. Algorithms retain
  their public `(state, point)` datasets.
- Iterative convergence explicitly collects all intermediate fields in downwind
  order. Final outputs use the requested models and restore farm order.
  Running algorithms do not mutate their previous-result attributes.
- The model-book `"default"` field set remains unchanged. Callers requiring
  every wake input explicitly select `"farm_results_all"` or a tailored collector.

## Consequences And Verification

Real farm/point parity tests cover single, NumPy, threads, process,
multiprocess, local Dask and local distributed clusters. Chunk/count-writer tests
verify that files are complete before calculation returns. Progress tests cover
terminal/notebook selection, redirected logs, quiet modes and failures.

MPI parity was also exercised with the installed system runtime. Ray and MPI
runtime tests are opt-in through `FOXES_TEST_CLUSTER_ENGINES=1`; Slurm lifecycle
tests use synthetic infrastructure. Ray startup timed out in this environment,
so real Ray numerical parity remains unverified. Installing Python extras does
not provision working Ray services or Slurm infrastructure.

The complete branch suite remains blocked by unrelated removed state-provider
exports. These failures are reported rather than addressed with compatibility
aliases. Existing output consumers now import the core base directly, but their
old evaluation methods are not the new chunk hooks.

## Alternatives

Implicit raw-result collection and restored result-manager compatibility APIs
were rejected. Changing the default field selection to satisfy point callers
was rejected by the user. Backend-specific competing progress bars were
replaced with one shared display contract.
