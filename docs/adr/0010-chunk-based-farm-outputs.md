# ADR-0010: Chunk-Based Farm Outputs

- Status: Accepted
- Date: 2026-11-06
- Supersedes: None

## Context

The output-development branch introduces `foxes.core.Output` and four concrete
farm-result outputs in `foxes.models.outputs.farm_results`. Output calculations
run on engine chunks before their results are unified. Dataset collection and
file writing must compose without sharing variable selections or consuming
each other's payloads.

The user requested regression tests and fixes for these implemented outputs,
not adoption of additional classes from `foxes.output`. The user also confirmed
that weekly files represent Monday-to-Sunday ISO calendar weeks.

## Decision

- All adopted outputs derive from `foxes.core.Output` and live under
  `foxes.models.outputs`.
- Workers call `calculate()` with prepared NumPy chunk containers; the engine
  calls `update_results()` with ordered payloads and `finalize_results()` after
  all chunks are processed.
- `FarmResultsOutput` collects one dataset under `FC.FARM_RESULTS`. Buffered
  writers use model-private keys and independently select their physical fields.
- Farm arrays keep `(FC.STATE, FC.TURBINE)` dimensions inside chunk payloads,
  including broadcast fields. State/turbine coordinates are explicit. Dataset
  creation compacts weights only if all included states are turbine-independent.
- Count files contain exactly the positive configured state limit except for
  the final remainder. The limit applies even within the final engine chunk.
- Period files use chronological, non-NaT datetime coordinates and calendar
  day, ISO week, month or year boundaries. Weekly filenames use the ISO year;
  missing periods produce no empty files.
- Writers forward their NetCDF options, apply standard path transformations
  and propagate worker failures by awaiting submitted file jobs.

## Compatibility And Migration

`Downwind.calc_farm(outputs=...)` accepts output models, model-book names or
lists of either. Physical variable selections move to
`FarmResultsOutput(variables=...)`; no variable-name compatibility branch is
added. A single output with one result is unwrapped; multiple outputs retain
the result-name mapping. Writer-only and disabled output return empty mappings.

This focused change does not migrate remaining `foxes.output` classes or repair
the other engines/algorithms still calling removed result-management APIs.
NumPy/thread farm integration is verified explicitly. Those wider migrations
remain visible blockers rather than hidden behind fallbacks.

## Consequences

- Dataset collection and writers can use different field selections in either
  execution order.
- File boundaries no longer depend on engine chunk sizes.
- Collection still retains all selected farm data. Buffered writers retain
  incomplete intervals and submitted write futures; no hard memory ceiling is
  promised for outstanding jobs or a large calendar period.
- `normalize=True` affects collected farm weights after finalization, not
  files that have already been written.
- Output hooks deliberately have a narrower prepared-data signature than
  generic `DataCalcModel.calculate`; targeted override annotations describe
  that existing specialization.

## Verification

[Farm output regression tests](../../tests/2_models/test_farm_results_outputs.py)
exercise real NetCDF files, exact state counts, ISO year boundaries, periods
within the final chunk, variable selections, metadata, input errors, worker
failures and independent output chains. Downwind tests compare NumPy/thread
calculations across chunk sizes and changing wind directions.

## Alternatives Considered

- Reusing one dataset payload for every output was rejected because writers
  may request different fields and consume their own pending results.
- Rolling seven-day windows were rejected in favor of the user-confirmed ISO
  calendar-week contract.
- Migrating all engines, algorithms and existing output classes in this change
  was excluded by the user's confirmed scope.

## References

- [Architecture](../architecture.md#chunk-based-farm-outputs)
- [Naming conventions](../naming-conventions.md#output-names-and-result-keys)
- [Output API](../source/api_output.rst)
