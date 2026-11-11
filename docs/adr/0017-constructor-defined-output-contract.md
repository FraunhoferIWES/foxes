# ADR-0017: Constructor-Defined Output Execution

- Status: Accepted
- Date: 2026-11-10
- Supersedes: The dataset-bound consumer contract for the classes promoted here.

## Context

Flow animations depended on dataset-bound plotting/evaluation classes and grids
under `foxes.output`. Those classes accepted large datasets and algorithms in
constructors and selected planes and reductions at method-call time, outside
the adopted core output lifecycle.

## Decision

Promote animation dependencies and slice consumers into `foxes.models.outputs`
as `foxes.core.Output` subclasses. Constructors define operations, planes,
reductions, variables, bins and rendering without calculation or scientific
data retention. Stateless private helpers take explicit data.

User execution goes through `calc_farm`, `calc_points` or `calc_outputs`.
The public `outputs` selector retains strings, instances and lists.
`get_calc_points(algo)` accesses initialized loaded data without retaining grids
on the output. Farm chunks evaluate requested points in bounded point slices;
output-only runs delegate point chunking to the engine. Point runs use supplied
points or precomputed results and ignore the hook. Composed point requests must
agree. Precomputed public point results retain `FV.X`, `FV.Y` and `FV.H`.

Runtime buffers stay in chunk payloads and result mappings; grids and selection
ordering stay in loaded data. Parent finalization performs reductions, formatting,
rendering and writing. Layout composition inside existing state-support plotting
uses stateless helpers instead of recursively executing an algorithm.

The model book registers lightweight presets for all newly adopted classes and
plane/reduction factories for the three slice families. The required state-rose
point and multi-slice positions have explicit 100-metre reference defaults.
Users configure selected presets before initialization or supply instances.

## Compatibility And Migration

Remove promoted classes and grid/rendering modules from `foxes.output`, without
aliases. Update Python callers, YAML/WindIO and notebooks to constructor-defined
outputs. YAML adopted-output entries have no `functions` list; unadopted
`foxes.output` consumers retain their own interface. Python `add_*` evaluation
callers assign the returned dataset; YAML orchestration updates its farm dataset.
Default farm fields and output model-book selectors are unchanged.

## Consequences

- Scientific output configuration is reproducible and serializable without data.
- Farm and point execution share output ownership and numerical implementations.
- Bounded scientific point blocks do not bound final result/animation memory.
- Callers needing independent wake points retain all required farm inputs.
- Migrated numerical/rendering helper modules retain cohesive existing
  algorithms, including their larger plotting and reduction functions.

## Verification

The output-contract, animation, grid, NetCDF, sequential, point-subset and grouped
evaluation tests cover constructor ownership, all execution paths, bounded points,
precomputed reuse, plane/selection parity, composition errors and file writes.
Changed notebooks are executed separately. Repository closure uses the complete
pytest suite and pre-commit.

## Alternatives Considered

- Retaining dataset-bound compatibility methods: rejected under forward-only
  development and because it would preserve two execution contracts.
- Storing loaded grids/results on model instances: rejected because running
  models must remain small and read-only.
- Removing string/list `outputs` selectors: rejected; the public selector is
  intentionally independent of the resolved output composition.
