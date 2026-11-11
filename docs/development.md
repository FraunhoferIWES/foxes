# FOXES Development Guide

## Purpose

This guide collects repeatable repository workflows for human contributors and
coding agents. [AGENTS.md](../AGENTS.md) remains the policy source;
[architecture](architecture.md) explains runtime ownership and contracts; and
[naming conventions](naming-conventions.md) defines FOXES vocabulary, types,
dimensions, and identifiers. [Docstring conventions](docstrings.md) define the
public Python documentation contract.

Prefer a focused check that can disprove the current change before running a
broad suite. When code files change, finish with the full test suite. Every
development change finishes with repository-wide pre-commit, current docstrings
and FOXES documentation, and an entry in the current-version final changelog
section.

## Forward-Only Development

FOXES development is strictly forward-looking by default under
[ADR-0001](adr/0001-forward-only-development.md). Implement the target contract
directly, update all maintained callers, and remove the superseded code, tests,
examples, aliases, formats, fallback behavior, and documentation in the same
change.

Do not add a compatibility shim, deprecation branch, dual input format, or other
legacy path for convenience. A legacy exception requires explicit user
authorization, a bounded scope, and an ADR with an objective removal condition.
Identifying downstream impact remains mandatory; preserving the old contract
does not.

## Environment And Dependencies

FOXES supports Python 3.10 through 3.14. Keep production syntax compatible with
3.10 even when developing on a newer interpreter.

Use `uv` for local Python workflows:

```console
uv sync --extra dev --extra test --extra mpi --extra shp --upgrade
uv run pytest tests
uv run pre-commit run --all-files
```

The first command is the default local synchronization. Purpose-specific CI and
documentation jobs use narrower extras as shown later in this guide.

The dependency sets in `pyproject.toml` are optional extras, not uv dependency
groups. Install them with `--extra`; `--dev` does not select the `dev` extra.
Multiple extras can be combined by repeating the option.

| Extra | Use |
|---|---|
| `test` | pytest, notebook tests, pre-commit, mypy, and test runtime support |
| `dev` | interactive development and object-size investigation |
| `doc` | Sphinx, AutoAPI, numpydoc, MyST-NB, and documentation notebooks |
| `opt` | `foxes-opt` integration |
| `icon_dream` | ICON-DREAM conversion dependencies |
| `era5` | ERA5/metpy support |
| `shp` | geospatial shapefile support |
| `mpi` | MPI engine support |
| `ray` | Ray engine support |
| `multiprocess` | multiprocess backend support |
| `dask` | Dask and distributed/jobqueue support |
| `engines` | All optional engine packages, combining `dask`, `mpi`, `multiprocess` and `ray` |

Install all engine packages with `uv sync --extra engines` (or
`pip install 'foxes[engines]'` for a published release). This aggregate extra
reuses the individual engine extras so their version requirements stay aligned.
NumPy, single-chunk, thread and standard-library process engines need no
additional packages. MPI still requires a system MPI runtime, and Slurm-backed
Dask clusters require an available Slurm installation; the extra installs Python
packages, not external execution infrastructure.

### Engine Validation And Progress

Run engine, algorithm and farm-output checks together:

```console
uv run pytest tests/0_consistency/engines tests/0_consistency/algorithms tests/2_models/test_farm_results_outputs.py
```

For turbine performance evaluation, run
`uv run pytest tests/2_models/turbine_models/test_evaluation.py tests/0_consistency/core/test_farm_controller_tmodel_sels.py -q`.
Synthetic cases verify exact annualized yields, P75/P90 factors, mixed turbine
power units, changing turbine order, full-shaped selections, explicit failures
and read-only running model state. Ratio regressions cover
zero-power ratios without division warnings (`0 / 0` becomes zero; nonzero
power over zero ambient power becomes positive infinity), while preserving
NaNs and rejecting invalid nominal capacities. Coverage includes integer power
inputs, signed zeros and empty/scalar selections.
Real Downwind/Iterative calculations cover
NumPy, single-chunk, thread and process engines, model-specific turbine
selections, no-wake farms and ambient-only metrics. Model-name filtering checks
ensure the final evaluation pass does not rerun other turbine models.
Output-selection regressions request derived fields without registering any
turbine evaluation model, covering Downwind/Iterative, original turbine order,
ambient runs and NumPy/thread/process chunks. Unrestricted collectors retain
available fields only. Collector/writer aggregation is tested separately from
the superseded evaluation APIs.
Iterative cases include final power under-relaxation and original-order model
selection masks.

Real Ray/MPI parity tests are opt-in with `FOXES_TEST_CLUSTER_ENGINES=1`.
Use only a configured runtime and bounded worker counts; Slurm lifecycle checks
use synthetic infrastructure rather than submitting jobs. Local Dask defaults to
process scheduling and resolves bounded batches concurrently.

For adopted animations, run
`uv run pytest tests/2_models/test_flow_animations.py -q`.
For the remaining adopted writers, tables, maps, points and metadata plots, run
`uv run pytest tests/2_models/test_remaining_outputs.py -q`.
These cases verify all algorithm entry points, uneven NumPy/thread/process
chunks, state-free weighted point payloads, dynamic coordinates, original-order
renaming/CSV tables and parent-only plotting. YAML-only Matplotlib calls are
tested separately from model execution, including shared figures and entirely
figure-less execution. They do not require a pyplot output class.
For constructor-defined evaluation, layout, rose and slice outputs, also run
`uv run pytest tests/2_models/test_output_contract.py tests/0_consistency/test_output_grids.py tests/2_models/model_smoke/test_output_nc_writers.py -q`.
These tests exercise all three algorithm entry points, unchanged string/list
selectors, actual supplied target ordering, precomputed reuse without science,
disagreeing composed grids, bounded point blocks, all planes, state selection,
YAML execution and single-write multi-plane datasets. They also verify that all
adopted consumers are registered as lightweight model-book presets, that slice
factories validate plane/reduction choices, and that named presets execute
through the unchanged string/list selector. Slice-first hierarchy regressions
verify that static flow plots never initialize animation behavior or expose
animation-labelled shared data. Exact precomputed values, target reordering and
weighted means are checked in every plane in both NumPy and xarray formats.
Run `uv run pytest tests/2_models/test_output_payloads.py -q` for compact worker
payloads. Synthetic cases assert eliminated dimensions, fixed-size rose bins,
selected layout geometry and singleton static coordinates,
not just final numerical parity. They cover uneven chunks, zero-weight NaNs,
turbine-dependent weights,
negative point/state positions, global denominator thresholds, optional flow
vectors/overlays and NumPy/thread/process parity. Full datasets, animation frames
and unaggregated interval writers retain their required state-resolved fields.
Run `uv run pytest tests/2_models/test_result_aggregation.py -q` for domain
aggregation. Verify `aggregate_turbines`, `aggregate_points` and independent
`aggregate_states`, physical sums/means, recomputed power ratios, native-unit
errors, unassigned groups and per-entity normalization over partial selections.
Inspect worker payloads for eliminated dimensions, including file-interval
statistics before parent assembly. Compare uneven state/target tiles and exact
count/calendar files, not averages of chunk means.
Run `uv run pytest tests/2_models/test_aggregation_configuration.py tests/2_models/test_windio_outputs.py -q`
for the complete aggregation preset matrix and YAML/WindIO integration.
Resolve every selector to its actual class/options, including named field sets,
filters and all writer policies. Verify scalar pre-reduction counts over partial
selections and uneven worker tiles, exact per-file provenance and NetCDF
serialization. Raw collectors must not gain aggregation counts. Check preset
copy isolation, selection references, invalid flags and normalized/circular
WindIO flow means without changing existing slice weighted-sum semantics.
Grouped model-book checks use `farms_results`, `cluster_results` and
`total_results`, verify
matching writer stems, and reject superseded or redundant grouped spellings.
Run `uv run pytest tests/2_models/test_model_book.py -q` for model-book precedence:
check unique factory templates, every closed option, advertised examples and
representative numeric tokens against actual lookup and constructor parameters.
Disable factory caching during precedence tests. Membership alone does not prove
that a finite-option factory will construct the requested name.
Style cleanup of changed files separates component construction, state
selection, payload validation and tile assembly from their lifecycle coordinators;
keep declarative field tables intact rather than splitting them for line counts.
Run `MPLBACKEND=Agg uv run pytest tests/2_models/test_output_show.py -q` for
noninteractive rendering. Tests never use `show=True`, including mocked checks.
The shared pytest configuration disables display on test-owned model-book
presets/factories and rejects enabled display at the output hook. Tests verify
suppression, unchanged results, real encoded files, frame rates and figure
cleanup. Static model-book plotting presets normally enable display; animation
presets do not. Set `show=False` for headless scripts rather than changing the
library's backend or swallowing warnings. See
[ADR-0021](adr/0021-noninteractive-animation-and-tests.md).
Synthetic tests verify actual GIF frames, finalization-only rendering, numerical
point parity, bounded grid blocks, shared coordinates, target reuse, private
payloads, selection errors and writer cleanup. NumPy, single-chunk, threads,
process, multiprocess, local Dask and local distributed-cluster farm runs are
covered; point runs also exercise NumPy/threads/process collection. MP4 encoding
is tested when external ffmpeg is available.
Iterative cases verify requested-output initialization, calculation, buffering
and finalization occur only in the final evaluation pass. Streamline tests
use Iterative and compare animation fields against separate point results.
Memory checks verify that loaded grids and state-sized boolean selection masks
stay out of running model instances, raw configuration is restored from the
lifecycle stash, and worker calculations leave model serialization unchanged.

For synchronized flow/turbine animations, also run
`uv run pytest tests/2_models/test_flow_turbine_animation.py tests/3_examples/test_output_selection.py -q`.
These checks cover reusable trace configuration, original turbine ordering,
selected-state numerical parity, real two-panel GIF/MP4 encoding, failure
cleanup and the `dyn_wakes` example's local output registrations.

For iterator animation workflows, also run
`uv run pytest tests/2_models/test_sequential_animation_outputs.py -q`.
Animation composition is exported from `foxes.models.outputs`; all Sequential
plugins are exported from `foxes.models.seq_plugins`. Namespace checks ensure
the moved classes are not retained as legacy aliases.
These checks cover step-local numerical parity, state/chunk selections, real
GIF/MP4 output, synchronized traces and debug overlays, fractional frame rates,
and explicit model-selection errors. Execute changed animation notebooks with
`uv run pytest --nbmake notebooks/dyn_wakes.ipynb notebooks/timelines.ipynb notebooks/sequential.ipynb --nbmake-timeout=1800`.
Timeline state-chunk parity is covered by
`uv run pytest tests/2_models/test_timelines_chunking.py -q`: initial source
guesses converge to full-state results on NumPy and process engines. The
timelines notebook exercises state chunks and bounded point evaluation together.

Run all examples quietly with
`uv run python examples/run_all.py --quick --nofig`.
Every example accepts `-v`/`--verbosity`. The runner passes `-v 0` by default,
prints one entry per example and an aggregate completion summary, and propagates
diagnostics without global output redirection. Standalone scripts default to
level `2` (reports); level `1` selects a concise summary and `3` enables library
progress. Explicit verbosity in a
documented example command takes precedence over the runner default.
Plot-variable selection uses `--var`, leaving `-v` reserved for verbosity.
Levels `2` and `3` print each collected farm-results dataset once, independently
of fields required by later reports. This includes every comparison and optional
benchmark run; levels `0` and `1` suppress datasets. Writer-only and non-farm
selections never fabricate farm data. YAML/WindIO receive the same unreduced
example level, with scientific verbosity reduced inside their readers.
Animation examples forward this level directly to their selected animation
outputs. Level `1` announces creation; level `2` adds writing and successful
completion messages, frame counts and encoder settings; level `3` also reports
individual frame composition and encoding. Level `0` stays quiet, and failed
writes never announce completion.
Tests execute every example once and assert quiet algorithm/engine instances,
no displayed figures, restored configuration and no remaining artifacts.
Mocked reporting tests cover every Python script at levels `0` through `3`,
direct/composed results, partial fields, missing collectors and non-farm outputs
without repeating their numerical example calculations.

Engine `progress_bar=True` automatically uses tqdm for terminals/notebooks and
newline-only snapshots for logs. Unicode terminal bars are selected when the
stream supports them. Use `False` for explicit snapshots, `None` for no progress,
or `verbosity=0` for a quiet engine. Progress goes to stderr; a full chunk counter
is followed by output finalization before success is reported. No notebook widget
dependency is required: tqdm selects the available notebook/text renderer.
Algorithm progress labels identify the upstream FOXES operation, for example
`ProcessEngine [Downwind.calc_farm]` or `ProcessEngine [Iterative.calc_points]`.
The prefix is the concrete algorithm class, including subclasses such as
`Sequential`, rather than the base `Algorithm` class or a unique instance name.
Direct engine calls may supply keyword-only `progress_label`; when unset they
use the model class, not its unique instance name.
The launch helper and both default-engine delegation paths forward this label
unchanged; intermediate dispatch functions do not replace the upstream operation.

Select NetCDF writers through output models rather than engine calculation
options. Engine output tests cover worker-local chunk files and count-limited
buffered files, including file completion before the engine returns.

The engine migration's focused checks do not certify unrelated unfinished branch
migrations. The full-suite gate still exposes mismatched default-field
expectations and unsupported calculation options; do not hide these failures
with aliases or claim full-suite success.

Put a new dependency in the narrowest justified extra. A dependency required by
an import of base `foxes` belongs in the main dependencies; an optional adapter
must not make its package mandatory. `uv.lock` is currently ignored and is not a
repository source of dependency versions.

### Security Floors

The 2026-10-13 audit checked all published, non-yanked stable versions admitted
by direct build, runtime, and extra requirements against public OSV advisories.
The following minimum versions exclude the affected releases identified by that
audit:

| Dependency | Minimum | Scope | Advisories |
|---|---|---|---|
| tqdm | 4.66.3 | Runtime | [GHSA-g7vv-2v7x-gj9p](https://github.com/advisories/GHSA-g7vv-2v7x-gj9p) |
| Requests | 2.33.0 | `icon_dream` | [GHSA-9hjg-9r4m-mvj7](https://github.com/advisories/GHSA-9hjg-9r4m-mvj7), [GHSA-9wx4-h78v-vm56](https://github.com/advisories/GHSA-9wx4-h78v-vm56), [GHSA-gc5v-m9x4-r6x2](https://github.com/advisories/GHSA-gc5v-m9x4-r6x2) |
| GeoPandas | 1.1.2 | `shp` | [GHSA-6497-prx7-gpmq](https://github.com/advisories/GHSA-6497-prx7-gpmq) |
| Distributed | 2026.1.0 | `dask`, `doc` | [GHSA-c336-7962-wfj2](https://github.com/advisories/GHSA-c336-7962-wfj2) |
| JupyterLab | 4.6.4 | `doc`, `test`, `dev` | [GHSA-3325-v43h-43rv](https://github.com/advisories/GHSA-3325-v43h-43rv), [GHSA-3jqq-pw4j-pqcj](https://github.com/advisories/GHSA-3jqq-pw4j-pqcj), [GHSA-6966-vjj6-99xv](https://github.com/advisories/GHSA-6966-vjj6-99xv) |
| Tornado | 6.5.9 | `doc`, `test`, `dev` | [GHSA-3hv7-mjh2-fv65](https://github.com/advisories/GHSA-3hv7-mjh2-fv65), [GHSA-c2m8-h5v5-343r](https://github.com/advisories/GHSA-c2m8-h5v5-343r), [GHSA-chx6-46f5-w4vp](https://github.com/advisories/GHSA-chx6-46f5-w4vp) |

Dask also requires >=2026.1.0 to align with the patched Distributed release.
The existing setuptools >=84.0.0 build and extra requirements remain unchanged.
These declarations do not replace an audit of a resolved environment: transitive
dependencies, native libraries, and later advisories remain outside this check.

## Finding The Owning Code

Start from the smallest behavior owner, then inspect one abstraction boundary
and the nearest tests.

| Change | Start in | Then inspect |
|---|---|---|
| Core model lifecycle or data lookup | `foxes/core/` | Concrete callers and `tests/0_consistency/` |
| Wake, rotor, turbine, ground, or point behavior | `foxes/models/` | Its core base class, `ModelBook`, and `tests/2_models/` |
| Farm/point orchestration | `foxes/algorithms/` | Core algorithm contract and consistency tests |
| Chunking or parallel execution | `foxes/engines/` | Engine contract and engine consistency tests |
| State or file ingestion | `foxes/input/` | Input docs, examples, and boundary/error tests |
| Chunk-based output collection or writing | `foxes/models/outputs/` | `foxes/core/output.py`, Downwind, pool engine and farm-output tests |
| Constructor-defined outputs | `foxes/models/outputs/` | Selected fields, private chunk payloads and parent-owned finalization |
| Factory, geometry, or shared infrastructure | `foxes/utils/` | All owning call sites and `tests/4_utils/` |
| Public import | Owning package `__init__.py` | Root exports, API docs, and import tests |

Useful searches include the class or function name, the relevant `FC` dimension,
the relevant `FV` variable, and the model-book key. Search examples and YAML
files before changing a constructor or registered name; they are often the most
direct view of the public contract.

Do not treat `build/lib/foxes/` as source. The maintained package is `foxes/`.

## Code Quality

These rules elaborate [AGENTS.md](../AGENTS.md#code-quality):

- Prefer idiomatic, readable code over cleverness. Write code that reads
	naturally, as if written by a human: clear names, direct control flow and
	purposeful explanations.
- Start every logical code block with a brief, one-line comment explaining its
	purpose, and always separate logical blocks with a blank line.
- Put docstrings first in module, class and function bodies. Implementation
	comments belong below the complete docstring, never directly above it.
	`tests/4_utils/test_source_style.py` checks package code, tests, examples and
	notebook code cells for this placement rule.
- Keep each module focused on one concern. Keep functions under 40 lines and
	files under 700 lines where practical; justify exceptions in the completion
	report.
- Handle errors explicitly. Do not use empty catch blocks or swallow
	exceptions. Follow FOXES library diagnostics rather than configuring global
	application logging.
- Write clear, polished NumPy-style public documentation following
	[docstring conventions](docstrings.md).
- Record intentional compromises as
	`TODO(minor): <specific, actionable description>`, never vague deferred-work
	markers.

For all models, follow the
[model lifecycle and parallelization contract](architecture.md#model-lifecycle).
Prepare persistent configuration before running; runtime state belongs in
chunk containers, local variables, engine-owned chunk stores or explicit result
mappings, not on `self`.

## Documentation And Decision Records

Update documentation whenever a public API, CLI option, configuration value,
dependency, setup process, or build/deployment process changes. Review every
affected public docstring, including unchanged docstrings for continued
correctness, and deliver any updates with the code. Examples must be runnable.
Keep all affected FOXES facts synchronized; known drift blocks completion.

When a human or AI establishes, selects, changes or supersedes a durable
decision, update the relevant record in the same change:

- [Architecture](architecture.md): system context, selected technologies, module
	ownership, public contracts, data ownership, integrations, security,
	operational practices and cross-cutting decisions.
- [Naming conventions](naming-conventions.md): domain vocabulary and stable
	names for code, files, contracts, configuration and identifiers.
- [ADRs](adr/README.md): consequential architectural or naming decisions, using
	the index's rules for new and superseding records.

Record confirmed decisions, not alternatives presented as decisions, guesses
or placeholders. Ask a focused question when the decision is materially unclear.
Keep general policy in `AGENTS.md` concise: retain critical guardrails there,
link to these owning guides for detail, and avoid circular links that leave a
requirement unspecified.

## Focused Validation

Run the narrowest applicable command first:

```console
# One test
uv run pytest tests/path/test_module.py::test_behavior -q

# One test module or model family
uv run pytest tests/path/test_module.py -q

# Production type checking, as configured in pyproject.toml
uv run mypy foxes

# Repository hooks on touched files
uv run pre-commit run --files foxes/path.py tests/path/test_module.py
```

For the implemented chunk-based farm outputs, run:

```console
uv run pytest tests/2_models/test_farm_results_outputs.py -q
```

These tests cover dataset selection, state coordinates, broadcast/uniform
weights, metadata, real NetCDF file counts and encodings, calendar boundaries,
ISO week years, invalid inputs, asynchronous write failures and output
composition. Real Downwind runs compare NumPy and thread results across
different state-chunk sizes, changing wind directions and normalized weights.
The tests use synthetic inputs and temporary output directories.

Those farm-output tests validate their NumPy/thread paths. Broader engine,
algorithm and animation coverage is described above; do not treat one focused
suite as certification of every runtime. The full suite collects successfully;
older tests still expose unsupported calculation options and default-field
expectations that differ from the committed model-book selection. Report exact
failing test IDs separately from regressions in the current change.

For the analogous point outputs and their shared farm-writer policies, run:

```console
uv run pytest tests/2_models/test_point_results_outputs.py tests/2_models/test_farm_results_outputs.py -q
```

Point regressions use immutable synthetic `TData` and temporary real NetCDF
files. They cover both engine chunk axes, exact count/calendar intervals,
ISO week years, ambient/waked selections, independent writer buffers, dynamic
geometry and every target point. Real NumPy/thread/process calculations and
precomputed-point runs verify model-book default/all selectors without giving
outputs responsibility for scientific chunking. Packing, filename transforms,
invalid coordinates, incomplete/duplicate tiles and worker errors are explicit
checks.

Before closure, broaden to the repository gates:

Reference-calibrated sector-field changes should include
`tests/2_models/test_sector_field_mapping.py` and
`tests/2_models/test_ref_point_fields.py`. These cover original field-state
identity across locally sorted directions, north wraparound, blending choices,
and chunk boundaries.

```console
# Full Python test suite, required when code files changed
uv run pytest tests

# Notebook execution used by CI
uv run pytest --nbmake notebooks

# All formatting, linting, type, and file-hygiene hooks
uv run pre-commit run --all-files
```

Pre-commit owns the configured Ruff and mypy invocations. A hook can modify a
file and still exit non-zero; inspect the diff and rerun it.

Run notebooks with `MPLBACKEND=Agg` in an isolated copy of `notebooks/` and
`examples/` when checking generated files and animations. Do not overwrite
notebook outputs during validation. Keep notebook process counts explicit and
bounded; this changes execution resources, not the scientific inputs. Compare
each plotted case using that case's algorithm/results, rather than the last
algorithm left in a loop.

## Development Closure

A change is not complete until all closure evidence describes the final working
tree:

1. When code files changed, add or update tests for changed logic. Run the most
	discriminating focused tests first, then run `uv run pytest tests`
	successfully. Documentation-only changes run their applicable link, render,
	version, or structure checks instead.
2. Run `uv run pre-commit run --all-files` after the last edit. If a hook changes
	a file, inspect it and rerun the entire command.
3. Review every affected public Python API against
	[docstring conventions](docstrings.md). Update parameters, units, dimensions,
	behavior, side effects, returns, errors, and examples to match the final
	implementation.
4. Update `docs/architecture.md`, `docs/naming-conventions.md`, this guide,
	`docs/source/`, examples, ADRs, and any other FOXES information affected by the
	change. Do not knowingly leave stale prose, diagrams, commands, or contracts.
5. Read `project.version` from `pyproject.toml`. Confirm the last version section
	in `CHANGELOG.md` is exactly `## v<version>` and add a style-consistent entry
	for the change to that section.

Run notebook tests when notebooks change. A Sphinx build is not part of default
closure; use it as an explicit targeted check when requested or when rendered
Sphinx/AutoAPI output itself is under investigation. A documentation-only change
does not run the full runtime test suite, but it still runs its applicable
documentation checks and all other closure gates.

## Completion Report

When reporting an implementation, state:

1. **Summary**: what changed and why.
2. **Code**: implementation details, including any exceeded size guideline and
	its justification.
3. **Tests**: coverage and results, or why tests were not needed.
4. **Docs**: documentation changes, or `N/A`.
5. **Self-review**: rule compliance, declared deviations and any named exception
	recorded in `docs/data-classification-exceptions.md`.

## Test Structure

Repository-file inputs to pytest tests must come only from `foxes/`,
`examples/`, `notebooks/`, or `tests/`. Tests may create and read temporary
outputs. Documentation, release metadata, CI configuration, and other repository
files are validated separately, not by runtime tests. See
[ADR-0009](adr/0009-test-file-dependency-boundary.md).

The numbered test directories encode intent:

| Area | Purpose |
|---|---|
| `tests/0_consistency/` | Engine, API, and cross-implementation consistency |
| `tests/1_verification/` | Analytical or trusted-reference verification |
| `tests/2_models/` | Model-family behavior and integration smoke coverage |
| `tests/3_examples/` | Execution of commands represented by example READMEs |
| `tests/4_utils/` | Utility contracts |

Run `uv run pytest tests/3_examples -q` for every example, once each, with
bounded local inputs and no figure display. Scripts expose `main(argv=None)`;
the runner imports them directly and calls YAML/WindIO Python APIs without
shell commands. ICON-DREAM uses synthetic ICON/EWW inputs, Slurm uses a local
engine, and WindIO uses the bundled public document with local includes inlined.
This tests WindIO translation/execution, not its optional external YAML loader.
Normal runs retain real-input and scheduler defaults.

`test_output_selection.py` checks all script parsers, one-command README
selection, deduplication, and cleanup/restoration on success and failure without
repeating the numerical examples. Collector/writer/animation numerical and
encoding contracts remain covered by their model tests. Sequential selections
are limited to the farm collector and animation models.

`tests/conftest.py` fixes default engine chunk sizes to 64 states and 500 points
for deterministic behavior across machines. Do not make a test depend on a
machine's CPU count, memory, or implicit default chunking.

Logic tests cover a successful path, a meaningful error path, and a relevant
edge case. For numerical code, assert values, dimensions, ordering, and finite
or NaN behavior as applicable. Use deterministic synthetic arrays or public
packaged fixtures. Do not use private research or customer data as a fixture.

Pure styling, static text and documentation-only restructuring do not require
new runtime test cases or the full runtime suite. Applicable documentation
checks and the remaining closure gates still apply.

The layout-population interpolation regressions in
`tests/2_models/test_point_cloud_data.py` require the optional `foxes-opt`
integration and otherwise skip. With a sibling editable optimizer environment,
run them against the current FOXES checkout without synchronizing that
developer-owned environment:

```console
uv run --no-sync --project ../foxes-opt pytest tests/2_models/test_point_cloud_data.py -k layout_optimization_population -q
```

Static output-grid shape, memory, state selection, and serial chunking are
covered by `tests/0_consistency/test_output_grids.py`.

`tests/0_consistency/algorithms/test_downwind_ambient.py` covers ambient variable
retention, defaults, explicit outputs, error handling, and unchanged wake effects
for Downwind and Iterative using only the single engine. Binned-reduction and
mean-field tests also cover internal use of retained ordinary point outputs.

### Model Coverage

`tests/2_models/turbine_models/test_sector_management.py` covers original
turbine identity, state-turbine selection, invalid rules, wrapped direction
ranges, and shutdowns with changing downwind order and state chunking in both
Downwind and Iterative. Its synthetic three-turbine runs also check that wake
deficits remain on unmanaged turbines.

`tests/2_models/turbine_models/test_power_mask.py` covers derating, boost
continuity, Betz-limit thrust saturation, invalid transition widths, NaN masks,
selection, zero-width transitions, and shutdown.

`tests/2_models/test_ref_point_fields.py` covers meso/micro reference disk
membership, per-reference heights, circular wind-vector means, scalar means,
loading bounds, invalid radii, empty neighborhoods, and the exact-point default.
The smoothing fixtures are synthetic CFD grids and run with a single engine.

Public model-family tests use parametrized class paths and
`tests/_model_smoke_helpers.py`. Extend that matrix when a new public model fits
an existing family. A focused unit test is still required for distinctive
mathematics, validation, or failure behavior; the smoke helper proves only that
the public model can participate in a realistic algorithm path.

When adding a registered model, test both the concrete class and name-based
selection where the registration is public. Assert the `FC` dimensions and `FV`
outputs declared by the base contract.

### Engines And Parallel Code

Prefer in-process API tests. Constructing
`foxes.Engine.new(engine_type="process", ...)` in a test is supported; spawning
a separate CLI or shell process is usually unnecessary and is fragile in
conda-forge feedstock CI.

Engine-facing changes should compare numerical results with a simple/single
backend and exercise more than one chunk. Cover worker failure propagation,
resource cleanup, ordering, and shared/cache state when those paths change.
Keep worker entry points importable and serializable, and keep runtime imports
outside `TYPE_CHECKING` when workers execute them.

The real MPI subprocess smoke test is an established, narrow exception,
disabled by default. It runs only when
	`FOXES_RUN_MPI_TESTS=1`, `mpi4py.futures` is importable, and `mpiexec` or
	`mpirun` is available.

Do not copy this exception into ordinary unit tests. Example tests no longer
spawn CLI/shell processes or skip under `CONDA_BUILD`.

## Running Examples

Every example directory owns exactly one `run-all` and one `run-all-quick`
command marker in its README. Additional tutorial commands are not aggregate
runner entries. Every Python script remains a monolithic tutorial, with a
callable `main(argv=None)` and normal CLI launch. This is an explicit exception
to the usual function-size guideline.

```console
uv run python examples/run_all.py --dry
uv run python examples/run_all.py --quick --nofig
uv run python examples/run_all.py --include compare_wakes --nofig
```

The runner copies each example into a temporary directory, restores the working
directory, FOXES configuration (including UTM state), backend/interactive mode,
and removes its figures and generated files even on failure. Normal aggregate
runs also discard files; run an individual script to retain its outputs.
`--quick` selects bounded local fixtures independently of `--nofig`.
`--nofig` suppresses display, not explicitly selected file writers/animations.
All scripts accept `-o/--outputs` with intention-specific defaults and skip
reporting when required collected fields are absent. Use `--exclude` for
unavailable integrations in normal mode, and `--step` only to resume manual
runs. See [ADR-0023](adr/0023-in-process-example-runs.md).

## Documentation

Execute the user-oriented output notebook with
`uv run pytest --nbmake notebooks/outputs.ipynb --nbmake-timeout=300 -q`.
It uses six synthetic states, six public-curve turbines, a bounded NumPy engine
and temporary writer/animation directories. It covers composed result keys,
automatic metrics, physical aggregation, selected-state counts, point results,
layout/rose/flow plots, count/calendar files and GIF output.

Public Python APIs follow [docstring conventions](docstrings.md). Types remain in
annotations; NumPy-style docstrings describe the scientific and runtime
contract. Sphinx uses numpydoc and AutoAPI for package references and MyST-NB
for Markdown/notebook content.

When an explicit Sphinx rendering check is useful, use the same command as CI:

```console
uv sync --extra doc
uv run sphinx-build -E -b html docs/source docs/build/html
```

Open `docs/build/html/index.html` for local inspection. `docs/build/` and
generated AutoAPI pages under `docs/source/_*` are build output; do not edit or
commit them as source.

Update the relevant `docs/source/` page when public behavior, setup, an input
format, or an example changes. Keep examples executable. The root
`CHANGELOG.md` is the changelog source; `docs/source/CHANGELOG.md` is a symlink
to it.

Every development change adds a concise bullet to the final version section,
whose heading must match `project.version` from `pyproject.toml`. Add a
user-relevant fix under that version's bug-fix grouping. Preserve the section's
current style and keep the full-changelog link correct.

## CI Parity

GitHub Actions runs the Python tests and notebooks on every supported Python
version, currently 3.10 through 3.14. GitLab CI runs pre-commit, tests, and
notebooks in its configured uv/Python image. A local change that passes only on
the newest interpreter is not sufficient.

The normal CI commands are:

```console
uv sync --extra test
uv run pre-commit run --all-files
uv run pytest tests
uv run pytest --nbmake notebooks
```

Distribution-build jobs deliberately use the PyPA build frontend directly.
That CI implementation is an exception to the local `uv run` rule, not a model
for ordinary development commands.

## Release-Sensitive Changes

`pyproject.toml` is the package-version source and Sphinx reads it directly.
Release tags use `v<version>`; publish CI rejects a tag that does not match the
project version.

Before a release-sensitive change is complete, check:

1. Public imports and console entry points.
2. Model-book names, factory templates, YAML/WindIO contracts, and packaged data
	names.
3. The xarray result variables, dimensions, coordinates, and ordering.
4. The matching changelog section and user documentation.
5. Tests and notebooks across the supported Python floor and ceiling where the
	change is version-sensitive.

## Generated And Transient Paths

Do not edit these as implementation sources:

- `build/`, `dist/`, and `foxes.egg-info/`
- `docs/build/` and generated AutoAPI output
- Python, pytest, mypy, Ruff, notebook, and pre-commit caches
- Dask worker space, local logs, scratch files, and ordinary `results/` output

Some result-like files are intentional test/reference assets. Confirm that a
tracked test owns the file before changing it, and regenerate it only when the
task explicitly changes that expected contract.

## Change Checklists

### Model Or Algorithm

- Identify the core base class and required input/output variables.
- Preserve dimensions, state/turbine order, lifecycle, and chunk-local behavior.
- Keep model instance state read-only while running; verify the ownership of
	chunk data and run-level aggregation under threads and worker copies.
- Update exports and `ModelBook` only when the feature is public/name-selectable.
- Add focused mathematics/validation tests and realistic integration coverage.
- Update affected API docs and examples, and always add the current changelog
	entry required by the completion gate.

### Engine Or Data Container

- Test a small chunk, multiple chunks, empty/minimal axes where valid, and error
	propagation.
- Compare public results with the single/simple execution path.
- Check cleanup, serialization, imports in workers, and deterministic ordering.
- Measure representative memory and runtime; do not move a full target/state
	array into every task to simplify slicing.
- Run the broad consistency tests after focused engine tests pass.

### Input Or Output Adapter

- Validate external schema, names, dimensions, and values at the boundary.
- Keep optional imports isolated to the adapter.
- Test a valid sample, malformed input, and a relevant edge case in-process.
- Document the format and provide a runnable public example when appropriate.
- For plots or other visual output, follow the
	[project visualization conventions](../.github/instructions/visualization.instructions.md).
	FOXES uses free design; external publication requires a separate design and
	accessibility review.
