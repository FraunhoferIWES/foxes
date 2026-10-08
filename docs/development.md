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

Put a new dependency in the narrowest justified extra. A dependency required by
an import of base `foxes` belongs in the main dependencies; an optional adapter
must not make its package mandatory. `uv.lock` is currently ignored and is not a
repository source of dependency versions.

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
| Evaluation, plotting, or writing | `foxes/output/` | Result schema and output tests/examples |
| Factory, geometry, or shared infrastructure | `foxes/utils/` | All owning call sites and `tests/4_utils/` |
| Public import | Owning package `__init__.py` | Root exports, API docs, and import tests |

Useful searches include the class or function name, the relevant `FC` dimension,
the relevant `FV` variable, and the model-book key. Search examples and YAML
files before changing a constructor or registered name; they are often the most
direct view of the public contract.

Do not treat `build/lib/foxes/` as source. The maintained package is `foxes/`.

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

Before closure, broaden to the repository gates:

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

## Test Structure

The numbered test directories encode intent:

| Area | Purpose |
|---|---|
| `tests/0_consistency/` | Engine, API, and cross-implementation consistency |
| `tests/1_verification/` | Analytical or trusted-reference verification |
| `tests/2_models/` | Model-family behavior and integration smoke coverage |
| `tests/3_examples/` | Execution of commands represented by example READMEs |
| `tests/4_utils/` | Utility contracts |

The aggregate example test excludes `examples/windio`. Changes to that example
or its CLI flow need a separate focused check with the required integration
available; do not infer coverage from `tests/3_examples/` alone.

`tests/conftest.py` fixes default engine chunk sizes to 64 states and 500 points
for deterministic behavior across machines. Do not make a test depend on a
machine's CPU count, memory, or implicit default chunking.

Logic tests cover a successful path, a meaningful error path, and a relevant
edge case. For numerical code, assert values, dimensions, ordering, and finite
or NaN behavior as applicable. Use deterministic synthetic arrays or public
packaged fixtures. Do not use private research or customer data as a fixture.

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

### Model Coverage

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

Subprocesses have two established, narrow exceptions:

- Example tests load `examples/run_all.py` in-process, while that harness runs
	the documented example commands. These tests are skipped when `CONDA_BUILD` is
	set.
- The real MPI subprocess smoke test is disabled by default. It runs only when
	`FOXES_RUN_MPI_TESTS=1`, `mpi4py.futures` is importable, and `mpiexec` or
	`mpirun` is available.

Do not copy either exception into ordinary unit tests.

## Running Examples

Each example directory owns a `README.md` with runnable commands. The aggregate
runner discovers those commands and can list or filter cases. Run it from the
examples directory:

```console
cd examples
uv run python run_all.py --dry
uv run python run_all.py --include compare_wakes --nofig
```

Use `--nofig` in automation. Use `--exclude` for unavailable optional
integrations and `--step` only for resuming a manual aggregate run. New examples
must clean up their generated artifacts and keep their README command current.

## Documentation

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
- For plots or other visual output, follow the applicable corporate design and
	accessibility guidance under `docs/fraunhofer-design/`.
