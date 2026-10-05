# FOXES Naming And Typing Conventions

## Purpose

This document defines stable FOXES vocabulary and naming rules for Python code,
scientific data, model registries, tests, documentation, and public contracts.
Use the same term at every layer unless an external format requires a mapping.
See [architecture](architecture.md) for ownership and runtime contracts and
[development](development.md) for contributor commands.

Record a broadly consequential change to these rules in an
[ADR](adr/README.md). Do not rename an established public identifier solely to
make it match a newer style preference; model names, variable names, dimensions,
configuration keys, and xarray fields are compatibility-sensitive.

## General Rules

- Prefer a precise domain term over a generic noun such as `item`, `manager`,
	`handler`, or `data_object`.
- Use singular names for one object and plural names for collections. Preserve
	established collective types such as `States` and `PartialWakesModel`.
- Keep the same noun in code, docstrings, tests, YAML, and user documentation.
- Use an established FOXES abbreviation only when it appears below or in the
	public constants. Do not create near-synonyms for shortness.
- Name booleans as predicates or flags (`initialized`, `running`, `accept_nan`),
	counts with `n_`, indices with `_i` or `_index`, and starting offsets with
	`_i0` where that established convention applies.
- Include physical units in docstrings and metadata, not in a public variable
	name, unless the existing external contract already includes the unit.
- Preserve external names at I/O boundaries and translate them once into FOXES
	vocabulary.

## Domain Vocabulary

| Preferred term | Meaning | Avoid or distinguish from |
|---|---|---|
| FOXES | Farm Optimization and eXtended yield Evaluation Software | `Foxes` when naming the project |
| state | One ambient-condition sample; it may, but need not, represent a time | `timestep` unless time semantics are required |
| states | The `States` provider for one or more ambient conditions | A raw xarray dataset or generic `data` |
| wind farm / `farm` | The `WindFarm` and its ordered turbines | Site, layout, or results when those are meant specifically |
| turbine | A positioned `Turbine` in a wind farm | Turbine type or turbine model |
| turbine type | A model that supplies physical turbine characteristics such as power and thrust curves | Turbine instance or controller |
| turbine model | A model applied in the turbine model chain | Turbine type |
| model | A `foxes.core.Model` or derived calculation component | Algorithm or engine |
| model book / `mbook` | The `ModelBook` registry of named model instances and factories | A module import or arbitrary dictionary |
| model-book name | A key used to select a registered model | Python class name |
| factory name | A parameterized model-book name parsed by an `FDict` factory | Constructor call or class name |
| algorithm / `algo` | The object that assembles and orchestrates a complete farm or point calculation | Numerical model or execution backend |
| engine | The backend that chunks, dispatches, and collects calculations | Algorithm or worker process |
| farm results | The xarray dataset returned by `calc_farm()` | `FData`, which is an internal chunk container |
| point results | The xarray dataset returned by `calc_points()` | `TData`, which is an internal target container |
| ambient | The unwaked/reference value, represented by established `FV.AMB_*` variables | Inflow when a specific input source is meant |
| wake model | A model for wake deficits or related wake effects | Wake frame, wake deflection, or superposition |
| wake frame | The coordinate/order representation used to locate wakes | Wake model |
| wake superposition | The rule that combines multiple wake contributions | Partial-wake model |
| partial wakes | Rotor/target overlap integration for a wake model | Wake superposition |
| target | One requested evaluation location group | Turbine or target point |
| target point | One quadrature/evaluation point belonging to a target | Arbitrary public point result |
| chunk | The subset of states and/or points handled by one engine task | Full calculation or physical spatial region |

## Python Symbols And Files

- Modules, functions, methods, parameters, and local variables use
	`snake_case`; classes use `PascalCase`; module constants use `UPPER_CASE`.
- Existing scientific/public spellings such as `MData`, `FData`, `TData`,
	`ModelBook`, `WindFarm`, `PCtFile`, and the `FV` values remain canonical.
- Prefix non-public implementation details with one underscore. Do not export a
	private helper through a package `__init__.py`.
- Public re-exports use the explicit form `from .module import Name as Name`, as
	established throughout `foxes`. Keep the package root intentionally small.
- Package and source directories use lowercase names. Example directories use
	descriptive `snake_case`. Tests use `test_<subject>.py` and
	`test_<behavior>()` or an equally explicit behavior name.
- Put code in the module that owns the concept described in
	[architecture](architecture.md#module-boundaries). Do not name a miscellaneous
	module `helpers.py` when an existing focused module owns the behavior.
- Use `from __future__ import annotations` consistently with the surrounding
	module. Production annotations must remain valid for Python 3.10.

## Common Type Annotations

Infer types from the owning base class and call site rather than from a name
alone, but use these established meanings unless the local contract says
otherwise.

| Name | Usual type | Notes |
|---|---|---|
| `model` | `foxes.core.Model` or a derived class | In engine calculation paths this is usually `DataCalcModel` |
| `algo` | `foxes.core.Algorithm` | Use a concrete algorithm type only when its extra API is required |
| `farm` | `foxes.core.WindFarm` | Not an xarray dataset |
| `states` | `foxes.core.States` | May be a concrete input implementation |
| `mdata` | `foxes.core.MData` | Internal model/chunk data container |
| `fdata` | `foxes.core.FData` | Internal farm data container |
| `tdata` | `foxes.core.TData` | Internal target data container |
| `mbook` | `foxes.models.ModelBook` | Registry, not a plain mapping |
| `dbook` | `foxes.utils.DataBook` | Static/user data registry |
| `farm_results` | `xarray.Dataset` | Public farm-calculation result |
| `point_results` | `xarray.Dataset` | Public point-calculation result |
| `loaded_data` | `foxes.core.LoadedData` | Contains `coords`, `data_vars`, and `extra_data` |
| `ax` | `matplotlib.axes.Axes` | Add `None` only when the function creates an axes |
| `fig` | `matplotlib.figure.Figure` | Add `None` only when allowed by the contract |
| `wake_model` | `foxes.core.WakeModel` | A registered name may be `str` at a configuration boundary |
| `wake_frame` | `foxes.core.WakeFrame` | A registered name may be `str` at a configuration boundary |
| `wake_superposition` | `foxes.core.WakeSuperposition` | Keep distinct from partial wakes |
| `partial_wakes` | `foxes.core.PartialWakesModel` | Singular object despite the established plural name |
| `turbine_type` | `foxes.core.TurbineType` | A model-book name may be `str` |
| `turbine_model` | `foxes.core.TurbineModel` | A model-book name may be `str` |
| `rotor_model` | `foxes.core.RotorModel` | A model-book name may be `str` |
| `ground_model` | `foxes.core.GroundModel` | A model-book name may be `str` |
| `wake_deflection` | `foxes.core.WakeDeflection` | A model-book name may be `str` |
| `farm_controller` | `foxes.core.FarmController` | A model-book name may be `str` |
| `profile` | `foxes.core.VerticalProfile` | Use the concrete profile subclass when needed |
| `boundary` | `foxes.utils.geom2d.AreaGeometry` | Often optional; follow the signature |

A configuration parameter may intentionally receive a model-book name instead
of an instance. Annotate the actual boundary, including concrete mappings or
sequences of names when supported. Do not broaden an instance-only API to `str`
because another constructor resolves names.

Before using `Any` or `object`, search `foxes.core`, the nearest abstract base
class, and call sites. Prefer concrete built-in generics such as
`dict[str, np.ndarray[Any, Any]]`. Add a type alias only when it clarifies a
repeated, non-trivial contract; use the existing `LoadedData` alias rather than
redeclaring its structure. Imports needed by runtime worker code must not be
hidden behind `TYPE_CHECKING`.

## Dimensions And Structural Constants

Import `foxes.constants as FC`. Constants are part of the data contract, not
mere spelling conveniences.

| Constant | Value/concept | Use |
|---|---|---|
| `FC.STATE` | `state` | Ambient-condition axis |
| `FC.TURBINE` | `turbine` | Wind-farm turbine axis |
| `FC.TARGET` | `target` | Target axis |
| `FC.TPOINT` | `target_point` | Points within each target |
| `FC.POINT` | `point` | Public point-result axis |
| `FC.XY` | `xy` | Two-component horizontal coordinate axis |
| `FC.XYH` | `xyh` | Three-component x/y/height coordinate axis |
| `FC.TARGETS` | `targets` | Target-coordinate data key |
| `FC.TWEIGHTS` | `tpoint_weights` | Target-point weight key |
| `FC.STATE_TURBINE` | `state-turbine` | Composite dimension identifier |
| `FC.STATE_TARGET_TPOINT` | `state-target-tpoint` | Composite target-data identifier |

Farm variables ordinarily lead with `(FC.STATE, FC.TURBINE)`. Target variables
ordinarily lead with `(FC.STATE, FC.TARGET, FC.TPOINT)`, and `FC.TARGETS` adds
`FC.XYH`. Static `FC.TARGETS` keep that dimension tuple with a singleton
`FC.STATE` axis until an engine runner broadcasts them to the active state
chunk. State-dependent target coordinates carry the full state axis. Point
calculations use these target dimensions internally;
`calc_points()` selects the single target point and exposes `FC.TARGET` as
`FC.POINT` in the returned xarray dataset. Keep dimensions as tuples of
constants. Do not substitute array shape comments for dimension metadata.

Use established count and chunk names: `n_states`, `n_turbines`, `n_targets`,
`n_tpoints`, `chunk_size_states`, `chunk_size_points`, `chunki_states`,
`chunki_points`, `n_chunks_states`, `n_chunks_points`, and `states_i0`. Use
`sel` for label/value selections and `isel` for positional selections where the
API follows xarray terminology.

## Physical Variables

Import `foxes.variables as FV`. Use constants such as `FV.WS`, `FV.WD`,
`FV.TI`, `FV.RHO`, `FV.P`, and `FV.CT` instead of string literals. Use the
corresponding `FV.AMB_*` constant for unwaked values. Preserve canonical case,
including intentionally unusual existing names such as `FV.p` and `FV.AMB_p`.

- Coordinates: `FV.X`, `FV.Y`, `FV.H`, `FV.D`, and `FV.TXYH`.
- Wind: `FV.WS`, `FV.WD`, `FV.UV`, `FV.U`, and `FV.V`.
- Turbulence and atmosphere: `FV.TI`, `FV.TKE`, `FV.RHO`, `FV.T`, and `FV.p`.
- Turbine operation: `FV.OPERATING`, `FV.YAW`, `FV.YAWM`, `FV.P`, `FV.CT`,
	`FV.CAP`, and `FV.MAX_P`.
- Derived evaluation: `FV.YLD`, `FV.EFF`, `FV.CAPF`, `FV.FLF`, and the rotor
	effective wind-speed variables.

Create temporary model-specific keys with `model.var(variable)` and recover the
base name with `model.unvar(name)`. Do not reproduce its prefixing logic by
concatenating `model.name` manually.

## Data Container Names

Use `mdata`, `fdata`, and `tdata` only for their corresponding core containers.
Use `farm_results` and `point_results` for public xarray results. A plain NumPy
array should be named for its physical quantity or role, not `data` when a more
specific name is available.

For `LoadedData`:

- `coords` contains coordinates or dimensioned coordinate tuples.
- `data_vars` contains `name -> (dimension_tuple, ndarray)` entries.
- `extra_data` contains non-array or specially managed additional data.

Do not call all three channels `data`, and do not put a dimensioned model array
in `extra_data` merely to avoid declaring its dimensions.

## Models, Registries, And Factories

- Concrete model classes use descriptive `PascalCase` names. Their default
	`Model.name` starts from the class name but can be replaced by a model-book key.
- Model-book collection names are plural `snake_case`, for example
	`rotor_models`, `wake_models`, `wake_frames`, and `ground_models`.
- Registered keys are public identifiers. Follow the established family style;
	do not normalize existing mixed-case turbine names or underscored model names.
- Factory templates use angle-bracket placeholders, for example `grid<n2>`.
	Placeholder names match converted constructor concepts and include validation.
- `new()` means family-specific subclass lookup by class name. Do not document or
	call a general `Model.new()` because none exists.
- A variable ending in `_model` is an instance unless its annotation explicitly
	allows a registered name. Use `_models` for a sequence/mapping of selections.

Before adding or renaming a registered key, check `ModelBook`, examples, YAML
files, docs, and tests. Preserve factory caching and ensure errors list useful
available alternatives.

## Public APIs, Configuration, And CLI

- Constructor and YAML keys use the owning Python parameter name unless an
	external schema requires an adapter.
- Configuration values belong to `foxes.config`; environment variables use an
	uppercase `FOXES_` prefix when a process-wide override is genuinely needed.
- Console-script names declared in `pyproject.toml` are public. Keep reusable
	logic importable below the argument parser.
- Filesystem parameters end in `_file`, `_path`, or `_dir` according to what the
	caller may supply; do not use those suffixes interchangeably.
- Error messages name the relevant model or container and the expected versus
	actual variable, dimensions, shape, key, or backend.

## Documentation Names

Use [docstring conventions](docstrings.md) for required coverage, NumPy-style
structure, scientific contracts, examples, and review. This document remains
authoritative for the names used inside those docstrings.

- Section entries use the exact signature parameter or semantic return name.
- Use established `FC` dimensions, `FV` variables, model-book keys, and domain
	terms rather than prose-only synonyms.
- Link to the owning public API rather than inventing a friendlier name that
	users cannot search for.

## Test Names

The top-level numbered test areas retain their established meanings:

- `0_consistency`: cross-implementation and engine consistency
- `1_verification`: comparison with analytical or reference expectations
- `2_models`: model-family and model-integration behavior
- `3_examples`: executable example coverage
- `4_utils`: utility contracts

Name focused tests after observable behavior. Use parametrization identifiers
that expose the model, engine, or case being tested. Reuse
`tests/_model_smoke_helpers.py` for the model-family integration matrix instead
of creating parallel smoke-test terminology.

## ADR Naming

- File format: `NNNN-short-kebab-case-title.md`
- Start at `0001` and increment monotonically.
- Keep an ADR title stable after merge unless a later ADR supersedes the
	decision.
