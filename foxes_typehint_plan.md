# foxes type-hint and docstring remediation plan

## Goal

Audit and update the foxes codebase to satisfy two primary goals:

1. strict, complete type annotations
2. NumpyDoc-style docstrings aligned with sphinx-immaterial rendering

This file is the restartable plan for the project. It records the current status and can be re-read at any time to resume work without additional context.

## Original plan (full restart blueprint)

### Objective

Systematically modernize the foxes package so that all public APIs are fully typed and their docstrings follow the repository rule: types belong in the Python signature, while NumpyDoc text only documents names and semantics.

### Execution order

1. `foxes.core`
2. `foxes.engines`
3. `foxes.algorithms`
4. `foxes.input`
5. `foxes.utils`
6. `foxes.models` (sub-packages in this order)
  1. `foxes.models.axial_induction`
  2. `foxes.models.farm_controllers`
  3. `foxes.models.farm_models`
  4. `foxes.models.ground_models`
  5. `foxes.models.partial_wakes`
  6. `foxes.models.point_models`
  7. `foxes.models.rotor_models`
  8. `foxes.models.turbine_models`
  9. `foxes.models.turbine_types`
  10. `foxes.models.vertical_profiles`
  11. `foxes.models.wake_deflections`
  12. `foxes.models.wake_frames`
  13. `foxes.models.wake_models`
  14. `foxes.models.wake_superpositions`
7. `foxes.output`
8. remaining packages

### Rules to preserve

- Use `uv` for all Python commands and mypy runs.
- Run smoke tests after each sub-package is completed.
- Prefer concrete type annotations such as `dict[A, B]` over bare `dict`.
- Avoid `Any`/`object` unless a runtime contract genuinely requires it.
- Do not introduce local type aliases; keep public type contracts explicit at each signature during this remediation pass.
- Remove type text from NumpyDoc docstrings; type information belongs in Python signatures only.
- Keep changes narrow and focused; no unrelated refactors.
- Do not mark a package as complete until its public docstrings are scanned for legacy typed-parameter drift.

### Work pattern for each package

1. Audit the public API and identify public functions, methods, and classes with missing or legacy docstrings in every Python file under the package directory.
2. Fix type annotations first, especially constructor and method signatures.
3. Rewrite NumpyDoc sections so they read like:
   - `name` / `value` / `algo` / `results`
   - descriptions only, without `name: type` pairs
4. Validate with package-local mypy and a focused smoke test set.
5. Run the package-wide AST docstring audit over every Python file, then run the supplemental anti-drift grep scan; only then mark the package complete.

This is a hard stop: if the package-wide AST docstring audit returns any hit in any Python file, that package is not complete, even if the active file looks fixed and even if mypy/pytest pass. The supplemental grep scan must also be empty. A session that ends with a package marked complete while either scan is non-empty is invalid and must be corrected before the work is considered finished.

### Validation checklist per package

- Set `CURRENT_PACKAGE_DIR` to the package directory currently being updated.
- Set `CURRENT_PACKAGE_TESTS` to the focused test path for that package.
- `uv run mypy "$CURRENT_PACKAGE_DIR"`
- `uv run pytest "$CURRENT_PACKAGE_TESTS" -q`
- `grep -RInE '^[[:space:]]{12}[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)[[:space:]]*(,|$)' "$CURRENT_PACKAGE_DIR" || true`

The package-wide AST docstring audit is the authoritative failing gate. It must inspect every `*.py` file below `"$CURRENT_PACKAGE_DIR"`, parse each docstring with `ast.get_docstring`, inspect every `Parameters`, `Other Parameters`, `Returns`, `Yields`, and `Attributes` section, and report any field whose name is followed by a type annotation or type prose. The audit must be empty before a package can be marked complete. Run it after every edit batch. The grep scan is supplemental and must also be empty for its targeted legacy pattern. Every command must use `"$CURRENT_PACKAGE_DIR"`; never paste a previous package path into a new validation command.

### Authoritative Current Status

The status snapshot below and the latest validation record are authoritative. Older notes are historical context only; they must not override this snapshot or define the next action.

- `foxes.core`: complete (package directory: `foxes/core`)
- `foxes.engines`: complete (package directory: `foxes/engines`)
- `foxes.algorithms`: complete (package directory: `foxes/algorithms`)
- `foxes.input`: complete (package directory: `foxes/input`)
- `foxes.utils`: pending (package directory: `foxes/utils`)
- `foxes.models`: pending (package directory: `foxes/models`)
- `foxes.output`: pending (package directory: `foxes/output`)
- remaining packages: pending (package directory: `foxes/<remaining_package>`)

### Current Known Issue Pattern

Legacy docstrings repeatedly use forms like:

- `algo: foxes.core.Algorithm`
- `args: tuple, optional`
- `kwargs: dict, optional`
- `turbines: list of foxes.core.Turbine`
- `results: dict`

These are not allowed; they must be replaced by plain parameter names in the NumpyDoc text and the type information kept in the signature.

### Hard stop / fail-fast criteria

- a docstring line of the form `name: type` in any Parameters, Other Parameters, Returns, Yields, or Attributes section
- a docstring fragment like `list of ...`, `tuple of ...`, `dict[...]`, or `numpy.ndarray` written as plain text in a parameter description
- any AST-audit hit in any Python file under the package directory
- a grep hit for the legacy typed-parameter pattern in the package directory
Use these exact command patterns in every package pass after setting the current package variables:

```bash
CURRENT_PACKAGE_DIR=foxes/<package-currently-being-cleaned>
CURRENT_PACKAGE_TESTS=tests/<package-related-tests>
uv run mypy "$CURRENT_PACKAGE_DIR"
uv run pytest "$CURRENT_PACKAGE_TESTS" -q
grep -RInE '^[[:space:]]{12}[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)[[:space:]]*(,|$)' "$CURRENT_PACKAGE_DIR" || true
```

Interpretation:
- a non-empty AST-audit result means the package is not complete, regardless of which file contains the hit
- a non-empty `grep` result also means the package is not complete. The exact twelve-space match is supplemental: it targets common NumpyDoc field lines while avoiding ordinary Python signature annotations at other indentation levels, but it is not sufficient to cover every docstring layout
- `mypy` and pytest are necessary but not sufficient
- the scan target must always be `"$CURRENT_PACKAGE_DIR"`, not a previous package path
- a package can still be broken even when the active method appears fixed

### Required recording format for each package status update

Each package status block should include:

- package name
- package directory used in the validation commands
- pass/fail status
- exact validation commands run
- number of Python files inspected by the AST docstring audit and whether it was empty
- whether the supplemental grep anti-drift scan was empty
- remaining files or hits, if any
- next action

This ensures restarts are deterministic and do not depend on memory or the previous chat state. The package directory field is mandatory because a fixed package must be distinguished from a stale `core`-only validation record.

### Restart hard guard: never close a session with an unknown package state

Before ending a work session:

1. Read this file.
2. Enumerate every `*.py` file below the active package directory.
3. Run the AST docstring audit across that complete file list.
4. Run the supplemental grep scan for the active package directory.
5. If either scan is non-empty, keep working; do not mark the package complete.
6. Record the file count and exact output, or a note that both scans are empty.
7. Only then update the package status and continue to the next package.

This makes the hidden failure mode impossible to miss in a fresh session, because the grep result is a required part of the completion record.

### Current Next Action

- Begin the `foxes.utils` type and docstring remediation.

### foxes.input status

- package directory: `foxes/input`
- pass/fail: complete
- `uv run mypy foxes/input` — passed (`46` source files)
- focused smoke tests: `uv run pytest tests/0_consistency/input tests/2_models/model_smoke/test_states.py -q` — passed (`26 passed, 2 skipped`)
- full regression check: `uv run pytest -q` — passed (`230 passed, 5 skipped`)
- Python files inspected by AST docstring audit: `46`; final result `0` typed-field hits and `0` type-prose hits in `Parameters`, `Other Parameters`, `Returns`, `Yields`, and `Attributes` sections
- supplemental grep anti-drift scan: empty
- remaining files or hits: none in `foxes/input`
- next action: begin `foxes.utils` remediation

## Historical Validation Log

The entries below preserve useful evidence from earlier passes. Their “next action” lines are historical and do not override the authoritative status snapshot or current next action above.

## foxes.core status

### Completion status (2026-08-14)

- package directory: `foxes/core`
- pass/fail: passed
- `uv run mypy foxes/core` — passed (`24` source files)
- focused smoke tests — passed (`27 passed, 2 skipped`)
- docstring audit — no explicit typed parameter or return fields found by the AST-based audit
- note: the legacy grep pattern still reports four multiline signature annotations in `foxes/core/engine.py`; these are documented false positives, so the AST-based docstring audit is authoritative
- next action: audit `foxes/engines`

### Completed
- Baseline audit completed.
- Initial public API docstring cleanup started in the core model/state interfaces.
- Mutable default dictionary issue in `Data.get_slice` corrected to `None` + local initialization.
- Legacy docstring type-text cleanup applied in the high-risk core entry points: `data_calc_model.py`, `model.py`, `algorithm.py`, `data.py`, `engine.py`, and `wind_farm.py`.
- Additional legacy docstring cleanup applied in `states.py`, `farm_controller.py`, `farm_data_model.py`, `partial_wakes_model.py`, `point_data_model.py`, and `wake_model.py`.
- Full repository baseline on 2026-08-14: `230 passed, 5 skipped`.
- Algorithms pass started in `downwind.py` and `sequential.py`; both constructors now use concrete `WindFarm` and `States` annotations instead of `Any`.
- Algorithms package mypy is now clean across 22 source files after local inference fixes in `population.py`, `init_farm_data.py`, and `sequential.py`.
- Additional algorithms docstring cleanup applied in the Downwind model modules, including `point_wakes_calc.py`, `farm_wakes_calc.py`, `init_farm_data.py`, `reorder_farm_output.py`, `set_amb_farm_results.py`, and `set_amb_point_results.py`.
- AST-based docstring drift scan is clean for `foxes/algorithms/downwind/models/*.py`.
- `Any` audit on 2026-08-14 found 105 textual matches in 12 algorithm files; most are dynamic model factories, variadic compatibility parameters, or array/data payload boundaries. Concrete fixes applied to `Downwind.states` and `Sequential.plugins`.
- Iterative package docstring cleanup completed for `iterative.py`, `convergence.py`, `urelax.py`, and iterative `farm_wakes_calc.py`; AST-based scan is clean for `foxes/algorithms/iterative`.
- Sequential model docstring cleanup completed for `sequential/models/seq_state.py` and `sequential/models/plugin.py`; AST-based scan is clean for `foxes/algorithms/sequential`.

### Anti-regression guard
- Before closing the current package, enumerate every Python file below `"$CURRENT_PACKAGE_DIR"` and run the AST docstring audit across the complete list. The audit must inspect every `Parameters`, `Other Parameters`, `Returns`, `Yields`, and `Attributes` section:
  - report every field with a `name: type` declaration
  - report type prose such as `list of ...`, `tuple of ...`, `dict[...]`, or `numpy.ndarray`
- Then run the supplemental grep scan for the legacy style:
  - `grep -RInE "^[[:space:]]{12}[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)[[:space:]]*(,|$)" "$CURRENT_PACKAGE_DIR"`
- Any AST-audit or supplemental grep match is a blocker until it is fixed or intentionally justified in the package plan.

### Retroactive audit update (2026-08-14)

- Scope: all Python files under `foxes/core`, `foxes/engines`, `foxes/algorithms`, and the completed `foxes/input/farm_layout` sub-package.
- Files checked: `69`
- Initial genuine typed NumpyDoc field hits: `58` across `11` files.
- Affected files were cleaned in `foxes/core/data.py`; `foxes/engines/dask.py`, `mpi.py`, `multiprocess.py`, `numpy.py`, `pool.py`, `process.py`, `ray.py`, and `single.py`; `foxes/algorithms/iterative/models/convergence.py`; and `foxes/input/farm_layout/from_eww.py`.
- Final AST audit: `69` Python files checked, `0` typed-field hits.
- Revalidation: core mypy passed (`24` files) and focused tests passed (`27 passed, 2 skipped`); engines mypy passed (`11` files) and focused tests passed (`38 passed, 1 skipped`); algorithms mypy passed (`22` files) and focused tests passed (`3 passed`); farm-layout mypy passed (`12` files) and the focused input smoke test passed (`1 passed`).
- Supplemental grep: empty for the revalidated scopes apart from documented multiline Python-signature false positives in `foxes/core/engine.py` and algorithm signatures; the AST audit is authoritative for docstrings.
- Result at the time: core, engines, and algorithms were complete; this entry is superseded by the authoritative completion status for input above.
- Superseded: the package-wide input audit was completed successfully; `foxes.input` is complete and the next package is `foxes.utils`.

### Historical core validation note (2026-08-14)

- package: `foxes.core`
- package directory: `foxes/core`
- pass/fail: complete; the targeted cleanup and package-wide AST docstring audit are validated
- validation commands run:
  - `uv run mypy foxes/core` — passed (`24` source files)
  - `uv run pytest tests/2_models/model_smoke/test_states.py tests/0_consistency/core/test_farm_controller_tmodel_sels.py -q` — passed (`27 passed, 2 skipped`)
  - the original scan was non-empty because it matched ordinary Python signatures; the plan now uses the narrower twelve-space field pattern
- anti-drift scan: the legacy grep remains non-empty only because of four false-positive multiline signature annotations in `foxes/core/engine.py`; the AST-based docstring scan found no explicit typed parameter or return fields
- remaining files or hits: none identified in core docstrings
- next action: begin the `foxes.engines` audit

### Historical output validation note (2026-08-14)

- package directory: `foxes/output`
- pass/fail: validated out of execution order; package remains pending until after `foxes.models`
- `uv run mypy foxes/output` — passed (`25` source files)
- full regression check — passed (`230 passed, 5 skipped`)
- docstring anti-drift scan — empty for the package using the required twelve-space legacy pattern check
- remaining files or hits: package status is intentionally not closed; output changes were made before the models pass
- next action: complete the `foxes.models` package audit, then reassess output against the execution order

### Historical engines validation note (2026-08-14)

- package directory: `foxes/engines`
- pass/fail: passed
- `uv run mypy foxes/engines` — passed (`11` source files)
- focused smoke tests — passed (`38 passed, 1 skipped`)
- docstring audit — explicit typed fields were removed from the backend runner and calculation docstrings; the targeted residual search is clean

### Historical algorithms validation note (2026-08-14)

- package directory: `foxes/algorithms`
- pass/fail: complete
- baseline: `uv run mypy foxes/algorithms` initially reported 14 errors; all were resolved during this pass
- focused validation after the first edit:
  - `uv run mypy foxes/algorithms/downwind/downwind.py` — passed
  - `uv run pytest tests/0_consistency/algorithms/test_downwind_calc_points_subset.py tests/0_consistency/algorithms/test_sequential_state_count.py tests/0_consistency/iterative/test_iterative.py -q` — passed (`3 passed`)
- completed: `Downwind.__init__` and `Sequential.__init__` now annotate `farm` as `WindFarm` and `states` as `States`; their public docstrings now use plain NumpyDoc field names
- additional validation: `uv run mypy foxes/algorithms/sequential/sequential.py` passed; focused sequential tests passed (`2 passed`)
- Sequential docstring drift scan: clean for `foxes/algorithms/sequential/sequential.py`
- Point-wake validation: `uv run mypy foxes/algorithms/downwind/models/point_wakes_calc.py` passed; focused point calculation test passed (`1 passed`)
- Downwind-model batch validation: `uv run mypy foxes/algorithms` passed (`22` source files); focused algorithm tests passed (`3 passed`); AST docstring scan returned no hits
- `Sequential.__init__` now uses `plugins: list[SequentialPlugin]`; its focused mypy check and tests remain green
- Iterative validation: `uv run mypy foxes/algorithms` passed (`22` source files); iterative/sequential tests passed (`2 passed`); iterative AST docstring scan returned no hits
- Sequential-model validation: `uv run mypy foxes/algorithms` passed (`22` source files); sequential/iterative tests passed (`2 passed`); sequential AST docstring scan returned no hits

### Algorithms completion update (2026-08-14)

- package directory: `foxes/algorithms`
- pass/fail: passed
- `uv run mypy foxes/algorithms` — passed (`22` source files)
- focused tests — passed (`3 passed`)
- AST-based docstring drift scan — empty for all `foxes/algorithms/**/*.py`
- concrete typing improvements: Downwind and Sequential now use `WindFarm`/`States`; Sequential plugins use `SequentialPlugin`; population and Sequential return contracts were narrowed where known
- superseded: `foxes.input` was audited and completed; the active next package is `foxes.utils`

### Historical input work log (2026-08-14)

- package directory: `foxes/input`
- pass/fail: in progress
- baseline: `uv run mypy foxes/input` reports 204 errors across 16 of 46 source files, concentrated in state loaders and YAML/WindIO readers
- first focused slice completed: `farm_layout/from_df.py` and `farm_layout/from_csv.py` now have concrete public signatures and plain NumpyDoc fields
- focused validation: `uv run mypy foxes/input/farm_layout/from_csv.py foxes/input/farm_layout/from_df.py` passed; nearby input tests passed (`3 passed`)
- farm-layout subpackage complete: concrete boundary normalization added in `from_arrays.py`; helper docstrings cleaned across the farm-layout modules
- farm-layout validation: `uv run mypy foxes/input/farm_layout` passed (`12` source files); dependent regressions passed (`3 passed`); AST docstring scan returned no hits
- first state-loader slice completed: `states/scan.py` now has explicit intermediate array/selector types and normalized coordinate data; its legacy docstring fields were cleaned
- ScanStates validation: `uv run mypy foxes/input/states/scan.py` passed; state-model smoke tests passed (`23 passed, 2 skipped`)
- second state-loader slice completed: `states/states_table.py` now has explicit nullable dataset, array, restored-index, and coordinate types
- StatesTable validation: `uv run mypy foxes/input/states/states_table.py` passed; state and weight-dimension tests passed (`24 passed, 2 skipped`)
- StatesTable aliases removed: `ProfileDefinition`, `StateSelection`, and `StateLocations` were replaced with explicit unions; local type aliases are now prohibited by the plan
- third state-loader slice completed: `states/wrg_states.py` now separates raw array lists from stacked arrays and normalizes its coordinate data
- WRGStates validation: `uv run mypy foxes/input/states/wrg_states.py` passed; state-model smoke tests passed (`23 passed, 2 skipped`)
- fourth state-loader slice completed: `states/multi_height.py` now has explicit heights/coefficient arrays, normalized variable coordinates, and only the required override suppression
- MultiHeight validation: `uv run mypy foxes/input/states/multi_height.py` passed; state and Downwind multi-height regressions passed (`24 passed, 2 skipped`)
- DatasetStates alias cleanup: removed `DatasetSelection`, `PreprocessDataset`, `InterpolationParameters`, `DataGroup`, and `PreparedDataGroup`, and expanded their consumers in `single_state_field.py`, `weibull_sectors.py`, and `newa_states.py`
- DatasetStates alias-removal validation: related state/point-cloud tests passed (`29 passed, 2 skipped`); remaining mypy output is the pre-existing xarray union cluster in `dataset_states.py`
- DatasetStates typing continuation: separated reader results, fly-mode chunk lists, preload/lazy datasets, and prepared data mappings; normalized xarray keys, UTM metadata, drop-variable names, and interpolation points
- DatasetStates focused baseline reduced to 16 mypy errors; related state/point-cloud tests remain green (`29 passed, 2 skipped`)
- PointCloudData local typing slice: normalized fill-value handling, separated NaN index variables, and annotated point-cloud expansion arrays; focused mypy now reports only the existing `_read_ds` coordinate-shape contract and two broad interpolation-parameter call-site errors
- PointCloudData docstring audit: removed legacy typed NumpyDoc fields across attributes, constructors, readers, and interpolation methods; the required anti-drift grep is empty for `foxes/input/states/point_cloud_data.py`
- NEWAStates typing slice: aligned the interpolation-grid override with the `DatasetStates` tuple contract, narrowed fill-value handling, and separated NaN index variables; focused mypy passed and the `NEWAStates` smoke test passed (`1 passed`)
- SingleStateField typing slice: typed dynamic xarray selectors and the interpolation output mapping; focused mypy passed and the focused test passed (`1 passed`)
- PointCloudData typing slice: made tuple-valued point coordinates and merged interpolation parameters explicit; focused mypy passed, point-cloud smoke tests passed (`3 passed`), and the supplemental scan was empty
- WeibullSectors typing slice: typed dynamic reader/selectors and normalized DataArray-to-NumPy conversion; focused mypy passed, the `WeibullSectors` smoke test passed (`1 passed`), and the supplemental scan was empty
- MesoMicroField boundary slice: made loaded coordinate/data mappings explicit and reduced its focused mypy backlog from `40` to `36` errors; the remaining errors are concentrated in later calculation-shape contracts
- Current input backlog: `foxes/input/states/meso_micro_field.py` (`36` errors) and `foxes/input/states/ref_point_fields.py` (`25` errors)
- MesoMicroField calculation slice: separated the temporary MData mapping from the later stacked result array and annotated the temporary target-point array; focused state smoke tests remain green (`23 passed, 2 skipped`), while mypy still reports calculation-shape and dynamic LoadedData errors
- Input package AST audit: `46` Python files inspected, `0` typed-field or legacy type-prose hits; supplemental grep is empty
- Input package syntax check: `uv run python -m compileall -q foxes/input` passed
- superseded: the remaining calculation-shape typing was completed, package-wide mypy and both docstring scans passed, and `foxes.input` was marked complete in the authoritative status above
