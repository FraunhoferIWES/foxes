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
5. `foxes.models`
6. `foxes.output`
7. remaining packages

### Rules to preserve

- Use `uv` for all Python commands and mypy runs.
- Run smoke tests after each sub-package is completed.
- Prefer concrete type annotations such as `dict[A, B]` over bare `dict`.
- Avoid `Any`/`object` unless a runtime contract genuinely requires it.
- Remove type text from NumpyDoc docstrings; type information belongs in Python signatures only.
- Keep changes narrow and focused; no unrelated refactors.
- Do not mark a package as complete until its public docstrings are scanned for legacy typed-parameter drift.

### Work pattern for each package

1. Audit the public API and identify public functions, methods, and classes with missing or legacy docstrings.
2. Fix type annotations first, especially constructor and method signatures.
3. Rewrite NumpyDoc sections so they read like:
   - `name` / `value` / `algo` / `results`
   - descriptions only, without `name: type` pairs
4. Validate with package-local mypy and a focused smoke test set.
5. Run the anti-drift grep scan and only then mark the package complete.

This is a hard stop: if the grep scan returns any hit for the package, that package is not complete, even if the active file looks fixed and even if mypy/pytest pass. A session that ends with a package marked complete while the grep scan is non-empty is invalid and must be corrected before the work is considered finished.

### Validation checklist per package

- Set `CURRENT_PACKAGE_DIR` to the package directory currently being updated.
- Set `CURRENT_PACKAGE_TESTS` to the focused test path for that package.
- `uv run mypy "$CURRENT_PACKAGE_DIR"`
- `uv run pytest "$CURRENT_PACKAGE_TESTS" -q`
- `grep -RInE '^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)' "$CURRENT_PACKAGE_DIR" || true`

The grep scan is not advisory. It is the failing gate and must be empty before a package can be marked complete. Run it after every edit batch. Every command must use `"$CURRENT_PACKAGE_DIR"`; never paste a previous package path into a new validation command.

### Current package status snapshot

- `foxes.core`: in progress (package directory: `foxes/core`)
- `foxes.engines`: pending (package directory: `foxes/engines`)
- `foxes.algorithms`: pending (package directory: `foxes/algorithms`)
- `foxes.input`: pending (package directory: `foxes/input`)
- `foxes.models`: pending (package directory: `foxes/models`)
- `foxes.output`: pending (package directory: `foxes/output`)
- remaining packages: pending (package directory: `foxes/<remaining_package>`)

### Current known issue pattern

Legacy docstrings repeatedly use forms like:

- `algo: foxes.core.Algorithm`
- `args: tuple, optional`
- `kwargs: dict, optional`
- `turbines: list of foxes.core.Turbine`
- `results: dict`

These are not allowed; they must be replaced by plain parameter names in the NumpyDoc text and the type information kept in the signature.

### Hard stop / fail-fast criteria

- a docstring line of the form `name: type` in a Parameters or Returns section
- a docstring fragment like `list of ...`, `tuple of ...`, `dict[...]`, or `numpy.ndarray` written as plain text in a parameter description
- a grep hit for the legacy typed-parameter pattern in the package directory
Use these exact command patterns in every package pass after setting the current package variables:

```bash
CURRENT_PACKAGE_DIR=foxes/<package-currently-being-cleaned>
CURRENT_PACKAGE_TESTS=tests/<package-related-tests>
uv run mypy "$CURRENT_PACKAGE_DIR"
uv run pytest "$CURRENT_PACKAGE_TESTS" -q
grep -RInE '^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)' "$CURRENT_PACKAGE_DIR" || true
```

Interpretation:
- a non-empty `grep` result means the package is not complete
- `mypy` and pytest are necessary but not sufficient
- the scan target must always be `"$CURRENT_PACKAGE_DIR"`, not a previous package path
- a package can still be broken even when the active method appears fixed

### Required recording format for each package status update

Each package status block should include:

- package name
- package directory used in the validation commands
- pass/fail status
- exact validation commands run
- whether the grep anti-drift scan was empty
- remaining files or hits, if any
- next action

This ensures restarts are deterministic and do not depend on memory or the previous chat state. The package directory field is mandatory because a fixed package must be distinguished from a stale `core`-only validation record.

### Restart hard guard: never close a session with an unknown package state

Before ending any work session:

1. Read this file.
2. Run the package grep scan for the active package directory.
3. If the scan is non-empty, keep working; do not mark the package complete.
4. Record the exact output or a note that the scan is empty.
5. Only then update the package status and continue to the next package.

This makes the hidden failure mode impossible to miss in a fresh session, because the grep result is a required part of the completion record.

## Rules

- Use `uv` for all Python commands and mypy runs.
- Run smoke tests after each sub-package is completed.
- Prefer concrete type annotations such as `dict[A, B]` over bare `dict`.
- Avoid `Any`/`object` unless a runtime contract genuinely requires it.
- Remove type text from NumpyDoc docstrings; type information belongs in Python signatures only.
- Keep changes narrow and focused; no unrelated refactors.
- Anti-regression rule: before a package is marked complete, run a grep-based drift check for legacy NumpyDoc patterns such as `name: type`, `list of ...`, `tuple of ...`, and `dict[...]` text inside docstring parameter/returns blocks, and fix any hits before closing the task.
- Completion gate: each package entry must include both validation output and the anti-drift scan result in the restart file so the same mistake cannot silently reappear.

## Execution order

1. foxes.core — in progress
2. foxes.engines — pending
3. foxes.algorithms — pending
4. foxes.input — pending
5. foxes.models — pending
6. foxes.output — pending
7. remaining packages — pending

## foxes.core status

### Completed
- Baseline audit completed.
- Initial public API docstring cleanup started in the core model/state interfaces.
- Mutable default dictionary issue in `Data.get_slice` corrected to `None` + local initialization.
- Legacy docstring type-text cleanup applied in the high-risk core entry points: `data_calc_model.py`, `model.py`, `algorithm.py`, `data.py`, `engine.py`, and `wind_farm.py`.

### Anti-regression guard
- Before closing the current package, run a grep-based scan for the legacy style:
  - `grep -RInE "^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)" "$CURRENT_PACKAGE_DIR"`
  - `grep -RInE "Parameters\n|Returns\n" "$CURRENT_PACKAGE_DIR"` followed by a quick manual review of the remaining docstrings.
- Any match is a blocker until it is fixed or intentionally justified in the package plan.

### Active targets
- `foxes/core/states.py`
- `foxes/core/farm_controller.py`
- remaining docstrings in `foxes/core/*.py` that still use typed parameter blocks

### Validation commands

- `uv run mypy "$CURRENT_PACKAGE_DIR"`
- `uv run pytest "$CURRENT_PACKAGE_TESTS" -q`
- `grep -RInE "^[[:space:]]+[A-Za-z_][A-Za-z0-9_]*:[[:space:]]*(foxes\.|dict|list|tuple|bool|str|int|float|array_like|numpy|xarray)" "$CURRENT_PACKAGE_DIR" || true`

## Notes

- The project-level mypy config is defined in `pyproject.toml` and is presently clean for the core package.
- The next step is to continue the core docstring and annotation pass in the remaining high-risk runtime/state files before moving to `foxes.engines`.
- This restart file now explicitly requires the anti-drift grep check to avoid a repeat of the same “cleaned in one file, missed in many others” issue.
