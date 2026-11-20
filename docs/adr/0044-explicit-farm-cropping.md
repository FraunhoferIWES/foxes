# ADR-0044: Explicit Farm Cropping

- Status: Accepted
- Date: 2026-11-20
- Supersedes: None

## Context

`DatasetStates` derives horizontal and rotor-height selections during loading.
WindIO flow grids can extend beyond those farm bounds while remaining inside
the original resource support. Cropping then causes interpolation errors that
are not specific to Weibull distributions. Output grids are obtained after
states initialization, so forwarding points alone does not restore discarded
support.

## Decision

`foxes.input.states.DatasetStates` owns `crop_to_farm`, a boolean constructor
option enabled by default. Disabling it retains horizontal and vertical input
support instead of deriving bounds from the farm. Explicit `sel`, `isel` and
`height_bounds` still apply. Horizontal farm margins are ignored when disabled.

The WindIO states factory disables this switch for every `DatasetStates`
instance before initialization, without class-name special cases.
Interpolation settings and the existing model lifecycle remain unchanged.

## Compatibility And Migration

Python and FOXES YAML callers retain automatic cropping by default and can set
`crop_to_farm=False` for grids beyond the farm. WindIO schemas, result variables,
state labels, `(state, turbine)` farm dimensions and `(state, point)` point
dimensions are unchanged. No legacy branch or alias is introduced.

## Consequences

- Flow reporting can use resource support beyond farm bounds in both horizontal
  and vertical directions without enabling extrapolation.
- Explicit caller selections remain authoritative.
- Disabling cropping increases loaded support and per-chunk model data.
- Preloaded and on-the-fly file sources use the same selection policy.

## Verification

- [Dataset-states regressions](../../tests/2_models/test_dataset_states.py)
  check unchanged default cropping, disabled cropping on in-memory/preloaded/
  on-the-fly inputs, explicit selections and height bounds, and NumPy/process
  output and point calculations beyond the farm.
- [WindIO output regressions](../../tests/2_models/test_windio_outputs.py)
  check horizontal/vertical support, actual flow NetCDF writing, source reuse
  and strict regular-grid out-of-source errors.

## Alternatives Considered

- Disable cropping only for Weibull fields: rejected because the loading
  policy belongs to the shared dataset-states contract.
- Enlarge support from output points: not selected because it requires
  coordinating loading or reloading with grids obtained after initialization.
- Enable extrapolation: rejected because the original resource already covers
  the requested grid and out-of-source behavior must remain explicit.
