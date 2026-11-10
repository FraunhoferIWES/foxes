# ADR-0016: State-Turbine Performance Evaluation

- Status: Accepted
- Date: 2026-11-09
- Supersedes: None

## Context

`TurbineEvaluation` in `foxes.models.turbine_models.evaluation` previously
implemented only efficiency. The requested yield, capacity and probability
metrics must run inside chunks without collapsing `fdata` dimensions or
depending on the engine's state partition. The user confirmed annualized,
unweighted per-state yields and explicitly supplied relative uncertainty.

The ambient turbine-model pass precedes copying power to `FV.AMB_P`.
Downwind does not reevaluate the first unwaked turbine during its wake sweep,
so turbine-model execution alone cannot finalize every ambient-dependent metric.

## Decision

- Keep all outputs in `(FC.STATE, FC.TURBINE)` arrays shared with `fdata`.
  Update only selected entries and leave running model instances read-only.
- Support nominal capacity, capacity factor, exact full-load indicator,
  efficiency, yield, P75/P90 and ambient capacity-factor/full-load/yield variants.
- Calculate yield as `P * P_unit_W * 8760 / 1e9` in GWh/a. Apply statistical
  state weights only during later aggregation.
- Name exceedance yields `FV.P75`, `FV.P90`, `FV.AMB_P75`, and `FV.AMB_P90`.
  Extend existing ambient mappings, extensive-variable sets and yield rounding.
- Require finite, nonnegative `power_uncert` for probability yields, using
  factors `1 - 0.675 * power_uncert` and `1 - 1.282 * power_uncert`.
- Use turbine-type power units and nominal powers, aligned through `FV.ORDER`.
  Reject invalid selected nominal powers and unsupported power units.
- Preserve the unclipped efficiency formula, including NumPy zero-division
  and NaN behavior.
- Add an evaluation-only controller pass after ambient copying and wakes,
  before output preparation. Iterative also evaluates final under-relaxed
  powers, aligning fresh original-order selections with downwind-ordered farm
  arrays without mutating model data. Power-changing models are not executed
  again.
- Select models through optional `model_names` on controller/model-list
  calculation, preserving order, parameters and turbine selections. `None`
  executes all models, an empty list executes none, and unknown names fail.

## Compatibility And Migration

Export `TurbineEvaluation` from `foxes.models.turbine_models`; callers register
their desired metric configuration in `ModelBook.turbine_models`. No default
model-book preset or default collected field changes. Existing configured and
calculation-time variable lists still merge, without mutating configuration.

Full-shaped results replace compact selected efficiency arrays, correcting the
existing turbine-model return-contract violation. No aliases, legacy modes,
new dependencies or serialized-file formats are introduced.

## Consequences

- Numerical outputs are independent of state chunk size and parallel backend.
- Mixed turbine power units produce consistently converted yields.
- Annualized state values are not duration-resolved energy or empirical
  percentiles. Summed probability yields assume the supplied common relative
  uncertainty; they do not model independent turbine uncertainties.
- Selected evaluation models run an extra vectorized pass, without repeating
  turbine types, controllers that modify power, or wake calculations.

## Verification

- [Metric tests](../../tests/2_models/turbine_models/test_evaluation.py) cover
  known values, mixed units, uncertainty and metadata errors, ordering,
  selections, model serialization, ambient-only cases and Downwind/Iterative
  runs on NumPy, single-chunk, thread and process engines, including final power
  under-relaxation.
- [Controller/model-list tests](../../tests/0_consistency/core/test_farm_controller_tmodel_sels.py)
  cover model-name filtering, execution order, empty and unknown selections.

## Alternatives Considered

- Weighted annual contributions were rejected in favor of the user-confirmed
  annualized state values.
- Full-state empirical percentiles were rejected because they require
  aggregation rather than independent chunk-local evaluation.
- Repeating all post-rotor models was rejected because power modifications can
  be non-idempotent.

## References

- [Architecture](../architecture.md#model-lifecycle)
- [Naming conventions](../naming-conventions.md#physical-variables)
- [Model documentation](../source/models.rst)
