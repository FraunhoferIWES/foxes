# ADR-0007: Smooth Power-Mask Boosts

- Status: Accepted
- Date: 2026-10-12
- Supersedes: None

## Context

[GitHub issue #26](https://github.com/FraunhoferIWES/foxes/issues/26) reports
discontinuous boosted power and thrust. In FOXES 1.9.8, `PowerMask` switches
directly to `FV.MAX_P` above nominal power minus `P_lim`. Its cubic induction
solver can also return no admissible root for boosts beyond the calibrated
Betz power maximum, producing nonfinite `FV.CT`.

The owner is `foxes.models.turbine_models.power_mask`. The registered `PMask`
model operates after the turbine type on arrays with dimensions
`(FC.STATE, FC.TURBINE)` and updates `FV.P` and `FV.CT`.

## Decision

Use cubic smoothstep to blend unmodified power into `FV.MAX_P` across the
final `P_lim` power units below nominal power. Require `P_lim` to be finite
and nonnegative; zero explicitly selects an abrupt boost above nominal power.
Keep direct derating, state-turbine selection, and NaN-mask semantics.

Bound the efficiency-adjusted power coefficient by `16/27` and select the
low-induction branch, allowing zero induction for shutdown. Saturate induction
at `1/3`, corresponding to thrust coefficient `8/9`, for oversized boosts.
Retain the requested boosted power. This is an engineering thrust approximation
and does not assert that a real controller can attain the requested power.

## Compatibility And Migration

Constructors, model-book names, power units, array dimensions, and result
variables remain unchanged. Positive `P_lim` now defines a smooth transition
instead of a switch threshold. The superseded abrupt positive-width behavior
is removed without a compatibility branch. Existing callers require no code
changes; public model documentation and the example describe the new behavior.

## Consequences

- Boosted power and thrust are continuous at the positive-width onset.
- Large boosts no longer produce invalid thrust roots or nonfinite results.
- The approximate thrust saturation does not replace turbine-specific
  controller or boosted power/thrust curves.
- Calculation stays chunk-local and vectorized, with no new dependencies.

## Verification

[Power-mask regressions](../../tests/2_models/turbine_models/test_power_mask.py)
cover transition boundaries, thrust saturation, derating, zero width, invalid
widths, NaN masks, selection, and shutdown.
[Sector-management integration](../../tests/2_models/turbine_models/test_sector_management.py)
also verifies power-mask shutdowns in Downwind and Iterative with state chunks.

## Alternatives Considered

- Retaining the abrupt boost switch leaves the reported discontinuity intact.
- Capping output power at the calibrated Betz maximum changes the established
  meaning of the requested boost and is not part of this thrust approximation.
- A turbine-specific boost controller requires additional physical input not
  supplied by this model and would be a separate modelling contract.

## References

- [FOXES architecture](../architecture.md)
- [Forward-only development](0001-forward-only-development.md)
- [GitHub issue #26](https://github.com/FraunhoferIWES/foxes/issues/26)
