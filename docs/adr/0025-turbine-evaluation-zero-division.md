# ADR-0025: Turbine Evaluation Zero Division

- Status: Accepted
- Date: 2026-11-11
- Supersedes: The efficiency zero-division rule in
  [ADR-0016](0016-state-turbine-performance-evaluation.md)

## Context

`TurbineEvaluation` computes efficiency from state-turbine power arrays.
Raw NumPy division gives NaN for zero power over zero ambient power and emits
division warnings. The requested contract treats zero power as zero efficiency,
including at zero ambient power, and nonzero power over zero as positive infinity.

## Decision

- Enforce the rule in `TurbineEvaluation._evaluate`, including automatic
  [output-selected evaluation](0022-output-selected-turbine-evaluation.md).
- Return zero for a zero numerator, including either signed zero denominator.
- Return positive `np.inf` for a nonzero numerator over either signed zero,
  regardless of numerator sign, without division warnings.
- Preserve unclipped finite ratios and propagate NaNs otherwise. An unknown
  numerator remains NaN even over zero; a zero numerator still gives zero when
  the denominator is unknown.
- Retain positive finite nominal-capacity validation, full state-turbine
  shapes, selection semantics and read-only running model state.

## Compatibility And Migration

This directly replaces raw efficiency zero-division semantics without a legacy
mode. Zero-wind efficiency changes from NaN to zero. Negative power over zero
ambient power gives positive, not negative, infinity. Result names, dimensions,
model-book entries, YAML/WindIO schemas and dependencies are unchanged.

## Consequences

- Zero-wind outputs have deterministic efficiency without divide-by-zero
  warnings, independently of engine chunking.
- Missing nonzero power data are not silently converted to valid efficiencies.
- Work stays chunk-local and does not add state or turbine aggregation.

## Verification

[Evaluation tests](../../tests/2_models/turbine_models/test_evaluation.py) exercise
signed zeros, ordinary and integer inputs, NaNs, selections, lifecycle and
nominal-capacity validation. Automatic zero-wind output evaluation is checked
with Downwind/Iterative and NumPy/thread/process engines.

## Alternatives Considered

- Suppressing warnings around raw division retains the wrong `0 / 0` result.
- Replacing all nonfinite results with zero hides unknown data and required
  infinite results.
- Allowing zero nominal capacity would weaken unrelated metadata validation.

## References

- [Architecture](../architecture.md#model-lifecycle)
- [Model documentation](../source/models.rst)
