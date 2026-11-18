# ADR-0042: Wake Growth Intercept And TI Multiplier

- Status: Accepted
- Date: 2026-11-18
- Supersedes: None

## Context

Wake constructors, turbine coefficient fields and WindIO used different
coefficient names or ordering. This made the same linear relation easy to
misconfigure. Coefficient naming is a public Python, YAML, model-book and
dataset contract.

## Decision

Use `k = k_0 + k_1 * TI` everywhere in FOXES. The dimensionless fields are
`FV.K_0 = "k0"` and `FV.K_1 = "k1"`; Python parameters are `k_0` and `k_1`.
`WakeK` handles direct wake configuration and the `kTI` turbine model computes
state-turbine growth from uniform values or farm coefficients. Wake evaluation
does not mutate the running coefficient handler.

Use model-book tokens `k0_<value>`/`k1_<value>` or
`ambk0_<value>`/`ambk1_<value>` for ambient TI. Put the intercept before the
multiplier in paired tokens. Require the separator before each value to avoid
colliding with existing constant `k004` notation.

WindIO retains its input schema: translate `k_a` to multiplier and `k_b` to
intercept, preserving `free_stream_ti` and turbulence-model inheritance.
Adopt WIFA's convention, `k = k_a * TI + k_b`, consistent with the
multiplier/intercept ordering in FLORIS and PyWake. The former FOXES WindIO
translation reversed these roles; correct it rather than preserving that
interpretation.

## Compatibility And Migration

Remove `FV.KB` and `FV.KTI` without aliases. Update maintained callers,
examples, notebooks and documentation in the same change. The `kTI` class and
unparameterized `kTI`/`kTI_amb` presets remain current model names, not aliases.
Parameterized turbine presets use forms such as `kTI_k1_02` and
`kTI_ambk0_001_ambk1_02`. Existing constant `k<value>` suffixes are unchanged.
Dataset coefficient keys change to `k0` and `k1`; dimensions stay
`(FC.STATE, FC.TURBINE)`.

The user explicitly requires rejection diagnostics for removed inputs.
Raise `DeprecationWarning` as an exception, rather than accepting or translating
old FOXES `ka`, `kb`, `kti` or `kTI` coefficient keywords or old parameterized
model-book names. Explain the equation and replacement names.
This bounded diagnostic-only exception to ADR-0001 has no compatibility
execution path; remove its guards when a subsequent public API revision
explicitly retires these migration diagnostics.
Downstream Python/YAML callers and packages such as `foxes-opt` must adopt the
new keywords, dataset fields and parameterized names. WindIO keys do not
change, but configurations targeting the former FOXES interpretation must swap
their coefficient values to preserve results. For constant growth 0.04,
use `k_a: 0.0` and `k_b: 0.04`.

## Consequences

- One coefficient convention applies across models and integrations.
- Constant growth needs no turbulence lookup; unspecified wake configuration
  still reads the computed `FV.K` field.
- Farm coefficient arrays remain chunk-local, with unchanged selections and
  dimensions; no engine-specific behavior or dependencies are added.
- Removed inputs fail loudly, including when Python normally hides warnings.

## Verification

- [WakeK tests](../../tests/0_consistency/core/test_wake_k.py): equation,
  ambient/local TI, constant/data-driven growth, selections, lifecycle and errors.
- [Turbine tests](../../tests/2_models/turbine_models/test_kTI_model.py):
  state-turbine coefficient arrays, overrides and missing coefficients.
- [Registry tests](../../tests/2_models/test_model_book.py): token conversion,
  constant-token separation, precedence and removed names.
- [WindIO tests](../../tests/2_models/test_niayifar.py): coefficient translation,
  defaults, turbulence inheritance and explicit ambient/local overrides.

## Alternatives Considered

- Retain legacy aliases: rejected in favor of a single contract and explicit errors.
- Use `k0<value>` without a separator: rejected because `k004` would mean
  either constant growth 0.04 or intercept 0.4.
- Change WindIO schema names: rejected because the external schema is unchanged.
