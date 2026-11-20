# ADR-0043: Unified Default Farm Variable Set

- Status: Accepted
- Date: 2026-11-20
- Supersedes: The unchanged default farm-field restriction in
  [ADR-0011](0011-engine-output-execution-and-progress.md)

## Context

The farm `default` variable set now includes `FV.CT` and every field in the
separate `calc` set. Keeping two selections duplicates the standard farm-input
contract for later waked point calculations.

## Decision

`foxes.models._model_book_outputs` owns one `default` selection for standard
layout, wind and power diagnostics and point-calculation inputs. Remove `calc`
from all farm collector and writer factories, including grouped and filtered
variants. `Downwind.calc_outputs` merges `farm_results_default` with explicit
farm-field requirements before new point science, without mutating either
selection. Unrestricted requirements still retain every available field.

## Compatibility And Migration

Replace `farm_results_calc` and corresponding grouped/filter/writer names with
their `default` variants. No legacy selector or alias remains, following
[ADR-0001](0001-forward-only-development.md). Python and YAML callers use the
same model-book names. Farm dimensions remain `(state, turbine)` and point
dimensions remain `(state, point)`; the default farm fields additionally include
thrust coefficients. Point variable sets and WindIO schemas are unchanged.

## Consequences

- Default farm datasets support standard downstream waked point calculations.
- Intermediate farm datasets also retain the default density and power fields.
- Model-specific extra wake inputs still require explicit fields or `all`;
  the finite default does not promise every configured model's requirements.

## Verification

- [Point-output tests](../../tests/2_models/test_point_results_outputs.py) compare
  default/all farm and point results on Downwind/Iterative with NumPy, thread and
  process engines.
- [Output-contract tests](../../tests/2_models/test_output_contract.py) verify
  merged selections, extra fields, unchanged configuration and state ordering.
- [Preset tests](../../tests/2_models/test_aggregation_configuration.py) exercise
  the maintained factory matrix and reject removed selectors.

## Alternatives Considered

- Keep `calc`: rejected because `default` already contains all its fields.
- Collect `all` implicitly: rejected because explicit finite selections must
  remain bounded and model-specific requirements must stay visible.
