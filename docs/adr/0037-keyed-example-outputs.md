# ADR-0037: Keyed Results And Figure Suppression

- Status: Accepted
- Date: 2026-11-12
- Supersedes: The output-selection and headless-script details of
  [ADR-0023](0023-in-process-example-runs.md), and the figure-suppression
  execution details of [ADR-0020](0020-output-display-control.md)

## Context

Examples previously overloaded `-o/--outputs` for farm calculation and retained
dataset-specific reports and plots afterward. A single output's value is
unwrapped by the calculation APIs, losing the public key needed for uniform
reporting. Disabling display or rendering alone still evaluates image-only point
grids, especially for dynamic wakes.

## Decision

- Add opt-in `ret_dict=True` to Downwind's `calc_farm`, `calc_points` and
  `calc_outputs`. Preserve existing defaults, keys and scientific payloads.
  Forward the flag through output-to-point delegation. Iterative applies it
  only to final outputs, not convergence datasets or downwind-order history.
- Keep each Python example workflow in one monolithic `main(argv=None)`.
  Reserve `-R/--results` for calculation selectors, defaulting to `farm_results`,
  and `-o/--outputs` for subsequent script-specific models. Keep `-r` for rotor
  models and `--ref_xy` for reference coordinates.
- Request keyed returns and print every key followed by its value on a new
  line at verbosity two and three. Supply the `farm_results` mapping entry,
  or `None` when absent, to subsequent outputs without dataset-specific reports.
  Run independent point grids in separate output calls.
- Skip image outputs and their point calculations under `-nf/--nofig`.
  Numerical slices, point exports, NetCDF files and required scientific input
  reductions remain enabled.
- Resolve `Output.nofig` centrally before output initialization and requirements
  preparation, including nested `OutputList` selections and Sequential animation
  plugins. Image-producing modes yield no result, not a numerical fallback.
  Keep `plot=False` and data-only layout operations numerical. Use
  `image_only=True` for numerical probes consumed exclusively by script plots.
  Examples configure the flag without type-based filtering or selector mutation.
- Preserve Sequential iteration and step-local animation plugins. Use output-only
  processing of accumulated farm data for collectors and farm outputs.

## Compatibility And Migration

Non-suppressed Python library return types, YAML/WindIO configuration, model-book keys,
dimensions, units and writer filenames remain unchanged. Maintained Python
examples and their README commands adopt the split selectors directly, without
aliases. Figure-less scripts no longer write GIF/MP4/PNG images. Existing
comparison tables and point probes remain explicitly selected outputs rather
than implicit farm-dataset consumers. Suppressed image-only selections return
an empty mapping; select numerical modes explicitly when data is required.

## Consequences

Uniform reporting retains actual grouped and non-farm keys without inferring
types. Headless examples avoid unnecessary point science. Monolithic tutorial
size is intentional; the runtime library gains no reporting or CLI machinery.
Numerical consumers still require suitable collected fields, and library errors
remain explicit rather than triggering script-level silent fallbacks.

## Verification

- [Output contracts](../../tests/2_models/test_output_contract.py) cover opt-in
  mapping shapes, default behavior, point delegation, disabled collection,
  explicit errors and Iterative convergence/history.
- [Figure suppression](../../tests/2_models/test_output_show.py) covers all
  calculation entry points, nested chains, numerical modes and companion probes;
  [Sequential tests](../../tests/2_models/test_sequential_animation_outputs.py)
  prohibit step-local point work and debug collection for suppressed animations.
- [Example contracts](../../tests/3_examples/test_output_selection.py) cover
  independent selectors, real keys, quiet modes and monolithic workflows.
- [Real example execution](../../tests/3_examples/test_examples.py) prohibits
  figures and image-only point work while retaining necessary numerical outputs.
