# Architecture Decision Records

This directory stores durable records of significant architectural decisions for
FOXES. The current implementation and architecture record predate this ADR
directory; do not invent retrospective decisions. Add records when a new change
requires a durable choice.

## Index

Keep one line per record so a reader can pick the relevant ADR without opening
all of them.

| ADR | Status | Decision |
|---|---|---|
| [0001](0001-forward-only-development.md) | Accepted | Development is forward-only by default; code changes require tests, and every change requires pre-commit, current docstrings, a current-version changelog entry, and synchronized FOXES documentation. |
| [0002](0002-corporate-design.md) | Superseded by [0030](0030-free-design.md) | Apply Fraunhofer corporate design to FOXES plots, animations, documentation, notebooks, and brand assets. |
| [0003](0003-compact-static-target-coordinates.md) | Accepted; superseded by [0013](0013-full-shaped-broadcast-data.md) | Keep static target coordinates compact with a singleton state axis and broadcast them inside engine runners. |
| [0004](0004-sphinx-design-token-adapter.md) | Superseded by [0030](0030-free-design.md); pytest requirement previously superseded by [0008](0008-documentation-independent-tests.md) | Adapt authoritative IWES design tokens to Sphinx CSS. |
| [0005](0005-static-output-grid-coordinates.md) | Accepted; internal target storage superseded by [0013](0013-full-shaped-broadcast-data.md) | Store output slice grids without a state axis and pass them through the static-point calculation contract. |
| [0006](0006-micro-reference-disk-averages.md) | Accepted | Average micro reference disks during loading with native support points and vector wind means. |
| [0007](0007-smooth-power-mask-boosts.md) | Accepted | Smooth near-rated power-mask boosts and saturate low-induction thrust at the Betz optimum while retaining requested power. |
| [0008](0008-documentation-independent-tests.md) | Accepted; corporate-design retention superseded by [0030](0030-free-design.md) | Keep pytest independent of documentation files and review documentation separately. |
| [0009](0009-test-file-dependency-boundary.md) | Accepted; supersedes the test-input boundary in 0008 | Limit pytest repository-file inputs to package code, examples, notebooks, and tests. |
| [0010](0010-chunk-based-farm-outputs.md) | Accepted | Adopt outputs through the core chunk lifecycle; isolate farm writer payloads and split files by exact counts or calendar periods, including ISO weeks. |
| [0011](0011-engine-output-execution-and-progress.md) | Accepted; supersedes the engine/algorithm scope restriction in 0010 | Unify engine output execution, ordered batches, point collection and parent-owned adaptive progress without changing default farm fields. |
| [0012](0012-chunk-animation-output.md) | Accepted | Adopt optional flow animations with shared point grids, target-data reuse, bounded scientific blocks and finalization-only assembly/writing. |
| [0013](0013-full-shaped-broadcast-data.md) | Accepted; supersedes [0003](0003-compact-static-target-coordinates.md) | Represent repeated dimensioned data as full-shaped read-only broadcast views and broadcast mismatched `Model.get_data()` results automatically. |
| [0014](0014-sequential-animation-outputs.md) | Accepted | Adapt flow-animation outputs to Sequential with step-local point evaluation and completion-only shared rendering. |
| [0015](0015-animation-helper-ownership.md) | Accepted | Keep animation composition in adopted outputs and all Sequential plugins in `foxes.models.seq_plugins`. |
| [0016](0016-state-turbine-performance-evaluation.md) | Accepted; registration entry point superseded by [0022](0022-output-selected-turbine-evaluation.md), zero-division semantics by [0025](0025-turbine-evaluation-zero-division.md) | Keep turbine performance metrics in state-turbine data, annualize unweighted power and require explicit uncertainty for P75/P90. |
| [0017](0017-constructor-defined-output-contract.md) | Accepted; evaluation scope superseded by [0022](0022-output-selected-turbine-evaluation.md), remaining adoption/YAML scope by [0024](0024-complete-output-adoption.md) | Promote animation dependencies to constructor-defined output models, keep scientific data run-local and preserve algorithm string/list selectors. |
| [0018](0018-slice-first-output-hierarchy.md) | Accepted | Derive flow animations and static plots from numerical slice data, with neutral point-grid collection and shared-data keys. |
| [0019](0019-reduced-output-payloads.md) | Accepted; evaluation scope superseded by [0022](0022-output-selected-turbine-evaluation.md) | Eliminate unnecessary output dimensions on workers and merge sufficient statistics, histograms and selected layout data without reconstructing state frames. |
| [0020](0020-output-display-control.md) | Accepted; animation defaults and test policy superseded by [0021](0021-noninteractive-animation-and-tests.md) | Default plotting constructors to no display, enable model-book plotting display and keep blocking animation playback parent-owned and run-local. |
| [0021](0021-noninteractive-animation-and-tests.md) | Accepted | Keep animation presets noninteractive and prohibit enabled display in tests, including mocked checks; static plotting presets remain interactive. |
| [0022](0022-output-selected-turbine-evaluation.md) | Accepted; deferred aggregation superseded by [0027](0027-domain-result-aggregation.md) | Apply turbine metrics through output selection; remove competing evaluation/aggregation outputs and keep unrestricted collectors limited to available fields. |
| [0023](0023-in-process-example-runs.md) | Accepted | Run every example once through callable mains/Python APIs, with isolated cleanup and independent bounded-input/headless choices. |
| [0024](0024-complete-output-adoption.md) | Accepted | Complete output adoption, reduce arbitrary-point means on workers, require explicit curve scans and retain direct YAML-only Matplotlib calls as an explicit exception. |
| [0025](0025-turbine-evaluation-zero-division.md) | Accepted | Return zero for zero-power efficiency, including zero ambient power, and positive infinity for nonzero power over zero ambient power, without division warnings. |
| [0026](0026-analogous-point-result-outputs.md) | Accepted; module ownership superseded by [0027](0027-domain-result-aggregation.md) | Mirror farm collectors and count/chunk/calendar writers for point results, with independent default/all wind-field sets and engine-owned target chunking. |
| [0027](0027-domain-result-aggregation.md) | Accepted; grouped model-book naming superseded by [0029](0029-grouped-output-preset-families.md), native-unit output policy by [0032](0032-output-dataset-units.md), collector keys by [0034](0034-aggregation-output-keys.md) | Separate domain collectors from all domain writers; aggregate physical fields using worker/file-interval statistics and independently normalized entity weights. |
| [0028](0028-aggregation-presets-and-provenance.md) | Accepted; grouped selector grammar superseded by [0029](0029-grouped-output-preset-families.md) | Register every aggregation/filter/field/writer combination, retain scalar pre-reduction counts, and share reductions with YAML/WindIO while keeping flow files gridded. |
| [0029](0029-grouped-output-preset-families.md) | Accepted; all-turbine naming superseded by [0031](0031-total-output-preset-family.md), dataset keys by [0034](0034-aggregation-output-keys.md) | Imply named turbine aggregation through farms_results/cluster_results preset families without changing constructor options or dataset keys. |
| [0030](0030-free-design.md) | Accepted | Use free design, remove corporate guidance and Sphinx token adapters, and require separate design/accessibility review before external publication. |
| [0031](0031-total-output-preset-family.md) | Accepted; dataset keys superseded by [0034](0034-aggregation-output-keys.md) | Use total_results for all-turbine aggregation presets, including every filter/field/mean/writer variant, without changing constructor options or dataset keys. |
| [0032](0032-output-dataset-units.md) | Accepted | Attach scientific units to output variables and convert power to the smallest turbine-type unit before reduction, preserving native internal solver history. |
| [0033](0033-coding-style-closure-review.md) | Accepted | Require an explicit coding-style review of changed code as a standard closure task, separate from linting and formatting, and report its result. |
| [0034](0034-aggregation-output-keys.md) | Accepted; custom collector naming superseded by [0035](0035-fixed-collector-keys.md), shared total keys by [0036](0036-domain-total-keys.md) | Use aggregation-specific `FC` constants for farm and point datasets, with shared keys for totals and explicit overrides for compositions. |
| [0035](0035-fixed-collector-keys.md) | Accepted; shared total keys superseded by [0036](0036-domain-total-keys.md) | Fix farm and point collector keys by aggregation and reject duplicates instead of allowing custom names. |
| [0036](0036-domain-total-keys.md) | Accepted; supersedes shared farm/point total keys in 0034 and 0035 | Use distinct farm and point total keys, including state means, so both results compose without ambiguity. |

## When To Add An ADR

Add or update an ADR when a decision has lasting consequences for contracts,
security, data ownership, module boundaries, integrations, operations, or naming
rules that are reused across the project.

For FOXES, this includes changes to:

- core dimensions, data-container responsibilities, or xarray result schemas;
- model lifecycle, model-book registration, factory-name syntax, or public model
	names;
- algorithm/engine ownership, chunking semantics, worker serialization, or
	supported execution backends;
- supported Python versions, dependency strategy, packaging, or release flow;
- public YAML/WindIO schemas, console commands, packaged-data lookup, or a
	material external integration;
- data-classification boundaries or the UI design policy; and
- terminology or type conventions used across multiple package areas.

Do not create an ADR for a local bug fix, an implementation detail contained by
an existing contract, a routine dependency patch, or a reversible test-only
choice.

## How To Add One

1. Copy `ADR-TEMPLATE.md` to `NNNN-short-kebab-case-title.md`.
2. Start numbering at `0001` and keep titles stable after merge.
3. Record context, the decision, consequences, and rejected alternatives.
4. Include compatibility, migration, and verification consequences for public
	FOXES contracts.
5. Use `Accepted` after adoption. When a later ADR replaces a decision, declare
	the old record under `Supersedes` in the new ADR. Leave the old file unchanged
	and point its index row to the replacement.
6. Add the record to the index above and update `docs/architecture.md` or
	`docs/naming-conventions.md` in the same change.

Do not create ADRs for reversible local implementation details with no durable
architectural consequence.
