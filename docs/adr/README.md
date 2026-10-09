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
| [0002](0002-corporate-design.md) | Accepted | Apply Fraunhofer corporate design to FOXES plots, animations, documentation, notebooks, and brand assets. |
| [0003](0003-compact-static-target-coordinates.md) | Accepted | Keep static target coordinates compact with a singleton state axis and broadcast them inside engine runners. |
| [0004](0004-sphinx-design-token-adapter.md) | Accepted | Adapt authoritative IWES design tokens to Sphinx CSS and prevent token drift with a consistency test. |
| [0005](0005-static-output-grid-coordinates.md) | Accepted | Store output slice grids without a state axis and pass them through the static-point calculation contract. |
| [0006](0006-micro-reference-disk-averages.md) | Accepted | Average micro reference disks during loading with native support points and vector wind means. |

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
