# Repository Instructions

All contributors, including AI coding agents, must follow these rules unless an
applicable project-local instruction explicitly and justifiably overrides them.

## Instruction Hierarchy

1. This file is the canonical source of general repository rules.
2. Read applicable scoped instruction files before changing files in their scope.
3. For UI, visual-design, or brand work, read `docs/fraunhofer-design/` when that
	 directory is present. See [UI Design Policy](#ui-design-policy).
4. Use `docs/architecture.md` for durable FOXES structure and
	 `docs/naming-conventions.md` for detailed vocabulary. They supplement this
	 file; they do not override it.
5. Use `docs/docstrings.md` for public Python documentation contracts.
6. Use `docs/development.md` for FOXES setup, navigation, test selection, and
	 quality-gate commands.

## Working Protocol

- Ask before implementing when requirements are materially unclear.
- Make one focused, logical change at a time.
- Do not refactor unrelated code or add dependencies without justification.
- Deliver tests, documentation, and decision records in the same change as the
	code they describe; the sections below state when each is required.
- Self-review against these rules before presenting work as complete.

## Development Direction

- Development is strictly forward-looking by default. Implement the current
	target contract directly and remove the superseded implementation, tests, and
	documentation in the same change.
- Do not add or retain legacy branches, compatibility shims, deprecated aliases,
	dual formats, fallback behavior, or migration-only code unless the user
	explicitly requires a bounded exception and an ADR records its removal
	condition.
- Update all in-repository callers and examples to the current contract instead
	of preserving an obsolete path for them.
- Keep `docs/architecture.md`, `docs/naming-conventions.md`,
	`docs/development.md`, API documentation, examples, and other FOXES facts
	accurate in every change. Known documentation drift blocks completion.

## UI Design Policy

Ask before the first UI change which policy the project follows, and name both
consequences. Do not decide silently or mix the two.

- **Corporate design**: `docs/fraunhofer-design/` binds the project — a
	restrained, white-dominant, flat visual language, publishable as a Fraunhofer
	IWES interface without brand rework.
- **Free design**: nothing in `docs/fraunhofer-design/` applies, accessibility
	included. Publishing outside the project's own context then requires an
	extensive design and accessibility review that can force a full restyling.

Corporate design applies until the project records something else, including when
the user declines to decide or the answer stays unclear. Record the confirmed
choice in the same change: an ADR, plus a line under Cross-Cutting Decisions in
`docs/architecture.md`. A later change supersedes that ADR instead of editing it.

Free design deletes what it lifted, in the same change: the scoped UI pointers of
every agent tool in use, including `.github/instructions/ui.instructions.md`,
`docs/fraunhofer-design/`, and any token adapter, generator, or drift test the
project derived from it. Nothing then constrains UI work, so record the
conventions the project does want as a project-local instruction.

## Data Classification

Use AI tools for data classified `public` or `restricted`. Do not send
`confidential` or `strictly confidential` content to an AI tool unless the user
explicitly releases that specific scope through the exception process below.
This includes prompts, attachments, fixtures, logs, screenshots, and files an
agent reads.

| Class | Definition |
|---|---|
| `public` | Intended for the general public, no damage if shared: press releases, product catalogues, terms and conditions. |
| `restricted` | For employees and other authorized persons only; damage from disclosure or loss limited and manageable. Roughly 95% of IWES data. |
| `confidential` | Disclosure or loss would cause IWES potentially considerable damage: certain project data, customer lists. |
| `strictly confidential` | Disclosure or loss would be existentially threatening, including legal consequences: personal data requiring special protection. |

Classification follows the data, not the file it sits in.

**Ask the user for the classification** before processing when a task involves
personal data beyond work contacts; customer, partner, contract, pricing, or
funding data; unpublished, patent-relevant, or NDA-covered research data;
credentials, keys, or unremediated vulnerabilities (the fixture under
[Security](#security) excepted); or content of unknown provenance. Name the
specific data and offer an alternative — synthetic or anonymized samples,
structure without real content, or a step the user performs without an AI tool.
Never guess a class, silently redact, or proceed with the parts that look
harmless.

An explicit release by the user is valid: proceed, record it in
`docs/data-classification-exceptions.md` in the same change, name that entry in
the completion report, and record the resulting data boundary in
`docs/architecture.md` when the software itself will process such data. A release
covers only the scope it names.

## Existing FOXES Project

This repository contains the established FOXES Python package, not an
application template or empty scaffold. `pyproject.toml` is authoritative for
the supported Python versions, dependencies, extras, and tool configuration.
Use `uv` for local Python workflows. The default environment command is
`uv sync --extra dev --extra test --extra mpi --extra shp --upgrade`.

Read `docs/architecture.md` for package boundaries, runtime contracts, and
extension points. Read `docs/naming-conventions.md` for FOXES vocabulary, common
type annotations, dimensions, variables, and naming rules.

## Priority

Security > Correctness > Tests > Documentation > Modularity > Style

## Engineering Rules

### Code Quality

- Prefer idiomatic, readable code over cleverness.
- Handle errors explicitly. Do not use empty catch blocks or swallow exceptions.
- Keep functions under 40 lines and files under 700 lines where practical;
	justify exceptions.
- Keep one module focused on one concern.

### Tests

Tests are required for changes to logic, including conditionals, transformations,
validation, state, side effects, and public behavior.

- Use pytest and the established test structure described in
	`docs/development.md`.
- Cover a happy path, an error path, and a relevant edge case.
- Keep tests deterministic, isolated, fast, and delivered with the change.

Pure styling, static text, and documentation-only restructuring do not require
new runtime test cases or the full runtime test suite. Documentation validation
and the remaining completion gate still apply.

### Documentation

Update documentation when a public API, CLI option, configuration value,
dependency, setup process, or build/deployment process changes. Document public
Python APIs according to `docs/docstrings.md`. Examples must be runnable. Review
and update every affected public docstring in the same change. Keep all
architectural, naming, workflow, and other FOXES information synchronized with
the implementation at all times.

### Decision Records

Keep `docs/architecture.md` and `docs/naming-conventions.md` current.

When a human or AI establishes, selects, changes, or supersedes a durable
decision, update the relevant record in the same change:

- Update `docs/architecture.md` for system context, selected technologies,
	module ownership, public contracts, data ownership, integrations, security,
	operational practices, and cross-cutting decisions.
- Update `docs/naming-conventions.md` for agreed domain vocabulary and stable
	naming rules for code, files, contracts, configuration, and identifiers.
- Create or update an ADR for consequential architectural or naming decisions.

Record confirmed decisions, not alternatives, guesses, or placeholders. Ask a
focused question when the decision is materially unclear.

### Security

- Never hardcode secrets or credentials.
- Do not put `confidential` or `strictly confidential` content into prompts,
	code, fixtures, or documentation without the explicit release described under
	[Data Classification](#data-classification).
- Do not evaluate untrusted input.
- Validate external input at system boundaries.
- Do not disable transport security in optional network integrations.

### Diagnostics

FOXES is a library. Raise contextual exceptions and use its established
verbosity/progress mechanisms; do not configure application-global logging from
library code. Failures must be diagnosable without code changes.

### Technical Debt

Record intentional compromises as `TODO(minor): <specific, actionable
description>`. Never use vague deferred-work markers.

## Completion Gate

A development change is complete only when all of the following hold:

1. If code files changed, required tests were added or updated, focused tests
	pass, and `uv run pytest tests` passes. Documentation-only changes run the
	applicable documentation checks instead of the full runtime suite.
2. `uv run pre-commit run --all-files` passes after the final edit.
3. Every affected public function and class has an accurate NumPy-style
	docstring following `docs/docstrings.md`; unchanged docstrings were reviewed
	for continued correctness.
4. Architecture, naming conventions, development guidance, API documentation,
	examples, and any other FOXES information affected by the change are current.
5. The last version section of `CHANGELOG.md` is updated for the change and its
	heading is exactly `## v<version>`, where `<version>` is the current
	`project.version` from `pyproject.toml`.

Notebook execution also passes when notebooks change. A Sphinx build is not part
of default closure. Do not report completion with a knowingly stale record or
defer one of the required closure steps to a follow-up change.

## Completion Report

When reporting an implementation, state:

1. **Summary**: what changed and why.
2. **Code**: implementation details, including any size guideline exceeded and
	the justification for it.
3. **Tests**: coverage and results, or why tests were not needed.
4. **Docs**: documentation changes, or `N/A`.
5. **Self-review**: confirmation that the rules above hold, any declared
	deviation, and any data-classification exception recorded in
	`docs/data-classification-exceptions.md`.
