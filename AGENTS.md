# Repository Instructions

All contributors, including AI coding agents, must follow these rules unless an
applicable project-local instruction explicitly and justifiably overrides them.

## Instruction Hierarchy

This file is canonical. Read applicable scoped instructions before editing.
The linked guides below are required for relevant work; they elaborate these
rules without overriding them.

| Work | Required guide |
|---|---|
| Structure, ownership, model lifecycle and parallelization | [Architecture](docs/architecture.md) |
| Vocabulary, symbols, dimensions and types | [Naming conventions](docs/naming-conventions.md) |
| Public Python documentation | [Docstrings](docs/docstrings.md) |
| Setup, code quality, tests, decisions and closure | [Development](docs/development.md) |
| UI, visual design or brand assets | [UI policy](#ui-design-policy) and `docs/fraunhofer-design/` when present |

## Working Protocol

- Ask before implementing when requirements are materially unclear.
- Make one focused, logical change at a time.
- Do not refactor unrelated code or add dependencies without justification.
- Deliver required tests, documentation, and decision records with the code.
- Self-review against these rules before presenting work as complete.

## Development Direction

Implement the target contract directly; remove superseded code, tests and docs,
and update all in-repository callers and examples in the same change. Do not
retain legacy branches, shims, aliases, dual formats, fallbacks or migration-only
code without an explicit user-required bounded exception and an ADR removal
condition. See [forward-only development](docs/development.md#forward-only-development).

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

FOXES is an established Python package, not a scaffold. `pyproject.toml` owns
supported runtimes, dependencies, extras and tool configuration. Use `uv`;
follow [environment setup](docs/development.md#environment-and-dependencies).

## Priority

Security > Correctness > Tests > Documentation > Modularity > Style

## Engineering Rules

### Code Quality

Write idiomatic, human-readable code; start logical blocks with brief one-line
comments and separate them with blank lines. Follow the
[code-quality rules](docs/development.md#code-quality) for size, modularity,
error handling and technical-debt markers, and the [docstring standard](docs/docstrings.md).

### Model Parallelization

For every model, treat `self` and its referenced objects as read-only between
`set_running()` and `unset_running()`. Keep runtime state in data containers,
chunk stores or result mappings; follow the
[lifecycle and ownership contract](docs/architecture.md#model-lifecycle).

### Tests

Logic changes require deterministic, isolated, fast pytest coverage of happy,
error and edge paths. Follow the [test structure](docs/development.md#test-structure)
and [closure requirements](#completion-gate); documentation-only changes use
documentation checks instead of runtime tests.

### Documentation

Keep affected public docstrings, API/CLI/configuration docs, dependencies,
workflows, architecture, naming and runnable examples current in the same change.
Known drift blocks completion. Record confirmed durable decisions in their
owning guide and consequential decisions in an ADR; follow
[documentation and decision records](docs/development.md#documentation-and-decision-records).

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

## Completion Gate

Follow [development closure](docs/development.md#development-closure):
focused tests and `uv run pytest tests` pass for code changes, applicable checks
pass for docs-only changes, and `uv run pre-commit run --all-files` passes after
the final edit. Review all affected public docstrings and records; update the
last changelog section headed exactly `## v<project.version>`. Execute changed
notebooks; Sphinx is not a default gate. Do not defer required closure steps or
report completion with stale records.

## Completion Report

Report **Summary**, **Code** (including justified size exceptions), **Tests**,
**Docs**, and **Self-review** (rule compliance, deviations and named
data-classification exceptions). See the
[report checklist](docs/development.md#completion-report).
