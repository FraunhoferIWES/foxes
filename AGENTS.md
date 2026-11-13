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
| Data handling and AI-tool use | [Data classification](docs/data-classification.md) |
| UI, visual design or brand assets | [UI policy](#ui-design-policy) and [project visualization conventions](.github/instructions/visualization.instructions.md) |

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

Before the first UI change, ask whether corporate or free design applies and
explain both consequences. Corporate design remains the default until recorded
otherwise. Record the choice in an ADR and
`docs/architecture.md#cross-cutting-decisions`; for free design, remove the
superseded UI guidance and record the conventions to retain. Follow the
[visualization instructions](.github/instructions/visualization.instructions.md).

## Data Classification

Use AI tools only for public or restricted data. Ask before processing
potentially confidential or unknown-provenance data, and proceed with such data
only after an explicit scoped release. Follow
[the full classification policy](docs/data-classification.md) and exception
process.

## Existing FOXES Project

FOXES is an established Python package, not a scaffold. `pyproject.toml` owns
supported runtimes, dependencies, extras and tool configuration. Use `uv`;
follow [environment setup](docs/development.md#environment-and-dependencies).

## Engineering Rules

Priority: Security > Correctness > Tests > Documentation > Modularity > Style.
Validate external input, never hardcode secrets or evaluate untrusted input,
and keep transport security enabled.
Raise contextual library exceptions; do not configure application-global
logging. For code quality, tests, documentation and model lifecycle, follow
[development](docs/development.md), [docstrings](docs/docstrings.md), and
[architecture](docs/architecture.md#model-lifecycle).

## Before-Release Rules

Follow the [before-release review](docs/development.md#before-release-review)
for the changelog, source documentation, ADRs, code and docstring checks, and
smoke-test review.

## Completion Gate

Follow the [development closure](docs/development.md#development-closure).

## Completion Report

Report **Summary**, **Code**, **Tests**, **Docs**, and **Self-review**. See the
[report checklist](docs/development.md#completion-report).
