# ADR-0033: Coding-Style Review At Development Closure

- Status: Accepted
- Date: 2026-11-12
- Supersedes: None

## Context

FOXES already requires coding conventions, tests and repository-wide
pre-commit at closure. Automated linting and formatting do not establish that
names, logical blocks, comments, docstrings or code organization are clear.
The user requested an explicit coding-style check in standard closing tasks.

## Decision

- Add an explicit coding-style review to
  [development closure](../development.md#development-closure), alongside the
  existing automated checks.
- Review changed blocks and directly affected context in every changed code
  file against the code-quality, naming and docstring guides.
- Correct in-scope violations and justify retained size exceptions. Do not use
  this gate to require unrelated cleanup.
- Repeat the affected review after later edits, including changes made by
  automated hooks.
- State the result under **Self-review** in the completion report.

## Compatibility And Migration

This adds a contributor closure task under
[ADR-0001](0001-forward-only-development.md). Public APIs, runtime behavior,
dependencies and existing test/pre-commit commands are unchanged.

## Consequences

- Passing linters alone is no longer sufficient evidence of style compliance.
- Closure includes a scoped human or agent review, not a new heuristic source
  checker or repository-wide refactoring requirement.

## Verification

Confirm that the repository instructions, development closure, completion-report
checklist and architecture decision index describe the same review obligation.
Documentation-only updates use documentation checks rather than pytest inputs.

## Alternatives Considered

- Rely only on linting and formatting: insufficient for readability and
  semantic naming.
- Add blanket automated comment or size rules: unable to judge clarity or
  justified exceptions reliably.

## References

- [Repository completion gate](../../AGENTS.md#completion-gate)
- [Code quality](../development.md#code-quality)
- [Completion report](../development.md#completion-report)
