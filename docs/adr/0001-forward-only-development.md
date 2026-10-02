# ADR-0001: Forward-Only Development And Complete Closure

- Status: Accepted
- Date: 2026-10-01
- Supersedes: None

## Context

FOXES is a mature scientific package with public Python, model-book, YAML,
command-line, and xarray contracts. Preserving every superseded contract would
accumulate parallel implementations, compatibility branches, duplicate tests,
and documentation that describes multiple eras of the package. That increases
maintenance cost and makes the intended current behavior harder to identify.

Development closure also needs one repository-wide meaning. Code without its
tests, quality gates, public docstrings, changelog entry, or current technical
records leaves future contributors and agents with an incomplete contract.

## Decision

FOXES development is strictly forward-looking by default:

- Implement the target contract directly and remove the superseded code, tests,
	documentation, examples, aliases, formats, and fallback behavior in the same
	change.
- Update all maintained in-repository callers instead of adding compatibility
	shims for them.
- Add a legacy or migration path only when the user explicitly requires a
	bounded exception. Record its scope and objective removal condition in the
	governing ADR.

A development change is complete only when:

- when code files changed, required tests are present and both focused tests and
	`uv run pytest tests` pass;
- documentation-only changes pass the applicable documentation validation
	instead of the full runtime test suite;
- `uv run pre-commit run --all-files` passes after the final edit;
- every affected public docstring is accurate;
- the final version section of `CHANGELOG.md` matches `project.version` from
	`pyproject.toml` and records the change; and
- architecture, naming, development, API, example, and other affected FOXES
	documentation is synchronized in the same change.

## Compatibility And Migration

This policy applies to new development from this decision onward. It does not
retroactively require unrelated cleanup. When a task changes an existing
contract, all maintained FOXES callers move to the new contract immediately;
the old path is not retained by default.

Downstream consumers may need to update when a public contract changes. Such an
impact must be identified and documented, but it does not itself justify legacy
code. An explicitly authorized exception states exactly what remains, for whom,
and when or under which condition it is removed.

## Consequences

- FOXES has one current implementation and one documented contract for changed
	behavior.
- Changes may require coordinated updates across code, tests, examples,
	documentation, and downstream consumers in a single development effort.
- Public changes can be immediately breaking for consumers that have not moved
	to the current contract.
- Completion takes longer than code-only delivery because validation and all
	records close together.

## Verification

- For code changes, the completion report names focused and full test results;
	for documentation-only changes, it names the applicable documentation checks.
- The final pre-commit run covers the entire repository.
- Review compares affected public docstrings and FOXES records with the final
	implementation.
- The last `CHANGELOG.md` version heading is compared with `project.version` in
	`pyproject.toml` and contains the change.

## Alternatives Considered

- Preserve compatibility shims and deprecate gradually by default. Rejected
	because it creates legacy paths as the normal case.
- Complete code first and update tests or documentation later. Rejected because
	it leaves the repository in a knowingly inconsistent state.
- Require closure checks only for releases. Rejected because stale information
	and missing coverage compound between releases.

## References

- [Repository instructions](../../AGENTS.md#development-direction)
- [Development guide](../development.md)
- [Architecture](../architecture.md#cross-cutting-decisions)
