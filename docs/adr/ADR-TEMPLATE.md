# ADR-NNNN: Short Decision Title

- Status: Proposed | Accepted
- Date: YYYY-MM-DD
- Supersedes: None | [ADR-NNNN](NNNN-title.md)

When a later ADR supersedes this one, leave this file unchanged. The replacement
record declares the relationship, and the ADR index points readers to it.

## Context

Describe the problem, constraints, and forces that make this decision necessary.
Link related ADRs, contracts, or evidence where useful.

For a FOXES contract, state the current behavior and affected package owners.
Call out relevant dimensions and `FV` variables, model lifecycle stages,
model-book names/factories, engine backends, public xarray results, supported
Python versions, and optional integrations rather than referring vaguely to
"the data" or "the model".

## Decision

State the decision clearly and concretely.

Name the owner module, public contract, and enforcement point. Separate required
behavior from an example implementation.

## Compatibility And Migration

Describe effects on Python callers, YAML/WindIO inputs, model-book names,
factory templates, dimensions and result variables, serialized files, examples,
and downstream packages such as `foxes-opt`. FOXES is forward-only by default
under [ADR-0001](0001-forward-only-development.md): describe how maintained
callers move to the target contract in the same change. Include a legacy path or
deprecation period only when the user explicitly authorizes a bounded exception;
record its removal condition.

## Consequences

- Positive consequence
- Negative consequence
- Performance, memory, parallelism, or optional-dependency consequence
- Follow-up work, risks, or limitations

## Verification

List the focused tests, cross-engine/numerical comparisons, documentation build,
or compatibility checks that demonstrate the decision. Link permanent tests,
not transient terminal output.

## Alternatives Considered

- Alternative A and why it was rejected
- Alternative B and why it was rejected

## References

- Related ADRs, `docs/architecture.md`, `docs/naming-conventions.md`, public API
	documentation, issues, or evidence
