# ADR-0008: Documentation-Independent Tests

- Status: Accepted
- Date: 2026-10-13
- Supersedes: The pytest token-parity requirement in ADR-0004.

## Context

Documentation-dependent tests require additional source files in isolated
package test environments. The user requested their complete removal across
FOXES, FOXES Optimization, and iwopy.

## Decision

Pytest tests must not read or depend on files under `docs/`. Remove the Sphinx
design-token test rather than skip it or substitute another docs-dependent test.
Retain corporate design, its token adapter, and separate documentation builds
and review.

## Consequences

Runtime tests need no documentation fixtures. Token parity is reviewed outside
pytest. Runtime behavior and the visual design are unchanged.

## Alternatives

Copying documentation into package test sandboxes or conditionally skipping
documentation tests retains the dependency and was rejected.

## Verification

Run the full pytest suite with documentation-file reads blocked, plus pre-commit.
