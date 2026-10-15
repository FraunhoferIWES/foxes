# ADR-0009: Test File Dependency Boundary

- Status: Accepted
- Date: 2026-10-15
- Supersedes: Test-input boundary in ADR-0008.

## Context

The user requested complete removal of tests that depend on repository files
outside package code, examples, notebooks, and tests across all three packages.
The FOXES suite already satisfies this broader boundary.

## Decision

Limit pytest repository-file inputs to `foxes/`, `examples/`, `notebooks/`, and
`tests/`. Tests may create and read temporary outputs. Do not introduce skips,
copied fixtures, or fallback paths for inputs outside the allowed directories.

## Consequences

Documentation, release metadata, CI configuration, and other repository files
are checked by their separate build, release, or review workflows. Existing
runtime, example, and fixture coverage remains unchanged. No production API,
naming convention, or dependency changes.

## Verification

Audit test and helper file inputs and run pre-commit for the documentation change.
