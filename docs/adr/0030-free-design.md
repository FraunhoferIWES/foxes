# ADR-0030: Free Design

- Status: Accepted
- Date: 2026-11-11
- Supersedes: [ADR-0002](0002-corporate-design.md),
  [ADR-0004](0004-sphinx-design-token-adapter.md), and the corporate-design
  retention requirement of [ADR-0008](0008-documentation-independent-tests.md).

## Context

While commissioning a user-oriented output notebook, the user explicitly chose
free design instead of the existing corporate-design policy. The choice applies
to the project, not just one notebook.

## Decision

FOXES uses free design for scientific plots, animations, notebooks,
documentation and visual assets. Corporate visual and accessibility rules do
not apply. Remove the corporate scoped instruction, source guidance and token
document, and both Sphinx CSS adapter/theme files. Sphinx-Immaterial continues
to supply documentation navigation and its ordinary visual defaults.

The [project visualization instructions](../../.github/instructions/visualization.instructions.md)
record the intended local conventions: ordinary scientific plotting defaults,
explicit physical meaning, caller-owned styling, and bounded, reproducible
tutorials. There is no mandatory palette, font or replacement token system.

## Compatibility And Migration

This changes visual policy and generated documentation appearance, not Python,
YAML, WindIO, model-book, dimension, variable or result contracts. Remove
corporate-style loading from the Sphinx configuration and scoped pointers from
the repository instructions used by Copilot and Claude.

Superseded ADRs remain unchanged historical records. References in those
records describe former resources that are deliberately removed by this
decision; they are not current implementation pointers. Documentation-independent
pytest boundaries from ADR-0008 and ADR-0009 remain in force.

## Consequences

- Visual work has no corporate-design approval or token-parity requirement.
- Publishing outside the project's own context requires an extensive design
  and accessibility review, which may require full restyling.
- Sphinx retains its existing theme dependency without a FOXES CSS adapter.
- Existing scientific APIs, numerical behavior and caller-supplied styles are
  preserved. No new dependencies are introduced.

## Verification

- Execute the output tutorial notebook against public/synthetic inputs.
- Check that active instructions and Sphinx configuration no longer reference
  removed corporate resources.
- Run documentation checks and repository-wide pre-commit after the final edit.

## Alternatives Considered

- Retain corporate design for documentation only. Rejected because the user
  selected free design for the project rather than a mixed policy.
- Keep corporate assets as optional advice. Rejected because the selected
  policy requires removing the corporate bindings in the same change.
