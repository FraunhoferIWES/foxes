# ADR-0002: Corporate Design

- Status: Accepted
- Date: 2026-10-02
- Supersedes: None

## Context

FOXES has no browser frontend or design-token adapter, but it exposes visual
surfaces through matplotlib plots and animations, examples, notebooks, Sphinx
documentation, and project or brand assets.

Repository policy requires an explicit choice between Fraunhofer corporate
design and free design before visual work. Corporate design is the default when
no different choice is recorded, and the repository already contains scoped UI
instructions, design tokens, and FOXES-specific visual guidance.

## Decision

FOXES follows the Fraunhofer corporate design defined in
`docs/fraunhofer-design/`.

The policy applies to scientific plots and animations, examples, notebooks,
documentation, and brand assets. Plotting APIs preserve caller-supplied axes,
styles, colours, labels, and output options unless an API explicitly promises a
corporate preset. Importing FOXES must not mutate global matplotlib state or
require an unavailable corporate font.

FOXES currently has no design-token adapter. A future browser interface, shared
plotting theme, or token adapter is a new architectural feature and must record
its ownership and synchronization mechanism before implementation.

## Compatibility And Migration

This decision does not change Python, YAML, WindIO, model-book, dimension,
variable, or result contracts. Existing visual surfaces adopt the current
corporate guidance directly; no compatibility path or migration format is
introduced.

## Consequences

- Visual changes use the approved tokens, chart sequence, typography, image,
	logo, and accessibility guidance where those rules apply.
- Scientific meaning, caller composition, labels, units, and non-colour
	distinctions remain part of public plotting contracts.
- Non-visual wake-modelling and numerical code does not acquire web-component
	or styling requirements.
- Choosing free design later requires a superseding ADR and removal of the
	corporate-design instructions and assets in the same change.

## Verification

- Validate this ADR and its local references with the documentation checks.
- Run repository-wide pre-commit after the final documentation edit.
- For future visual changes, validate the affected plot, animation, notebook,
	documentation, or asset according to the UI completion checklist.

## Alternatives Considered

- Free design. Rejected because no project-specific free-design system or
	accessibility policy has been selected, and corporate design is the repository
	default.
- Leave the choice implicit. Rejected because contributors and coding agents
	would lack a durable decision for plotting, documentation, and brand changes.
- Add a browser stack or token adapter now. Rejected because FOXES has no such
	interface and recording the visual policy does not justify new runtime code.

## References

- [Repository instructions](../../AGENTS.md)
- [Architecture](../architecture.md)
- [Fraunhofer IWES UI guidelines](../fraunhofer-design/ui-guidelines.md)
- [Fraunhofer IWES design tokens](../fraunhofer-design/design-tokens.json)
