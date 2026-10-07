# ADR-0004: Sphinx Design-Token Adapter

- Status: Accepted
- Date: 2026-10-07
- Supersedes: None

## Context

[ADR-0002](0002-corporate-design.md) applies Fraunhofer corporate design to
FOXES documentation and requires a future token adapter to record its ownership
and synchronization mechanism before implementation.

The Sphinx documentation uses Sphinx-Immaterial for navigation, search, and
responsive structure. Replacing that infrastructure is unnecessary, but its
default visual language does not consistently apply the authoritative IWES
colours, typography, spacing, flat geometry, and accessibility requirements.

## Decision

`docs/fraunhofer-design/design-tokens.json` remains the authoritative token
source. `docs/source/_static/iwes-tokens.css` is the Sphinx technology adapter
and exposes every token as an `--iwes-*` custom property.

`docs/source/_static/iwes.css` owns the Sphinx-Immaterial visual overrides and
uses only adapter values for IWES design decisions. Sphinx loads the adapter
before the theme layer through `docs/source/conf.py`. The documentation remains
white-dominant and flat and does not expose a dark colour scheme.

Sphinx-Immaterial continues to own documentation navigation, search, and
responsive behavior. The adapter does not introduce a FOXES application
frontend or a runtime dependency.

`tests/0_consistency/test_docs_design_tokens.py` resolves token references and
compares every source token with the CSS adapter. A token or adapter change is
incomplete until that test passes.

## Compatibility And Migration

This decision changes generated HTML presentation only. It does not change
Python, YAML, WindIO, model-book, dimension, variable, result, or source-RST
contracts. Existing approved FOXES and Fraunhofer assets remain unmodified.

## Consequences

- Documentation styling has one explicit technology boundary and one
  authoritative source of visual values.
- Token additions and value changes require a synchronized adapter update;
  drift fails the permanent consistency test.
- Sphinx-Immaterial upgrades require a rendered documentation check because
  upstream selectors can change.
- The repository keeps its existing documentation stack and gains no browser
  application build or runtime dependency.

## Verification

- `tests/0_consistency/test_docs_design_tokens.py` enforces token-adapter parity.
- A Sphinx HTML build with notebook execution disabled validates static-file
  loading and theme integration.
- Rendered checks at 320, 768, and 1280 pixels cover responsive layout, focus
  visibility, colour application, flat components, and horizontal overflow.

## Alternatives Considered

- Replace Sphinx-Immaterial with a custom theme. Rejected because navigation,
  search, and responsive behavior already meet the documentation needs.
- Hard-code IWES values only in component CSS. Rejected because duplicated
  values would have no complete synchronization boundary.
- Add a token-generation build step. Rejected because the compact adapter and
  drift test provide deterministic synchronization without another toolchain.

## References

- [ADR-0002](0002-corporate-design.md)
- [Architecture](../architecture.md#cross-cutting-decisions)
- [Fraunhofer IWES UI guidelines](../fraunhofer-design/ui-guidelines.md)
- [Fraunhofer IWES design tokens](../fraunhofer-design/design-tokens.json)
