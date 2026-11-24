# ADR-0045: Corporate Documentation Design

- Status: Accepted
- Date: 2026-11-20
- Supersedes: Documentation styling in [ADR-0030](0030-free-design.md).

## Context

The project-wide free-design decision removed the Fraunhofer documentation
theme, leaving Sphinx-Immaterial's blue header. The user explicitly selected
Fraunhofer corporate design for Sphinx documentation only, with the additional
requirement that no existing font sizes become smaller.

## Decision

Restore the [Fraunhofer IWES guide](../fraunhofer-design/ui-guidelines.md) and
its authoritative [design tokens](../fraunhofer-design/design-tokens.json).
Sphinx loads `iwes-tokens.css` before `iwes.css` from `docs/source/_static/`.
The existing Sphinx-Immaterial theme continues to own navigation, search and
responsive behavior.

Use the approved primary green (`#179c7d`) for the header with black controls
and text. Keep white-dominant, flat surfaces and the corporate typography,
focus indicators and reduced-motion behavior.

Map the complete primary/accent palette, including light/dark variants and
highlight surfaces, rather than only the header color. The mobile repository
panel shares the green header. Search, API badges, syntax highlighting,
keyboard hints, footer links and every notice variant use the existing semantic
tokens. Information uses the approved steel blue, warnings and errors retain
their labelled feedback colors, and no default indigo or theme-specific notice
colors remain. Existing token review flags still apply.
Version notices reuse the standard notice layout: a single opaque tinted
title and a white body, without a second tinted container or extra outer
padding. Added versions use light-green title backgrounds and dark-green
icons; changed versions use light steel-blue title backgrounds and steel-blue
icons. Both retain explicit text labels.
MyST-NB's documentation text-cell frames, standard/error output surfaces,
scrollbars and progress indicators use these same tokens rather than changing
independently with the device's dark-mode preference. Notebook sources and
scientific figures are unchanged.

Preserve or increase existing font sizes. In particular, keep the theme's 25px
second-level headings rather than applying the smaller 24px token. Body text
has a 16px minimum, navigation becomes at least 16px, and code has a 14px
minimum. Preserve the theme's responsive root font size and relative type
scale on wide screens rather than capping typography at those minima. Token
values and existing `needs-review` flags are not changed or implicitly approved.

Select a custom `iwes` primary and accent palette backed by the token adapter,
not the theme's built-in white or teal palettes. Local Sphinx builds disable
notebook execution by default. GitHub Pages builds explicitly select
`nb_execution_mode=auto` to render outputs from the cleared source notebooks;
execution failures stop publication. Notebook validation remains available
separately through `nbmake`.

Scientific plots, animations, notebooks and embedded scientific figures retain
free design. This decision does not restore project-wide corporate styling.

## Compatibility And Migration

This changes documentation presentation only. Python APIs, calculations,
YAML/WindIO contracts, model-book names and result formats remain unchanged.
Restore the documentation adapter directly; do not introduce a second theme
or an optional legacy styling mode.

## Consequences

- Documentation regains its Fraunhofer identity and readable green header.
- Typography does not shrink relative to the unstyled theme.
- Documentation and scientific-figure styling have explicit, separate scopes.
- Existing token review flags remain visible in the authoritative source.
- No new runtime or documentation dependencies are required.

## Verification

- Compare every adapter value with the resolved source token.
- Validate CSS loading with a bounded Sphinx theme build, without AutoAPI
  generation or notebook execution.
- Compare computed font sizes before and after styling and check the header,
  focus contrast and viewport overflow at 320, 768 and 1280 CSS pixels, plus
  wide-screen typography at 2048 CSS pixels.
- Audit computed colors, including pseudo-elements, with the mobile drawer
  open and search, hover, focus, selected navigation, all notice types and code
  highlighting exercised. Check text contrast against each painted surface.
- Run repository pre-commit and check the restored source files explicitly.

## Alternatives Considered

- Change only the header color: rejected because the user requested the
  corporate guide rather than free design with a green header.
- Restore corporate styling project-wide: rejected because the selected scope
  is Sphinx documentation only.
- Restore the smaller 24px second-level heading: rejected by the user's
  no-font-reduction requirement.
