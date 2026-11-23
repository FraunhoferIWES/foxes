# Fraunhofer IWES UI Guidelines

## Applicability

This document binds FOXES Sphinx documentation under
[ADR-0045](../adr/0045-corporate-documentation-design.md). Scientific plots,
animations, notebooks and other visual assets remain under the free-design
policy in [ADR-0030](../adr/0030-free-design.md).

### FOXES Scope

FOXES has no browser application frontend. Sphinx documentation uses
`docs/source/_static/iwes-tokens.css` as its design-token adapter and
`docs/source/_static/iwes.css` as its theme layer. Apply this guide to
documentation navigation, text, tables and interface assets. Scientific figures
embedded in the documentation retain their free-design conventions and
caller-supplied styling. Do not apply web component requirements to numerical
code. A new browser UI or shared plotting theme requires its own architectural
decision.

## Purpose and Authority

This document is the canonical policy for user interfaces, visual design, and
brand use in Fraunhofer IWES projects that follow the corporate design.
`design-tokens.json` is the authoritative source for approved design values.

Apply requirements in this order:

1. Accessibility
2. Corporate design
3. Consistency
4. Performance
5. Preference

Do not introduce or alter approved tokens, fonts, logo assets, icon libraries,
or visual directions without design approval. Map approved tokens to the chosen
technology without changing their values.

**Reading token values.** Once the project has a token adapter for its technology
and has recorded it in `docs/architecture.md`, read values from that adapter. It
resolves consumable token references. Verify parity against `design-tokens.json`
when changing an adapter; documentation adapters are reviewed outside pytest.
Parity does not approve values marked for review: `design-tokens.json` remains
authoritative for review status. Read it when no adapter exists, when adding,
changing, or reviewing tokens, or when a value is absent from the adapter.

## Visual Direction

Interfaces are white-dominant, spacious, flat, geometric, and restrained in
their use of green. Use minimal, purposeful motion. Never convey meaning through
colour alone; pair it with text, a label, or an icon.

## Colour and Accessibility

- Use semantic tokens for interface decisions and primitive tokens only when
  defining a semantic mapping.
- Use approved tints rather than opacity to create lighter brand colours.
- Use secondary colours sparingly for highlights, typographic emphasis, or
  illustration, not as default surfaces or brand colours.
- Meet a minimum contrast ratio of 4.5:1 for normal text and 3:1 for large text
  and UI controls. The primary green does not meet AA for small text on white;
  use an approved dark text colour instead.
- Use the approved feedback-colour mappings. Their review status is recorded in
  `design-tokens.json`.
- Use no gradients except the approved brand and badge gradients.

## Typography

- Use Roboto as the primary interface font and the approved monospaced font
  stack only for code-like values and identifiers.
- Use only the approved light (300) and bold (700) weights. Do not use italics.
- Use the defined type scale, line heights, and measure. Keep heading levels in
  document order.
- Use uppercase only for extra-small badges. Use tabular figures in tables and
  numeric displays.

For FOXES documentation, the user's no-font-reduction requirement takes
precedence over smaller sizes in the type scale. Keep the theme's existing 25px
second-level headings instead of reducing them to 24px. Body text stays at 16px,
navigation increases from 14px to 16px, and code has a 14px minimum. Source token
values and their review status remain unchanged.
These are minimum sizes, not fixed limits: retain the theme's responsive root
font size and use the larger of its relative size and the corporate token.
Wide-screen typography must not shrink when the corporate theme is applied.

## Layout and Responsiveness

- Use the approved four-pixel spacing scale and prefer layout gaps to arbitrary
  margins.
- Keep regular content inside the defined type area. Full-bleed imagery is a
  deliberate exception.
- Use the approved container, content width, gutters, and breakpoints.
- Build mobile-first layouts. A four-column layout becomes two columns below the
  large breakpoint and one column below the small breakpoint.
- Provide touch targets at least as large as the approved minimum.
- Keep footer content in the approved small-text format, including the
  Fraunhofer IWES copyright and any required classification.

## Shape, Elevation, and Motion

- All approved radii resolve to zero. Use flat, rectangular geometry.
- Prefer whitespace, a border, or an alternate surface to separate content.
  Never combine a border and a shadow for the same visual separation.
- Shadows are reserved for dropdowns and modal dialogs.
- Animate only transforms and opacity. Use the approved timings and do not add
  decorative or infinite motion.
- Respect the user's reduced-motion preference.

## Interaction States and Components

Interactive controls must support default, hover, focus-visible, active,
disabled, loading, error, and selected states. Containers must also provide a
useful empty state with explanatory text and a primary action where appropriate.

- Keep a clearly visible focus indicator. Loading states must preserve layout.
- A placeholder never replaces a visible input label.
- Reuse the project's established components before creating new ones.
- Components are layout-independent; their parent owns placement and width.
- Use semantic tokens instead of locally invented colour, size, spacing, or
  font values.
- Buttons use a clear verb-and-object label. Provide at most one primary action
  in a section.
- Tables use clear headers, horizontal separators, and right-aligned tabular
  numeric values. Choose either zebra rows or row borders, not both.
- Use only approved Fraunhofer IWES icon assets. Decorative icons are hidden
  from assistive technology and never replace labels on primary actions.

## Images, Charts, and Logos

Use only approved image sources. Prefer authentic research environments with
natural expression, deliberate lighting, geometric composition, and harmonious
colours. Avoid cut-outs, distorted wide-angle imagery, cartoons, cliches, and
interchangeable stock motifs.

- Every non-decorative image has meaningful alternative text and a visible
  credit. Decorative images use empty alternative text. Preserve image dimensions
  or an explicit aspect ratio to prevent layout shifts.
- Chart series use the approved chart sequence without reordering. For
  aggregatable categories, group seven or more minor categories into an "Other"
  group. Keep individually meaningful model or case series distinct. Charts
  require a text alternative and use horizontal grid lines only.
- FOXES scientific plots name axes and include units where the represented
  quantity has them. Legends distinguish model/case names, and line style,
  marker, label, or direct annotation reinforces colour when series must be
  compared. Wind-direction and circular data must retain their scientific
  convention rather than being cosmetically reordered.
- Documentation and notebook plots provide a nearby caption or prose summary of
  the conclusion. Tests for plotting logic assert data, artists, labels, limits,
  or returned objects rather than brittle whole-image pixels unless exact visual
  regression is the behavior under test.
- Use supplied logo assets exactly as provided. Do not redraw, recolour,
  distort, crop, rotate, or manually invert them. Maintain clear space and the
  approved minimum size. Use the picture mark alone only for a favicon or social
  avatar. Missing assets require a labelled placeholder and delivery note.

## Dark Mode

Dark mode is not enabled by default. Implement it only when explicitly required
and approved. Use token-level overrides and the approved white logo asset; never
make uncoordinated component-level dark-mode changes.

## UI Completion Checklist

- [ ] Only approved tokens, fonts, icons, images, and logo assets are used
- [ ] Text and interface contrast meet the required ratio
- [ ] Keyboard operation and visible focus are supported
- [ ] Required interaction and empty states are implemented
- [ ] Layout is responsive without horizontal scrolling at 320, 768, and 1280px
- [ ] Reduced motion is respected
- [ ] Images have appropriate alternative text and visible credits
- [ ] Charts include a text alternative and use the approved colour sequence
- [ ] Logo clear space, copyright, and required footer information are present
- [ ] Any design-token value requiring review is clearly flagged
