# ADR-0018: Slice-First Output Hierarchy

- Status: Accepted
- Date: 2026-11-10

## Context

`SliceData` inherited grid collection from `FlowAnimation`. Consequently,
`FlowPlots2D` and `SlicesData` also inherited animation filenames, frame rates,
validation and keys such as `animation_points`, although they did not animate.

## Decision

Make `FlowAnimation` derive from `SliceData`, alongside `FlowPlots2D` and
`SlicesData`. A private `PointDataOutput` base holds the shared numerical
point/grid lifecycle, bounded point evaluation and chunk assembly without
rendering configuration. `SliceData` owns numerical formatting and reduction;
`FlowAnimation` adds only animation configuration, validation and rendering.

Use model-private `slice_points`, `slice_selection` and `slice_chunks` keys for
all slice-family outputs. Grids and state-sized selection/order arrays remain
in loaded data, and payloads stay in run-local result mappings. Request point
weights for mean reductions, not state-resolved animation data.

## Compatibility And Migration

Keep the public algorithm `outputs` string/list/instance selector and existing
animation filenames, renderers, point precedence and chunking behavior.
Static flow plots use the registered `flow_plot_<plane>_<reduction>` selectors.
Replace internal animation-labelled collection keys directly, without aliases.
Synthetic chunk callers include the slice ordering metadata loaded by the
normal output lifecycle.

## Consequences

- Static slice and flow outputs have no animation settings or validation.
- Shared-data diagnostics describe numerical slices rather than animations.
- Animation collection reuses the slice lifecycle while preserving encoding,
  selected-state order, arbitrary target handling and figure cleanup.
- The private numerical collector keeps grid and rendering concerns separate.

## Verification

Regression tests reject reversed inheritance and animation initialization for
static plots. They check printed shared-data keys, all planes, exact target
reordering, state-resolved data and nonuniform weighted means in NumPy/xarray
formats. Existing animation suites cover backends, lifecycle ownership, bounded
point blocks, supplied target reuse, GIF/MP4 output and Sequential plugins.
