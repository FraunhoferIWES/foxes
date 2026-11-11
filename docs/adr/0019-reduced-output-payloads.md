# ADR-0019: Reduced Output Worker Payloads

- Status: Accepted
- Date: 2026-11-10

## Context

Constructor-defined outputs know their final operation before calculation.
Transporting full state frames for means, scalar evaluations, roses and layouts
wastes worker serialization and parent memory, even with bounded point chunks.
Reducing already collected frames only in finalization does not solve this.

## Decision

Eliminate dimensions on workers whenever the final result does not require
them. Mean slices transport weighted point sums. Farm evaluations transport
sums, extrema or mergeable valid counts and centred moments for population
deviations. Additive scalar farm summaries also reduce turbines; farm yield
applies individual power units before scalar reduction. Roses transport bin
numerators and denominators; layouts transport selected geometry and reduced
colour fields. Metadata-only outputs have no scientific payload.

Merge sufficient statistics globally without reconstructing eliminated axes.
Apply normalization, clipping and bin denominator thresholds only at their
established global stage. Preserve zero-weight NaN validation and subset-weight
semantics; do not average chunk means or renormalize selected state weights.
Negative positional selections retain bounded trailing candidates when the
selected global position is not yet known.

Transport static point coordinates with a singleton state axis and restore
full public dimensions during assembly. This concerns private output payloads,
not full-shaped model/data-container arrays under ADR-0013. Retain required
state-resolved fields for dataset collectors, writers and animations.

## Compatibility And Consequences

Algorithm selectors, final numerical schemas, configured scientific operations
and rendering behavior remain unchanged. Flow plots collect only their selected
field, adding wind fields for configured quivers and geometry for rotor overlays.
Models remain read-only while running; all statistics and buffers are run-local.
Per-state point calculation temporaries are still bounded by engine chunk sizes.

Collecting every raw frame for simpler finalization and averaging local means
were rejected: the former retains the memory problem and the latter changes
nonuniform-weight and unequal-chunk results.

## Verification

Synthetic tests assert returned array dimensions and bin byte bounds, alongside
parity with stateless numerical operations. Cases cover uneven chunks, allowed
NaNs, high-offset variance, turbine-dependent weights and units, negative
positions, global denominator thresholds, static/moving coordinates, optional
plot vectors/overlays and NumPy/thread/process execution.
