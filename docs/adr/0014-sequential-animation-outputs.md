# ADR-0014: Adapt animation outputs to Sequential iteration

## Status

Accepted.

## Context

Sequential wake models need current iterator state and wake history to evaluate
points. Replaying a completed farm dataset cannot reproduce these calculations.
Examples and notebooks should configure the adopted animation models instead
of maintaining separate plotting and encoding implementations.

## Decision

Use `SeqAnimationOutputPlugin` as a lifecycle adapter for `FlowAnimation` and
`FlowTurbineAnimation`. Resolve models through the algorithm's model book,
evaluate points in each selected step through the normal point engine, and keep
numerical buffers in the plugin. Render using shared output-model hooks after
the iterator completes. Reuse existing wake-debug overlays through isolated
generator histories.

Keep Sequential's `outputs` argument reserved for physical fields. The adapter
rejects non-animation outputs. Chunk selections identify iterator steps, with
target chunk index zero. State selections retain full input-state semantics.

## Consequences

Sequential execution and wake/controller calculations remain state-wise; engine
point limits bound scientific evaluation but not retained animation memory.
Abandoning iteration does not write files. Writers propagate failures and close
figures. Model-book filename suffix rules apply identically to all workflows.
Examples remove their old animation flags; notebooks display saved GIF files.
Fractional positive finite frame rates preserve existing notebook timing.

## Alternatives

- Recalculate points after iteration: rejected because wake history is no longer
  available.
- Run ordinary farm chunk outputs for every step: rejected because that changes
  Sequential's collector contract and finalization timing.
- Duplicate animation renderers in examples/plugins: rejected because settings,
  filename rules and writer cleanup would drift.

## Verification

Synthetic tests cover step-local numerical parity, selected states/chunks,
actual GIF/MP4 output, turbine traces, debug histories and explicit errors.
Example CLI runs and full animation-notebook execution exercise public workflows.
