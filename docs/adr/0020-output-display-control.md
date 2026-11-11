# ADR-0020: Explicit Output Display Control

- Status: Accepted
- Date: 2026-11-10

## Context

Plotting outputs create figures or save animations but need an explicit way to
open them through algorithm execution. Library construction and model-book
interactive selection have different intended defaults. The user confirmed
that model-book plotting presets should display and that animations should
play after saving.

## Decision

Add a lightweight `show` flag, defaulting to `False` on plotting constructors.
Enable it on model-book plotting presets and factories, including flow/turbine
and chunk animations. Users can disable it on a registered model before running.

Only parent-side rendering invokes blocking Matplotlib display, after any
configured file is saved. Keep animation references local while displaying at
their configured frame rate, and close transient animation/writer figures in
cleanup boundaries. Returned static figures and numerical schemas are unchanged.
No scientific arrays, figures or animations are added to model state.

Numerical modes and data-only operations do not display. `nofig=True` suppresses
figure/animation generation. Frame generators remain composable and do not
force display. Retain the caller's Matplotlib backend and its diagnostic
warnings; headless callers disable `show` explicitly.

## Consequences And Alternatives

Named plotting selections are interactive by default and can block until users
close their windows. Direct construction preserves non-displaying behavior.
Nonblocking animation display was rejected because animation references and
figure ownership would have to outlive finalization. Changing the global
backend or silently skipping unavailable displays was also rejected.

## Verification

Synthetic tests cover constructor/preset/factory defaults, named execution,
parent-thread display, suppression, unchanged values, save-before-show ordering,
animation frame rates, real GIF output and cleanup after display failures.
