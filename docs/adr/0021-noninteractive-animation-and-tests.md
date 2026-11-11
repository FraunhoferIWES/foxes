# ADR-0021: Noninteractive Animation Presets And Tests

- Status: Accepted
- Date: 2026-11-10
- Supersedes: The animation-preset defaults and positive display-test policy in
  [ADR-0020](0020-output-display-control.md).

## Context

The user confirmed that animation model-book presets must keep `show=False`
and that tests must never enable `show`, including mocked display checks.
Static plotting presets remain interactive for ordinary library users.

## Decision

Keep GIF, MP4, turbine-animation and chunk-animation presets noninteractive.
Direct plotting and animation constructors continue to default to `show=False`.
Static layout, flow and rose model-book presets continue to enable display.
Users may explicitly enable animation playback outside tests.

Configure every pytest model-book preset and output factory with `show=False`.
Guard the shared output display hook against enabled display during tests.
Do not use `show=True` in test code, even when Matplotlib display is mocked.
Test rendering, numerical values, frame rates, actual encoded files and figure
cleanup without opening display windows.

## Consequences And Alternatives

Named animation runs save files without blocking for a display window. The
explicit blocking playback implementation remains unchanged but is no longer
positively exercised by automated tests; this is the user-required test boundary.
Mocked positive display tests were explicitly rejected.

No production backend override or test-specific library branch is introduced.
The test configuration changes only test-owned model books and uses a temporary
guard; ordinary library defaults remain intact.

## Verification

Noninteractive tests cover direct construction, configured presets and named
execution, threaded rendering, numerical modes, no-figure suppression, frame
rates, real GIF contents and figure cleanup. Source checks reject enabled
`show` settings in tests.
