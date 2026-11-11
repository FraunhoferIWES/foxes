# ADR-0023: In-Process, Isolated Example Runs

- Status: Accepted
- Date: 2026-11-11
- Supersedes: None

## Context

The aggregate runner previously scanned all README commands, launched CLI and
cleanup subprocesses, repeated examples, and excluded WindIO from tests.
Examples need consistent output selection and fast, artifact-free automation
without losing ordinary interactive or real-data workflows.

## Decision

All scripts accept `-v`/`--verbosity`, defaulting to level-two reports;
the aggregate runner passes `-v 0` by default. Plot variables use `--var`,
reserving `-v` for verbosity. Level two adds reports and level three enables
library diagnostics. Explicit command verbosity takes precedence, and failures
and missing-output messages remain visible without redirecting output streams.
Levels two and three print every collected farm-results dataset once, before
field-specific report guards, including comparison and optional benchmark runs.
Levels zero and one suppress datasets. YAML/WindIO entry points receive the
unreduced example level and reduce scientific component verbosity themselves.
Writer-only and non-farm outputs are not reported as farm datasets.
Animation diagnostics use the selected script level directly, rather than the
reduced scientific-library level: creation at one; writing, successful
completion, frame counts and encoder settings at two; and individual-frame
progress at three.

Every example script exposes `main(argv=None)` while retaining CLI launch.
Tutorials remain monolithic, explicitly excepted from the function-size guideline.
Each README declares exactly one normal and one quick aggregate command.
`examples/run_all.py` calls scripts and YAML/WindIO APIs directly, once per
selected directory, using a temporary copy and restoring process/FOXES state.

`--quick` supplies bounded public or synthetic inputs and local engines.
ICON-DREAM uses synthetic ICON/EWW files; Slurm retains its normal scheduler
default but selects a local engine in quick mode. Quick WindIO inlines bundled
includes and exercises translation/execution without its optional YAML loader.
`--nofig` remains independently selectable and suppresses display without
disabling explicitly requested file outputs.

Every script supports `-o/--outputs`, with defaults matching its purpose and
explicit messages when reporting requires unavailable collected fields.
Sequential's existing iterator adapter supports farm collection and animations,
not arbitrary farm writers.

## Compatibility And Migration

The runner no longer scans arbitrary README commands or runs cleanup scripts.
All maintained example READMEs and smoke tests move to explicit command markers.
Aggregate runs discard temporary results; direct script runs retain outputs.
Scientific model APIs, dimensions, and result schemas are unchanged.

## Consequences

- Every documented example has bounded, deterministic smoke coverage once.
- Failures propagate rather than being hidden in successful aggregate exit codes.
- Runner tests verify cleanup, configuration/UTM restoration, and no display.
- Normal external-input/scheduler runs still require their real infrastructure;
  quick coverage does not certify external services or the optional WindIO loader.
- Transactional cleanup keeps the runner's isolation context together even
  though it exceeds the usual preferred function size.

## Verification

- [Example smoke tests](../../tests/3_examples/test_examples.py).
- [CLI and isolation contracts](../../tests/3_examples/test_output_selection.py).
- Existing output-model tests retain numerical collection, writing, and encoding
  coverage without repeating whole example scripts.

## Alternatives Considered

- Subprocesses plus cleanup scripts: rejected for fragile error propagation and
  artifact handling.
- Running every README command: rejected because optional tutorial variants
  repeat examples and may depend on previous generated files.
- Silently skipping external integrations: rejected in favor of explicit,
  bounded local equivalents.
