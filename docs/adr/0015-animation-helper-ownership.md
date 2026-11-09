# ADR-0015: Separate animation composition and Sequential plugins

## Status

Accepted.

## Context

Adopted flow animations still used composition and sequential debug helpers from
the legacy `foxes.output` namespace. Their new Sequential adapter also lived
there, obscuring which modules belong to the adopted animation workflows.

## Decision

Keep the animation composer in `foxes.models.outputs.flow_animations` and export
`Animator` from `foxes.models.outputs`.
Move all Sequential plugins (`seq_wake_debug_plugin.py`,
`seq_animation_output.py` and `seq_flow_ani_plugin.py`) to
`foxes.models.seq_plugins`. Update all maintained imports, examples and
notebooks. Remove the old modules and exports without aliases.

Helpers retain their existing roles; they do not become standalone chunk-output
models merely by moving packages.

## Consequences

Users import composition from `foxes.models.outputs` and iterator plugins from
`foxes.models.seq_plugins`. Animation output,
debug overlays and Sequential iteration behavior remain unchanged.
Wake-model checks import algorithm classes when called rather than during
package initialization, preventing cycles between models and algorithms.

## Alternatives

Keeping aliases or duplicate modules would preserve the split ownership and
conflict with forward-only development. Removing helpers would break the
adopted renderers and debug workflows. Nesting plugins beneath outputs would
misrepresent their distinct iterator lifecycle.

## Verification

Namespace assertions, real GIF/MP4 output tests, Sequential debug tests and
execution of the updated notebook verify the new ownership and preserved behavior.
