---
description: "Use when creating or modifying user interfaces, visual styles, design tokens, charts, images, or Fraunhofer IWES brand assets."
applyTo: "**/*.{css,scss,sass,less,html,htm,svg,ts,jsx,tsx,vue,svelte}"
---

# Fraunhofer IWES UI Instructions

This file is present, so the corporate design applies. A free-design project
deletes it and `docs/fraunhofer-design/` instead — see `AGENTS.md`, "UI Design
Policy", and establish that policy before the first UI change.

FOXES has no browser application frontend. Sphinx documentation adapts the
authoritative tokens through `docs/source/_static/iwes-tokens.css` and applies
them through `docs/source/_static/iwes.css`; the consistency test at
`tests/0_consistency/test_docs_design_tokens.py` prevents drift. Do not
introduce an application stack or another token copy as part of an unrelated
scientific plot change. For Python plotting code under `foxes/output/`,
examples, or notebooks, read the chart and FOXES-specific guidance in
`docs/fraunhofer-design/ui-guidelines.md`; Python is intentionally not included
in this file's broad `applyTo` glob because most package changes are non-visual.

Read and follow `docs/fraunhofer-design/ui-guidelines.md` before making UI,
visual-design, or brand changes. Token values come from
`docs/fraunhofer-design/design-tokens.json`, which is authoritative: a group's
`$type` applies to every token below it, and a value without `reviewStatus` is
approved. Once the project has a token adapter for its technology and has
recorded it in `docs/architecture.md`, application code reads values from that
adapter, which is either generated or covered by a test that compares it against
the token document.

Do not duplicate, alter, or bypass the corporate policy in this file. Put
FOXES-specific visual requirements in the design guide or architecture record.
