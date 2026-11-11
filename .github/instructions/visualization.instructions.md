---
description: "Use when creating or modifying FOXES scientific plots, animations, tutorial notebooks, or documentation appearance."
applyTo: "notebooks/**/*.ipynb,foxes/models/outputs/**/*.py,docs/source/**/*.css"
---

# FOXES Visualization Conventions

FOXES uses free design under [ADR-0030](../../docs/adr/0030-free-design.md).
No corporate palette, typography, design-token system, or corporate
accessibility rules apply.

- Use ordinary scientific plotting defaults unless a particular example needs
  another style. There is no required project-wide palette or font.
- Preserve scientific direction conventions, physical units, and the meaning of
  compared quantities. Give plots descriptive labels and nearby interpretation.
- Preserve caller-supplied axes and styling. Do not configure global Matplotlib
  defaults when importing the library.
- Keep tutorial inputs public or synthetic, computations bounded, and file
  creation explicit. Explain returned datasets, mappings, and file names.

Free design does not certify external publication. Publishing outside the
project's own context requires a design and accessibility review, which may
require extensive restyling.
