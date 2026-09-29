from typing import Any

from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import Collection
from matplotlib.lines import Line2D


def line_plot_pars(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
    parameter_name: str,
) -> dict[str, Any]:
    """Merge line-plot defaults with normalized caller parameters."""
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError(f"{parameter_name} must be a dictionary")
    pars = normalize_kwargs(defaults, Line2D)
    if overrides is not None:
        pars.update(normalize_kwargs(overrides, Line2D))
    return pars


def scatter_plot_pars(
    defaults: dict[str, Any],
    overrides: dict[str, Any] | None,
    parameter_name: str,
) -> dict[str, Any]:
    """Merge scatter defaults with normalized caller parameters."""
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError(f"{parameter_name} must be a dictionary")
    pars = normalize_kwargs(defaults, Collection)
    if overrides is not None:
        normalized = normalize_kwargs(overrides, Collection)
        if "c" in normalized:
            pars.pop("color", None)
        pars.update(normalized)
    return pars
