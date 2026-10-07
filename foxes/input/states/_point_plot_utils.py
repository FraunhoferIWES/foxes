from numbers import Integral
from typing import Any

import numpy as np
from matplotlib.cbook import normalize_kwargs
from matplotlib.collections import Collection
from matplotlib.lines import Line2D


def point_plot_stride(stride: int, parameter_name: str) -> int:
    """Validate a positive per-axis stride for diagnostic points only."""
    if isinstance(stride, bool) or not isinstance(stride, Integral):
        raise TypeError(f"{parameter_name} must be a positive integer")
    if stride < 1:
        raise ValueError(f"{parameter_name} must be a positive integer")
    return int(stride)


def grid_plot_axis(values: np.ndarray, stride: int) -> np.ndarray:
    """Sample a plotting axis while preserving its final coordinate."""
    if stride == 1 or values.size == 0:
        return values
    return np.unique(np.concatenate((values[::stride], values[-1:])))


def grid_plot_points(points: np.ndarray, stride: int) -> np.ndarray:
    """Sample both horizontal axes without changing the source point array."""
    if stride == 1 or points.size == 0:
        return points
    x_values = grid_plot_axis(np.unique(points[:, 0]), stride)
    y_values = grid_plot_axis(np.unique(points[:, 1]), stride)
    selected = np.isin(points[:, 0], x_values) & np.isin(points[:, 1], y_values)
    return points[selected]


def farm_plot_pars(
    overrides: dict[str, Any] | None,
    parameter_name: str,
) -> dict[str, Any]:
    """Copy caller parameters for a point plot's farm-layout overlay."""
    if overrides is not None and not isinstance(overrides, dict):
        raise TypeError(f"{parameter_name} must be a dictionary")
    return {} if overrides is None else overrides.copy()


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
