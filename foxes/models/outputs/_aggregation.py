"""NumPy sufficient statistics and physical result aggregation."""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

import foxes.constants as FC
import foxes.variables as FV

from ._result_data import _Results

if TYPE_CHECKING:
    from foxes.core import Algorithm


_EXTENSIVE = {FV.P, FV.CAP, FV.MAX_P, FV.YLD, FV.P75, FV.P90}
_DIRECTIONS = {FV.WD, FV.YAW}
_MASS = "__aggregation_mass"


def _ratio_inputs(variable: str) -> tuple[str, str] | None:
    base = FV.amb2var.get(variable, variable)
    power = FV.AMB_P if variable in FV.amb2var else FV.P
    if base == FV.EFF:
        return power, FV.AMB_P
    if base == FV.CAPF:
        return power, FV.CAP
    return None


def _requirements(
    variables: list[str] | None,
    aggregate: str | None,
    aggregate_states: bool,
    ambient_vars: bool = True,
    waked_vars: bool = True,
) -> list[str] | None:
    """Request ratio inputs and state weights without expanding unrestricted fields."""
    if variables is None:
        return variables
    variables = [
        v
        for v in variables
        if (ambient_vars or v not in FV.amb2var) and (waked_vars or v not in FV.var2amb)
    ]
    if aggregate is None and not aggregate_states:
        return variables
    required = list(variables)
    for variable in variables:
        inputs = _ratio_inputs(variable)
        if inputs is not None:
            required.extend(inputs)
    if aggregate_states:
        required.append(FV.WEIGHT)
    return list(dict.fromkeys(required))


def _angle(sine: np.ndarray, cosine: np.ndarray) -> np.ndarray:
    """Return circular means in degrees, or NaN for cancelling directions."""
    angle = np.mod(np.rad2deg(np.arctan2(sine, cosine)), 360.0)
    angle = np.where(np.isclose(angle, 360.0), 0.0, angle)
    return np.where(np.hypot(sine, cosine) > 1e-12, angle, np.nan)


def _state_statistics(payload: _Results) -> _Results:
    """Remove the state axis on workers, retaining per-entity weighted statistics."""
    fields = payload["data_vars"]
    if FV.WEIGHT not in fields:
        raise ValueError("State aggregation requires state weights")
    weight_dims, weights = fields[FV.WEIGHT]
    if FC.STATE not in weight_dims:
        raise ValueError("State weights require a state dimension")
    if np.any(~np.isfinite(weights) | (weights < 0)):
        raise ValueError("State weights must be finite and nonnegative")
    axis = weight_dims.index(FC.STATE)
    dims = tuple(d for d in weight_dims if d != FC.STATE)
    stats = {_MASS: (dims, np.sum(weights, axis=axis))}
    for variable, (dimensions, values) in fields.items():
        if variable in (FV.ORDER, FV.WEIGHT) or _ratio_inputs(variable) is not None:
            continue
        if dimensions != weight_dims:
            raise ValueError(
                f"State aggregation: incompatible dimensions for '{variable}'"
            )
        base = FV.amb2var.get(variable, variable)
        arrays = (
            [
                (f"__sin_{variable}", np.sin(np.deg2rad(values))),
                (f"__cos_{variable}", np.cos(np.deg2rad(values))),
            ]
            if base in _DIRECTIONS
            else [(variable, values)]
        )
        for name, array in arrays:
            # Zero-weight NaNs must not enter the weighted sums.
            weighted = np.zeros_like(array, dtype=float)
            np.multiply(array, weights, out=weighted, where=weights != 0)
            stats[name] = (dims, np.sum(weighted, axis=axis))
    return dict(
        coords={d: v for d, v in payload["coords"].items() if d != FC.STATE},
        data_vars=stats,
        attrs={
            **payload["attrs"],
            "n_states": len(payload["coords"][FC.STATE]),
            "__aggregation_fields": list(fields),
        },
    )


def _merge_statistics(chunks: list[_Results]) -> _Results:
    """Merge state-reduced chunks without reconstructing any state history."""
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk["data_vars"].keys() != first["data_vars"].keys():
            raise ValueError("State aggregation: chunk fields differ")
        if chunk["coords"].keys() != first["coords"].keys():
            raise ValueError("State aggregation: chunk coordinates differ")
        for name, values in first["coords"].items():
            if not np.array_equal(
                values, chunk["coords"][name], equal_nan=values.dtype.kind in "fcMm"
            ):
                raise ValueError(f"State aggregation: coordinate '{name}' differs")
        for name, (dims, values) in first["data_vars"].items():
            dimensions, array = chunk["data_vars"][name]
            if dims != dimensions or values.shape != array.shape:
                raise ValueError(f"State aggregation: dimensions differ for '{name}'")
    return dict(
        coords=first["coords"].copy(),
        data_vars={
            name: (dims, np.sum([c["data_vars"][name][1] for c in chunks], axis=0))
            for name, (dims, _) in first["data_vars"].items()
        },
        attrs={
            **first["attrs"],
            **{
                name: sum(c["attrs"][name] for c in chunks)
                for name in ("n_points", "n_states")
                if name in first["attrs"]
                and (
                    name == "n_points"
                    and FC.STATE in first["coords"]
                    or name == "n_states"
                    and FC.STATE not in first["coords"]
                )
            },
        },
    )


def _finish_states(payload: _Results) -> _Results:
    """Normalize each turbine/point independently, including partial state selections."""
    fields = payload["data_vars"]
    dims, mass = fields[_MASS]
    if np.any(~np.isfinite(mass) | (mass <= 0)):
        raise ValueError(
            "State aggregation requires positive finite total weight per turbine/point"
        )
    values = {FV.WEIGHT: (dims, np.ones_like(mass))}
    # Normalize per entity before grouping, including tiny partial-state masses.
    for name, (dimensions, total) in fields.items():
        if name == _MASS or name.startswith("__cos_"):
            continue
        if name.startswith("__sin_"):
            variable = name[len("__sin_") :]
            values[variable] = (
                dimensions,
                _angle(total / mass, fields[f"__cos_{variable}"][1] / mass),
            )
        else:
            values[name] = (dimensions, total / mass)
    return dict(
        coords=payload["coords"].copy(), data_vars=values, attrs=payload["attrs"].copy()
    )


def _reduce_group(
    name: str, values: np.ndarray, axis: int, farm: bool
) -> np.ndarray | np.generic:
    """Reduce one group's physical field without changing other dimensions."""
    base = FV.amb2var.get(name, name)
    if base in _DIRECTIONS:
        radians = np.deg2rad(values)
        return _angle(
            np.mean(np.sin(radians), axis=axis), np.mean(np.cos(radians), axis=axis)
        )
    if farm and base in _EXTENSIVE:
        return np.sum(values, axis=axis)
    return np.mean(values, axis=axis)


def _spatial(
    payload: _Results,
    dimension: str,
    groups: list[np.ndarray],
    group_dimension: str | None = None,
    labels: list[str] | None = None,
    farm: bool = True,
) -> _Results:
    """Sum extensive fields and average intensive fields over complete entity groups."""
    fields = {}
    for name, (dims, values) in payload["data_vars"].items():
        if name == FV.ORDER or _ratio_inputs(name) is not None:
            continue
        if dimension not in dims:
            fields[name] = (dims, values)
            continue
        axis = dims.index(dimension)
        reduced = [
            _reduce_group(name, np.take(values, indices, axis=axis), axis, farm)
            for indices in groups
        ]
        dimensions = (
            tuple(group_dimension if d == dimension else d for d in dims)
            if group_dimension
            else tuple(d for d in dims if d != dimension)
        )
        fields[name] = (
            dimensions,
            np.stack(reduced, axis=axis) if group_dimension else reduced[0],
        )
    coords = {d: v for d, v in payload["coords"].items() if d != dimension}
    if group_dimension is not None:
        coords[group_dimension] = np.asarray(labels)
    return dict(
        coords=coords,
        data_vars=fields,
        attrs={
            "__aggregation_fields": list(payload["data_vars"]),
            **payload["attrs"],
            "n_turbines" if farm else "n_points": len(payload["coords"][dimension]),
        },
    )


def _farm_groups(
    algo: Algorithm,
    aggregate: str,
    turbines: np.ndarray,
) -> tuple[list[np.ndarray], str | None, list[str] | None]:
    """Resolve original-order memberships after output power normalization."""
    if aggregate == "all":
        groups, dimension, labels = [np.arange(len(turbines))], None, None
    else:
        memberships = (
            algo.farm.wind_farm_list
            if aggregate == "wind_farm"
            else algo.farm.cluster_list
        )
        memberships_selected = [memberships[int(i)] for i in turbines]
        if "<unassigned>" in memberships_selected:
            raise ValueError("Aggregation reserves the group name '<unassigned>'")
        names = [
            name if name is not None else "<unassigned>"
            for name in memberships_selected
        ]
        labels = list(dict.fromkeys(names))
        groups = [np.flatnonzero(np.asarray(names) == name) for name in labels]
        dimension = FC.FARM if aggregate == "wind_farm" else FC.CLUSTER
    return groups, dimension, labels


def _finish(
    payload: _Results, variables: list[str] | None, aggregate_states: bool
) -> _Results:
    """Recompute ratios from reduced power, then remove internal input requirements."""
    fields = payload["data_vars"]
    original = payload["attrs"].pop("__aggregation_fields", list(fields))
    requested = original if variables is None else variables
    for variable in requested:
        inputs = _ratio_inputs(variable)
        if inputs is None:
            continue
        numerator, denominator = inputs
        if numerator not in fields or denominator not in fields:
            raise ValueError(f"Aggregation of '{variable}' requires {inputs}")
        dims, power = fields[numerator]
        ambient = fields[denominator][1]
        ratio = np.full_like(power, np.inf, dtype=float)
        np.divide(power, ambient, out=ratio, where=ambient != 0)
        ratio = np.where(np.isnan(power), np.nan, ratio)
        ratio = np.where(power == 0, 0.0, ratio)
        fields[variable] = (dims, ratio)
    payload["data_vars"] = {
        v: fields[v] for v in requested if v in fields and v != FV.ORDER
    }
    if aggregate_states:
        payload["attrs"]["aggregate_states"] = 1
    return payload


def _point_spatial_statistics(payload: _Results) -> _Results:
    """Reduce point tiles to additive sums/counts rather than retaining target arrays."""
    fields = {}
    for name, (dims, values) in payload["data_vars"].items():
        if FC.TARGET not in dims:
            raise ValueError(
                f"Point aggregation requires target-dependent field '{name}'"
            )
        axis = dims.index(FC.TARGET)
        dimensions = tuple(d for d in dims if d != FC.TARGET)
        base = FV.amb2var.get(name, name)
        arrays = (
            [
                (f"__sin_{name}", np.sin(np.deg2rad(values))),
                (f"__cos_{name}", np.cos(np.deg2rad(values))),
            ]
            if base in _DIRECTIONS
            else [(name, values)]
        )
        for key, array in arrays:
            fields[key] = (dimensions, np.sum(array, axis=axis))
    fields[_MASS] = (
        dimensions,
        np.full_like(
            next(iter(fields.values()))[1],
            len(payload["coords"][FC.TARGET]),
            dtype=float,
        ),
    )
    return dict(
        coords={d: v for d, v in payload["coords"].items() if d != FC.TARGET},
        data_vars=fields,
        attrs={
            **payload["attrs"],
            "n_points": len(payload["coords"][FC.TARGET]),
            "__aggregation_fields": list(payload["data_vars"]),
        },
    )
