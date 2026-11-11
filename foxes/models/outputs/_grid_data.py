"""Convert point-result fields into coordinate-labelled NetCDF-ready slices."""

from __future__ import annotations

from typing import Any

import numpy as np
from xarray import Dataset

import foxes.constants as FC
from foxes.utils import write_nc


def _slice_fields(
    point_results: Dataset, variables: list[str], state_mean: bool | np.ndarray
) -> dict[str, np.ndarray]:
    """Select and optionally reduce point fields without normalizing weights.

    Parameters
    ----------
    point_results
        Read-only variables on ``(FC.STATE, FC.POINT)``.
    variables
        Field names in requested output order.
    state_mean
        ``False`` retains states, ``True`` takes an arithmetic mean, or an array
        of shape ``(n_states,)`` supplies literal weights.

    Returns
    -------
    fields
        New variable-to-array mapping. Unreduced arrays may share input storage;
        reduced arrays have shape ``(n_points,)``. Inputs are not modified.
    """
    fields = {}
    for variable in variables:
        values = point_results[variable].to_numpy()
        if isinstance(state_mean, np.ndarray):
            fields[variable] = np.einsum("sp,s->p", values, state_mean)
        elif state_mean:
            fields[variable] = np.mean(values, axis=0)
        else:
            fields[variable] = values
    return fields


def _slice_coordinates(
    positions: list, point_results: Dataset, state_mean: bool | np.ndarray
) -> tuple[dict, dict, list, list]:
    """Describe reversed slice axes and the existing state-coordinate policy.

    Parameters
    ----------
    positions
        x, y, z coordinates: exactly two NumPy vectors and one scalar object
        exposing ``to_numpy()`` for the fixed coordinate.
    point_results
        Read-only dataset supplying state labels.
    state_mean
        Reduction selector; only literal ``False`` retains state metadata.

    Returns
    -------
    coordinates
        Coordinate mapping, with varying axes in reversed x/y/z order.
    attributes
        Fixed-axis value and a string state label for a single unreduced state.
    lengths
        Varying-axis lengths in forward x/y/z order.
    dimensions
        Varying-axis names in reversed order.
    """
    names = ["x", "y", "z"]
    varying = [i for i, pos in enumerate(positions) if isinstance(pos, np.ndarray)]
    fixed = [i for i in range(3) if i not in varying][0]
    coordinates = {}
    attributes = {names[fixed]: positions[fixed].to_numpy()}
    if (
        FC.STATE in point_results.coords
        and isinstance(state_mean, bool)
        and not state_mean
    ):
        if point_results.sizes[FC.STATE] > 1:
            coordinates[FC.STATE] = point_results[FC.STATE].to_numpy()
        else:
            attributes[FC.STATE] = str(point_results[FC.STATE][0].to_numpy())
    coordinates.update({names[i]: positions[i] for i in reversed(varying)})
    return (
        coordinates,
        attributes,
        [len(positions[i]) for i in varying],
        list(reversed([names[i] for i in varying])),
    )


def data2xr(
    x_pos: Any,
    y_pos: Any,
    z_pos: Any,
    point_results: Dataset,
    vars: list[str] | None = None,
    state_mean: bool | np.ndarray = False,
    to_file: str | None = None,
    **kwargs: Any,
) -> Dataset:
    """Convert point results into a slice dataset and optionally write NetCDF.

    Parameters
    ----------
    x_pos
        x coordinates in metres: a vector for a varying axis or a scalar
        object exposing ``to_numpy()`` for the fixed axis.
    y_pos
        y coordinates in metres, using the same vector-or-scalar convention.
    z_pos
        z coordinates in metres, using the same vector-or-scalar convention.
    point_results
        Read-only fields with dimensions ``(FC.STATE, FC.POINT)``. Point
        ordering must match the C-order grid.
    vars
        Field names in requested output order, or ``None`` for all data fields.
    state_mean
        ``False`` retains states, ``True`` takes an arithmetic mean, or a
        ``(n_states,)`` array supplies literal, unnormalized state weights.
    to_file
        Direct NetCDF destination, or ``None`` to avoid writing.
    kwargs
        Options forwarded to ``foxes.utils.write_nc`` when writing.

    Returns
    -------
    slice_results
        New dataset with reversed varying-axis order and a fixed-axis
        attribute. Unreduced fields lead with ``FC.STATE``; a sole state label
        is also stored as an attribute. Data may share input storage.
    """
    variables = list(point_results.data_vars) if vars is None else vars
    fields = _slice_fields(point_results, variables, state_mean)
    coordinates, attributes, lengths, dimensions = _slice_coordinates(
        [x_pos, y_pos, z_pos], point_results, state_mean
    )
    data_vars = {}
    for variable, values in fields.items():
        if values.ndim == 1:
            data_vars[variable] = (
                dimensions,
                np.swapaxes(values.reshape(*lengths), 0, 1),
            )
        else:
            data_vars[variable] = (
                [FC.STATE] + dimensions,
                np.swapaxes(values.reshape(values.shape[0], *lengths), 1, 2),
            )
    dataset = Dataset(coords=coordinates, data_vars=data_vars, attrs=attributes)
    if to_file is not None:
        write_nc(dataset, to_file, **kwargs)
    return dataset
