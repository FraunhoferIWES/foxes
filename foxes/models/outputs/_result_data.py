"""Result-payload assembly, state selection and farm dataset serialization."""

from __future__ import annotations

from collections.abc import Sequence
from pathlib import Path
from typing import Any, TypedDict

import numpy as np
from xarray import Dataset

import foxes.constants as FC
import foxes.variables as FV
from foxes.config import config
from foxes.utils import write_nc


class _Results(TypedDict):
    """Chunk coordinates, dimensioned field arrays and dataset attributes."""

    coords: dict[str, np.ndarray]
    data_vars: dict[str, tuple[tuple[str, ...], np.ndarray]]
    attrs: dict[str, Any]


def _combine_chunks(chunks: list[_Results]) -> _Results:
    """Join ordered result payloads without mutating source chunks.

    Parameters
    ----------
    chunks
        Nonempty state-ordered payloads with matching variable names and dimensions.

    Returns
    -------
    payload
        Newly concatenated state coordinates and state-dependent arrays,
        with copied static coordinates and attributes.

    Raises
    ------
    ValueError
        If variable selections or dimensions differ between chunks.
    """
    # Validate variable schemas before concatenating state arrays.
    first = chunks[0]
    for chunk in chunks[1:]:
        if chunk["data_vars"].keys() != first["data_vars"].keys():
            raise ValueError("Results: Variable selections differ between chunks")
        if chunk["coords"].keys() != first["coords"].keys():
            raise ValueError("Results: Coordinates differ between chunks")
        for coordinate in first["coords"]:
            if coordinate != FC.STATE and not np.array_equal(
                chunk["coords"][coordinate], first["coords"][coordinate]
            ):
                raise ValueError(
                    f"Results: Coordinate '{coordinate}' differs between chunks"
                )
        for variable, (dims, _) in chunk["data_vars"].items():
            if dims != first["data_vars"][variable][0]:
                raise ValueError(
                    f"Results: Dimensions differ between chunks for '{variable}'"
                )
            if FC.STATE not in dims and not np.array_equal(
                chunk["data_vars"][variable][1],
                first["data_vars"][variable][1],
                equal_nan=True,
            ):
                raise ValueError(
                    f"Results: Static field '{variable}' differs between chunks"
                )

    # Preserve static coordinates and concatenate along the state axis.
    coords = first["coords"].copy()
    coords[FC.STATE] = np.concatenate([chunk["coords"][FC.STATE] for chunk in chunks])
    data_vars = {
        variable: (
            dims,
            np.concatenate(
                [chunk["data_vars"][variable][1] for chunk in chunks],
                axis=dims.index(FC.STATE),
            )
            if FC.STATE in dims
            else values,
        )
        for variable, (dims, values) in first["data_vars"].items()
    }
    return dict(coords=coords, data_vars=data_vars, attrs=first["attrs"].copy())


def _select_states(payload: _Results, start: int, stop: int) -> _Results:
    """Select a half-open state interval from a result payload.

    Parameters
    ----------
    payload
        Result payload with dimensioned field arrays.
    start
        Inclusive state position.
    stop
        Exclusive state position.

    Returns
    -------
    selected
        Payload containing views of selected state arrays, shared static
        coordinates and a shallow copy of the attributes.
    """
    # Slice each state-dependent array without modifying the full payload.
    return dict(
        coords={
            name: values[start:stop] if name == FC.STATE else values
            for name, values in payload["coords"].items()
        },
        data_vars={
            name: (
                dims,
                values[
                    tuple(
                        slice(start, stop) if dim == FC.STATE else slice(None)
                        for dim in dims
                    )
                ],
            )
            for name, (dims, values) in payload["data_vars"].items()
        },
        attrs=payload["attrs"].copy(),
    )


def _create_dataset(
    payload: _Results,
    turbine_names: Sequence[str] | None,
    wind_farm_list: Sequence[str | None] | None,
    cluster_list: Sequence[str | None] | None,
) -> Dataset:
    """Assemble farm fields, metadata and compact uniform state weights.

    Parameters
    ----------
    payload
        Farm coordinates, dimensioned fields and attributes to assemble.
    turbine_names
        Names in turbine order; ``None`` omits the metadata field.
    wind_farm_list
        Wind-farm memberships in turbine order; absent entries are allowed.
    cluster_list
        Cluster memberships in turbine order; absent entries are allowed.

    Returns
    -------
    farm_results
        Dataset with state/turbine dimensions and requested nonempty metadata.
        Turbine-uniform weights are reduced to ``(FC.STATE,)`` and turbine
        order uses the configured integer dtype. Input arrays may be shared.
    """
    # Create the dataset and attach only requested, populated metadata.
    dataset = Dataset(
        data_vars=payload["data_vars"],
        coords=payload["coords"],
        attrs=payload["attrs"],
    )
    for name, values in [
        (FC.TNAME, turbine_names),
        (FC.FARM, wind_farm_list),
        (FC.CLUSTER, cluster_list),
    ]:
        if (
            FC.TURBINE in dataset.dims
            and values is not None
            and any(value is not None for value in values)
        ):
            dataset[name] = ((FC.TURBINE,), values)

    # Compress weights only after all states for this dataset are available.
    if FV.WEIGHT in dataset and dataset[FV.WEIGHT].dims == (FC.STATE, FC.TURBINE):
        weights = dataset[FV.WEIGHT].to_numpy()
        if weights.shape[1] > 0 and np.all(weights == weights[:, :1]):
            dataset[FV.WEIGHT] = ((FC.STATE,), weights[:, 0])
    if FV.ORDER in dataset:
        dataset[FV.ORDER] = dataset[FV.ORDER].astype(config.dtype_int)
    return dataset


def _data_to_nc(
    fpath: Path,
    payload: _Results,
    turbine_names: Sequence[str] | None,
    wind_farm_list: Sequence[str | None] | None,
    cluster_list: Sequence[str | None] | None,
    write_nc_pars: dict[str, Any],
) -> None:
    """Construct and write one farm dataset in the engine's worker context.

    Parameters
    ----------
    fpath
        NetCDF destination path.
    payload
        Dimensioned farm arrays and metadata to serialize.
    turbine_names
        Turbine names, or ``None`` to omit them.
    wind_farm_list
        Wind-farm memberships, or ``None`` to omit them.
    cluster_list
        Cluster memberships, or ``None`` to omit them.
    write_nc_pars
        Additional options forwarded to ``write_nc``.
    """
    # Use the same dataset construction for all writer variants.
    dataset = _create_dataset(payload, turbine_names, wind_farm_list, cluster_list)
    write_nc(ds=dataset, fpath=fpath, **write_nc_pars)
