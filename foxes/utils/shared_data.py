from __future__ import annotations

from collections.abc import Mapping
from typing import Any

import numpy as np
import xarray as xr


def is_shareable_extra_data(value: Any) -> bool:
    """Check whether an extra-data value supports native array sharing.

    Parameters
    ----------
    value
        Candidate top-level extra-data value.

    Returns
    -------
    shareable
        Whether the value can be encoded as non-object NumPy arrays and metadata.

    """
    if isinstance(value, np.ndarray):
        return value.dtype.kind != "O"
    if isinstance(value, xr.Dataset):
        return len(value.variables) > 0 and all(
            isinstance(variable.data, np.ndarray)
            and variable.dtype.kind != "O"
            and variable.nbytes > 0
            for variable in value.variables.values()
        )
    return False


def encode_shared_extra_data(
    extra_data: Mapping[Any, Any],
) -> tuple[dict[Any, dict[str, Any]], dict[str, np.ndarray]]:
    """Encode supported top-level extra data as metadata and NumPy arrays.

    Parameters
    ----------
    extra_data
        Atomic top-level values selected for sharing.

    Returns
    -------
    metadata
        Reconstruction metadata keyed by the original extra-data keys.
    arrays
        Flat non-object arrays keyed by internal identifiers.

    """
    metadata: dict[Any, dict[str, Any]] = {}
    arrays: dict[str, np.ndarray] = {}
    for entry_index, (key, value) in enumerate(extra_data.items()):
        if not is_shareable_extra_data(value):
            metadata[key] = {"kind": "inline", "value": value}
            continue

        if isinstance(value, np.ndarray):
            array_key = f"extra_{entry_index}"
            arrays[array_key] = value
            metadata[key] = {"kind": "ndarray", "array_key": array_key}
            continue

        variables = []
        for variable_index, (name, variable) in enumerate(value.variables.items()):
            array_key = f"extra_{entry_index}_variable_{variable_index}"
            arrays[array_key] = variable.data
            variables.append(
                {
                    "name": name,
                    "dims": tuple(variable.dims),
                    "attrs": dict(variable.attrs),
                    "encoding": dict(variable.encoding),
                    "is_coord": name in value.coords,
                    "array_key": array_key,
                }
            )
        metadata[key] = {
            "kind": "xarray_dataset",
            "attrs": dict(value.attrs),
            "encoding": dict(value.encoding),
            "variables": variables,
        }
    return metadata, arrays


def decode_shared_extra_data(
    metadata: Mapping[Any, Mapping[str, Any]],
    arrays: Mapping[str, np.ndarray],
) -> dict[Any, Any]:
    """Reconstruct shared extra data from metadata and resolved arrays.

    Parameters
    ----------
    metadata
        Metadata produced by :func:`encode_shared_extra_data`.
    arrays
        Resolved arrays matching the encoded internal identifiers.

    Returns
    -------
    extra_data
        Reconstructed top-level extra-data values.

    """
    extra_data: dict[Any, Any] = {}
    for key, descriptor in metadata.items():
        kind = descriptor["kind"]
        if kind == "inline":
            extra_data[key] = descriptor["value"]
            continue
        if kind == "ndarray":
            extra_data[key] = arrays[descriptor["array_key"]]
            continue
        if kind != "xarray_dataset":
            raise ValueError(f"Unknown shared extra-data descriptor kind {kind!r}")

        coords = {}
        data_vars = {}
        variable_encodings = {}
        for variable in descriptor["variables"]:
            name = variable["name"]
            decoded = xr.Variable(
                variable["dims"],
                arrays[variable["array_key"]],
                attrs=variable["attrs"],
            )
            if variable["is_coord"]:
                coords[name] = decoded
            else:
                data_vars[name] = decoded
            variable_encodings[name] = variable["encoding"]

        dataset = xr.Dataset(
            data_vars=data_vars,
            coords=coords,
            attrs=descriptor["attrs"],
        )
        dataset.encoding = dict(descriptor["encoding"])
        for name, encoding in variable_encodings.items():
            dataset[name].encoding = dict(encoding)
        extra_data[key] = dataset
    return extra_data
