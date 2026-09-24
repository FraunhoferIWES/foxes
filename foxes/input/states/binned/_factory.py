"""Construct topology-specific binned states from artifact metadata."""

from pathlib import Path
from typing import Any

import xarray as xr

from foxes.config import config, get_input_path

from .field_data import BinnedFieldData
from .point_cloud_data import BinnedPointCloudData


def read_binned_data(
    file_path: str | Path,
    **kwargs: Any,
) -> BinnedFieldData | BinnedPointCloudData:
    """
    Create the binned states class declared by an artifact file.

    Only the NetCDF header is inspected here. The selected states object keeps
    the original path and loads the artifact data during model initialization.

    Parameters
    ----------
    file_path
        Path to a binned NetCDF artifact.
    kwargs
        Additional arguments forwarded to :class:`BinnedFieldData` or
        :class:`BinnedPointCloudData`.

    Returns
    -------
    BinnedFieldData or BinnedPointCloudData
        States object matching the artifact's ``foxes_state_class`` attribute.

    Raises
    ------
    ValueError
        If ``foxes_state_class`` is missing or does not identify a supported
        binned states class.
    """
    with xr.open_dataset(
        get_input_path(file_path),
        engine=config.nc_engine,
    ) as data:
        class_name = data.attrs.get("foxes_state_class")

    if class_name == BinnedFieldData.__name__:
        return BinnedFieldData(file_path, **kwargs)
    if class_name == BinnedPointCloudData.__name__:
        return BinnedPointCloudData(file_path, **kwargs)
    raise ValueError(
        f"Binned data artifact '{file_path}' has unsupported "
        f"foxes_state_class {class_name!r}; expected one of: "
        f"{BinnedFieldData.__name__}, {BinnedPointCloudData.__name__}"
    )
