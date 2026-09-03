import numpy as np
from xarray import DataArray, Dataset, SerializationWarning
from pathlib import Path
import warnings
from typing import Any
from collections.abc import Hashable

import foxes.variables as FV


def compute_scale_and_offset(
    min: float, max: float, n: int, hasnan: bool = True
) -> tuple[float, float, float | None]:
    """
    Computes scale_factor and add_offset for packing data
    into n-bit integers.

    Parameters
    ----------
    min
        Minimum value of the data
    max
        Maximum value of the data
    n
        Number of bits for packing
    hasnan
        NaN present in the data

    Returns
    -------
    scale_factor
        The scale factor
    add_offset
        The add offset
    fill_value
        The fill value for NaN

    Notes
    -----
    Source: https://docs.unidata.ucar.edu/nug/current/best_practices.html

    """
    if min == max:
        max = min + 1
    if hasnan:
        scale_factor = (max - min) / (2**n - 2)
        add_offset = 0.5 * (max + min)
        fill_value = -(2 ** (n - 1))
    else:
        scale_factor = (max - min) / (2**n - 1)
        add_offset = min + 2 ** (n - 1) * scale_factor
        fill_value = None
    return scale_factor, add_offset, fill_value


def pack_value(
    unpacked_value: float | np.ndarray,
    scale_factor: float,
    add_offset: float,
    dtype: np.dtype[Any] | type[Any],
    fill_value: float | None,
) -> np.ndarray:
    """
    Pack a floating point value into an integer representation.

    Parameters
    ----------
    unpacked_value
        The floating point value(s) to be packed
    scale_factor
        The scale factor
    add_offset
        The add offset
    dtype
        The dtype of packed values
    fill_value
        The fill value for NaN

    Returns
    -------
    packed_value
        The packed integer value(s)


    """
    if fill_value is None:
        return np.floor((unpacked_value - add_offset) / scale_factor).astype(dtype)
    else:
        with warnings.catch_warnings():
            warnings.simplefilter("error", category=RuntimeWarning)
            packed = np.where(
                np.isnan(unpacked_value),
                fill_value,
                np.floor((unpacked_value - add_offset) / scale_factor),
            )
            return packed.astype(dtype)


def unpack_value(
    packed_value: int | np.ndarray,
    scale_factor: float,
    add_offset: float,
    fill_value: float | None,
) -> np.ndarray:
    """
    Unpack an integer representation back into a floating point value.

    Parameters
    ----------
    packed_value
        The packed integer value(s) to be unpacked
    scale_factor
        The scale factor
    add_offset
        The add offset
    fill_value
        The fill value for NaN

    Returns
    -------
    unpacked_value
        The unpacked floating point value(s)


    """
    if fill_value is None:
        return np.asarray(packed_value * scale_factor + add_offset, dtype=np.float64)
    else:
        return np.asarray(
            np.where(
                packed_value == fill_value,
                np.nan,
                packed_value * scale_factor + add_offset,
            ),
            dtype=np.float64,
        )


def get_encoding(
    data: np.ndarray, complevel: int = 5, pack: bool = True
) -> dict[str, Any]:
    """
    Get the encoding parameters for a numpy array.

    Parameters
    ----------
        data
        The numpy array for which to get the encoding information.
    complevel
        The compression level (1-9)
    pack
        Whether to pack data using scale_factor and add_offset

    Returns
    -------
    encoding
        The encoding information of the numpy array.


    """
    enc: dict[str, Any] = {"zlib": True, "complevel": complevel}
    if pack:
        if np.issubdtype(data.dtype, np.integer):
            for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]:
                if np.all(data == data.astype(t)):
                    enc["dtype"] = t.__name__
        elif np.issubdtype(data.dtype, np.floating):
            min = np.min(data)
            max = np.max(data)
            hasnan = bool(np.any(np.isnan(data)))
            for t, n in zip([np.int8, np.int16], [8, 16]):
                scale_factor, add_offset, fill_value = compute_scale_and_offset(
                    min, max, n, hasnan
                )
                packed = pack_value(data, scale_factor, add_offset, t, fill_value)
                unpacked = unpack_value(packed, scale_factor, add_offset, fill_value)
                try:
                    np.testing.assert_allclose(data, unpacked, atol=scale_factor)
                    enc["dtype"] = t.__name__
                    enc["scale_factor"] = scale_factor
                    enc["add_offset"] = add_offset
                    enc["_FillValue"] = fill_value
                    break
                except AssertionError:
                    continue
    return enc


def write_nc(
    ds: Dataset,
    fpath: Path | str,
    round: dict[str, int] | int | None = None,
    complevel: int = 5,
    nc_engine: str | None = None,
    pack: bool = False,
    verbosity: int = 1,
    **kwargs: Any,
) -> None:
    """
    Writes a dataset to netCDF file

    Parameters
    ----------
    fpath
        Path to the output file, should be nc
    round
        The rounding digits, falling back to defaults
        if variable not found. If int, applies to all variables.
    complevel
        The compression level
    nc_engine
        The NetCDF engine to use
    pack
        Whether to pack data using scale_factor and add_offset
    verbosity
        The verbosity level, 0 = silent
    kwargs
            Additional parameters for writing the NetCDF file.


    """
    fpath = Path(fpath)
    if nc_engine is None:
        from foxes.config import config

        nc_engine = config.nc_engine
    nc_engine = nc_engine or "netcdf4"

    def _round(x: np.ndarray, v: str, d: int | None) -> np.ndarray:
        """Helper function to round values"""
        if d is not None:
            if np.issubdtype(x.dtype, np.integer):
                return x
            elif np.issubdtype(x.dtype, np.floating):
                if verbosity > 1:
                    print(f"File {fpath.name}: Rounding {v} to {d} decimals")
                r = np.round(x, d)
                return r
        return x

    def _keep_attrs(x: DataArray, encoding: dict[str, Any]) -> dict[Hashable, Any]:
        """Attributes of a variable, minus those the encoding now owns.

        The rebuilt Dataset below is assembled from plain arrays, so without this
        every variable and coordinate attribute is lost -- units, long_name,
        cell_methods. The output then no longer says in which unit it is given, or
        whether the values are means or instantaneous.

        Keys that the computed encoding sets are dropped, because xarray refuses to
        write a variable that carries the same key in both attrs and encoding.
        """
        return {k: val for k, val in x.attrs.items() if k not in encoding}

    enc: dict[Hashable, dict[str, Any]] = {}
    if round is not None:
        crds: dict[Hashable, tuple[Any, np.ndarray, dict[Hashable, Any]]] = {}
        for v, x in ds.coords.items():
            v = str(v)
            if isinstance(round, int):
                d = round
            else:
                d = round.get(v, FV.get_default_digits(v))
            data = _round(x.to_numpy(), v, d)
            enc[v] = get_encoding(data, complevel=complevel, pack=pack)
            crds[v] = (x.dims, data, _keep_attrs(x, enc[v]))
        dvrs: dict[Hashable, tuple[Any, np.ndarray, dict[Hashable, Any]]] = {}
        for v, x in ds.data_vars.items():
            v = str(v)
            if isinstance(round, int):
                d = round
            else:
                d = round.get(v, FV.get_default_digits(v))
            if v != FV.WEIGHT:
                data = _round(x.to_numpy(), v, d)
            else:
                data = x.to_numpy()
            enc[v] = get_encoding(data, complevel=complevel, pack=pack)
            dvrs[v] = (x.dims, data, _keep_attrs(x, enc[v]))
        ds = Dataset(coords=crds, data_vars=dvrs, attrs=ds.attrs)

    if verbosity > 1:
        print(
            f"Writing file {fpath} using pack={pack}, complevel={complevel}, engine={nc_engine}"
        )
    elif verbosity > 0:
        print("Writing file", fpath)

    kw: dict[str, Any] = dict(encoding=enc, engine=nc_engine)
    kw.update(kwargs)

    # silencing a warning about _FillValue = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SerializationWarning)
        ds.to_netcdf(fpath, **kw)
