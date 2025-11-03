import numpy as np
from xarray import Dataset, SerializationWarning
from pathlib import Path
import warnings

import foxes.variables as FV


def compute_scale_and_offset(min, max, n, hasnan=True):
    """
    Computes scale_factor and add_offset for packing data
    into n-bit integers.
    
    Parameters
    ----------
    min: float
        Minimum value of the data
    max: float
        Maximum value of the data
    n: int
        Number of bits for packing
    hasnan: bool
        NaN present in the data
    
    Returns
    -------
    scale_factor: float
        The scale factor
    add_offset: float
        The add offset
    fill_value: float   
        The fill value for NaN

    Notes
    -----
    Source: https://docs.unidata.ucar.edu/nug/current/best_practices.html

    """
    if min == max:
        max = min + 1
    if hasnan:
        scale_factor = (max - min) / (2 ** n - 2)
        add_offset = 0.5 * (max + min)
        fill_value = -2**(n - 1)
    else:
        scale_factor = (max - min) / (2 ** n - 1)
        add_offset = min + 2 ** (n - 1) * scale_factor
        fill_value = None
    return scale_factor, add_offset, fill_value

def pack_value(unpacked_value, scale_factor, add_offset, dtype, fill_value):
    """
    Pack a floating point value into an integer representation.
    
    Parameters
    ----------
    unpacked_value: float or np.ndarray
        The floating point value(s) to be packed
    scale_factor: float
        The scale factor
    add_offset: float
        The add offset
    dtype: numpy.dtype
        The dtype of packed values
    fill_value: float   
        The fill value for NaN

    Returns
    -------
    packed_value: int or np.ndarray
        The packed integer value(s)

    :group: utils

    """
    if fill_value is None:
        return np.floor((unpacked_value - add_offset) / scale_factor).astype(dtype)
    else:
        return np.where(
            np.isnan(unpacked_value),
            fill_value,
            np.floor((unpacked_value - add_offset) / scale_factor)
        ).astype(dtype)

def unpack_value(packed_value, scale_factor, add_offset, fill_value):
    """
    Unpack an integer representation back into a floating point value.

    Parameters
    ----------
    packed_value: int or np.ndarray
        The packed integer value(s) to be unpacked
    scale_factor: float
        The scale factor
    add_offset: float
        The add offset
    fill_value: float   
        The fill value for NaN

    Returns
    -------
    unpacked_value: float or np.ndarray
        The unpacked floating point value(s)
    
    :group: utils

    """
    if fill_value is None:
        return (packed_value * scale_factor + add_offset).astype(scale_factor.dtype)
    else:
        return np.where(
            packed_value==fill_value,
            np.nan,
            packed_value * scale_factor + add_offset
        ).astype(scale_factor.dtype)

def get_encoding(data, complevel=5):
    """
    Get the encoding parameters for a numpy array.

    Parameters
    ----------
    data: np.ndarray
        The numpy array for which to get the encoding information.
    complevel: int
        The compression level (1-9)

    Returns
    -------
    encoding: dict
        The encoding information of the numpy array.

    :group: utils

    """
    enc = {"zlib": True, "complevel": complevel}
    if np.issubdtype(data.dtype, np.integer):
        for t in [np.int8, np.uint8, np.int16, np.uint16, np.int32, np.uint32]:
            if np.all(data==data.astype(t)):
                enc["dtype"] = t.__name__
    elif np.issubdtype(data.dtype, np.floating):
        min = np.min(data)
        max = np.max(data)
        hasnan = np.any(np.isnan(data))
        for t, n in zip([np.int8, np.int16], [8, 16]):
            scale_factor, add_offset, fill_value = compute_scale_and_offset(min, max, n, hasnan)
            packed = pack_value(data, scale_factor, add_offset, t, fill_value)
            unpacked = unpack_value(packed, scale_factor, add_offset, fill_value)
            try:
                np.testing.assert_allclose(data, unpacked, atol=scale_factor)
                enc["dtype"] = t.__name__
                enc["scale_factor"] = scale_factor
                enc["add_offset"] = add_offset
                enc['_FillValue'] = fill_value
                break
            except AssertionError:
                continue
    return enc

def write_nc(
    ds,
    fpath,
    round={},
    complevel=5,
    nc_engine="netcdf4",
    verbosity=1,
    **kwargs,
):
    """
    Writes a dataset to netCDF file

    Parameters
    ----------
    fpath: str
        Path to the output file, should be nc
    round: dict or int
        The rounding digits, falling back to defaults
        if variable not found. If int, applies to all variables.
    complevel: int
        The compression level
    nc_engine: str
        The NetCDF engine to use
    verbosity: int
        The verbosity level, 0 = silent
    kwargs: dict, optional
        Additional parameters for xarray.to_netcdf

    :group: utils

    """

    def _round(x, v, d):
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

    enc = {}
    fpath = Path(fpath)
    if round is not None:
        crds = {}
        for v, x in ds.coords.items():
            if isinstance(round, int):
                d = round
            else:
                d = round.get(v, FV.get_default_digits(v))
            crds[v] = _round(x.to_numpy(), v, d)
            enc[v] = get_encoding(crds[v], complevel=complevel)
            #print("WRITENC ENC",v, enc[v])
        dvrs = {}
        for v, x in ds.data_vars.items():
            if isinstance(round, int):
                d = round
            else:
                d = round.get(v, FV.get_default_digits(v))
            if v != FV.WEIGHT:
                dvrs[v] = (x.dims, _round(x.to_numpy(), v, d))
            else:
                dvrs[v] = (x.dims, x.to_numpy())
            enc[v] = get_encoding(dvrs[v][1], complevel=complevel)
            #print("WRITENC ENC",v, enc[v])
        ds = Dataset(coords=crds, data_vars=dvrs)

    if verbosity > 0:
        print("Writing file", fpath)

    kw = dict(encoding=enc, engine=nc_engine)
    kw.update(kwargs)

    # silencing a warning about _FillValue = None
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=SerializationWarning)
        ds.to_netcdf(fpath, **kw)
